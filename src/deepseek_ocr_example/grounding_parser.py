"""
Grounding output parser for DeepSeek-OCR

Parses grounding tags (<|ref|>...<|/ref|><|det|>...<|/det|>) and
converts them to structured JSON with text content and bounding boxes.
"""
import re
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def parse_grounding_output(raw_output: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Parse grounding output containing <|ref|> and <|det|> tags.

    Args:
        raw_output: Raw OCR output with grounding tags

    Returns:
        Tuple of (parsed_elements, clean_text)
        - parsed_elements: List of dicts with 'type', 'text', 'bbox'
        - clean_text: Text with grounding tags removed
    """
    # Pattern to match <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    matches = re.findall(pattern, raw_output, re.DOTALL)

    parsed_elements = []
    clean_text = raw_output

    for match in matches:
        ref_text = match[0].strip()  # This is the actual text content
        det_coords = match[1].strip()

        try:
            # Parse coordinates: [[x1, y1, x2, y2]] or [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
            coords_list = eval(det_coords)

            # Ensure it's a list of lists
            if coords_list and isinstance(coords_list[0], list):
                bboxes = coords_list
            else:
                bboxes = [coords_list]

            # Infer type from text content
            text_lower = ref_text.lower()
            if ref_text == 'image' or 'image' in text_lower:
                element_type = 'image'
            elif any(keyword in text_lower for keyword in ['title', 'heading', 'header']):
                element_type = 'title'
            elif 'caption' in text_lower:
                element_type = 'image_caption'
            elif 'table' in text_lower or 'td' in ref_text or 'tr' in ref_text:
                element_type = 'table'
            else:
                element_type = 'text'

            element = {
                'type': element_type,
                'text': ref_text,  # The actual extracted text
                'bboxes': bboxes
            }

            parsed_elements.append(element)

        except (SyntaxError, ValueError) as e:
            # Skip malformed coordinates
            continue

    # Extract clean text by removing all grounding tags
    clean_text = re.sub(pattern, lambda m: m.group(1), raw_output)
    clean_text = clean_text.strip()

    return parsed_elements, clean_text


def normalize_coordinates(bbox: List[int], image_width: int, image_height: int) -> List[int]:
    """
    Convert normalized coordinates (0-999 scale) to pixel coordinates.

    Args:
        bbox: [x1, y1, x2, y2] in 0-999 scale
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        [x1, y1, x2, y2] in pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    x1 = int(x1 / 999 * image_width)
    y1 = int(y1 / 999 * image_height)
    x2 = int(x2 / 999 * image_width)
    y2 = int(y2 / 999 * image_height)
    return [x1, y1, x2, y2]


def draw_bounding_boxes(
    image: Image.Image,
    parsed_elements: List[Dict[str, Any]],
    output_path: Path,
    processed_image_size: Tuple[int, int] = None,
    original_image_size: Tuple[int, int] = None,
    box_color: Tuple[int, int, int] = (255, 0, 0),  # Red
    box_width: int = 3
) -> Image.Image:
    """
    Draw bounding boxes on original image for all detected elements.

    The coordinates from DeepSeek-OCR are relative to the preprocessed image.
    This function scales them back to the original image size.

    Args:
        image: Original PIL Image (will draw boxes on this)
        parsed_elements: List of parsed grounding elements
        output_path: Path to save annotated image
        processed_image_size: Size of preprocessed image (width, height)
        original_image_size: Size of original image (width, height)
        box_color: RGB color for boxes (default: red)
        box_width: Line width for boxes

    Returns:
        Annotated image
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Create semi-transparent overlay
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.load_default()
    except:
        font = None

    # Calculate scaling factors
    if processed_image_size and original_image_size:
        proc_width, proc_height = processed_image_size
        orig_width, orig_height = original_image_size
        
        # For padded preprocessing, we need to calculate the actual content area
        # The image was padded to maintain aspect ratio
        orig_aspect = orig_width / orig_height
        proc_aspect = proc_width / proc_height
        
        if orig_aspect > proc_aspect:
            # Original is wider - width matches, height is padded
            scale_x = orig_width / proc_width
            # Calculate actual content height in processed image
            actual_proc_height = proc_width / orig_aspect
            padding_top = (proc_height - actual_proc_height) / 2
            scale_y = orig_height / actual_proc_height
        else:
            # Original is taller - height matches, width is padded
            scale_y = orig_height / proc_height
            # Calculate actual content width in processed image
            actual_proc_width = proc_height * orig_aspect
            padding_left = (proc_width - actual_proc_width) / 2
            scale_x = orig_width / actual_proc_width
    else:
        # No scaling needed
        scale_x = 1.0
        scale_y = 1.0
        padding_top = 0
        padding_left = 0
        proc_width, proc_height = image.size

    for element in parsed_elements:
        element_type = element['type']
        bboxes = element['bboxes']

        # Assign color based on type
        if element_type == 'image':
            color = (0, 0, 255)  # Blue for images
        elif element_type in ['title', 'heading']:
            color = (255, 0, 0)  # Red for titles
        elif element_type == 'image_caption':
            color = (0, 255, 0)  # Green for captions
        else:
            color = box_color  # Default color

        for bbox in bboxes:
            # Step 1: Convert from 0-999 scale to preprocessed image pixel coordinates
            x1_proc = bbox[0] / 999 * proc_width
            y1_proc = bbox[1] / 999 * proc_height
            x2_proc = bbox[2] / 999 * proc_width
            y2_proc = bbox[3] / 999 * proc_height

            # Step 2: Account for padding
            if processed_image_size and original_image_size:
                if orig_aspect > proc_aspect:
                    # Width-based scaling, vertical padding
                    y1_proc -= padding_top
                    y2_proc -= padding_top
                else:
                    # Height-based scaling, horizontal padding
                    x1_proc -= padding_left
                    x2_proc -= padding_left

            # Step 3: Scale to original image size
            x1 = int(x1_proc * scale_x)
            y1 = int(y1_proc * scale_y)
            x2 = int(x2_proc * scale_x)
            y2 = int(y2_proc * scale_y)

            # Clamp to image bounds
            x1 = max(0, min(x1, img_draw.width - 1))
            y1 = max(0, min(y1, img_draw.height - 1))
            x2 = max(0, min(x2, img_draw.width - 1))
            y2 = max(0, min(y2, img_draw.height - 1))

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

            # Draw semi-transparent fill
            color_a = color + (30,)
            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0))

            # Draw label
            if font:
                label_text = element.get('text', element_type)
                if len(label_text) > 20:
                    label_text = label_text[:20] + '...'
                text_x = x1
                text_y = max(0, y1 - 20)

                # Draw label background
                try:
                    text_bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    draw.rectangle(
                        [text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
                        fill=(255, 255, 255, 200)
                    )
                    draw.text((text_x + 2, text_y + 2), label_text, font=font, fill=color)
                except:
                    pass

    # Paste overlay onto image
    img_draw.paste(overlay, (0, 0), overlay)

    # Save annotated image
    img_draw.save(output_path)

    return img_draw


def format_to_structured_json(parsed_elements: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert parsed elements to structured JSON format.

    Args:
        parsed_elements: List of parsed grounding elements
        metadata: Processing metadata

    Returns:
        Structured JSON object
    """
    # Group elements by type
    structured = {
        'elements': [],
        'metadata': metadata,
        'statistics': {
            'total_elements': len(parsed_elements),
            'element_types': {}
        }
    }

    for element in parsed_elements:
        element_type = element['type']

        # Count element types
        if element_type not in structured['statistics']['element_types']:
            structured['statistics']['element_types'][element_type] = 0
        structured['statistics']['element_types'][element_type] += 1

        # Add to elements list
        structured['elements'].append({
            'type': element_type,
            'text': element.get('text', ''),
            'bounding_boxes': element['bboxes'],
            'count': len(element['bboxes'])
        })

    return structured
