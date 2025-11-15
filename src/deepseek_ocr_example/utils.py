"""
Utility functions for image processing and output formatting
"""

import json
from pathlib import Path
from typing import Union, Dict, Any, List
from PIL import Image, ImageOps
import base64
from io import BytesIO


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """
    Load an image from file path with EXIF orientation handling

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object with correct orientation

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a valid image
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path)
        
        # Handle EXIF orientation - important for photos from cameras/phones
        corrected_image = ImageOps.exif_transpose(image)
        if corrected_image is None:
            corrected_image = image
        
        # Convert to RGB if necessary
        if corrected_image.mode != 'RGB':
            corrected_image = corrected_image.convert('RGB')
        
        return corrected_image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[tuple],
    width: int,
    height: int,
    image_size: int
) -> tuple:
    """
    Find the closest aspect ratio from target ratios
    
    Args:
        aspect_ratio: Original image aspect ratio
        target_ratios: List of (width_tiles, height_tiles) tuples
        width: Original image width
        height: Original image height
        image_size: Tile size (typically 640)
    
    Returns:
        Best matching (width_tiles, height_tiles) tuple
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 2,
    max_num: int = 6,
    image_size: int = 640,
    use_thumbnail: bool = False
) -> tuple:
    """
    Dynamically preprocess image into tiles based on aspect ratio
    
    This is the key technique from DeepSeek-OCR-vllm sample for improved accuracy.
    Instead of simply resizing, it intelligently splits images into tiles.
    
    Args:
        image: PIL Image object
        min_num: Minimum number of tiles
        max_num: Maximum number of tiles (reduce if GPU memory is limited)
        image_size: Size of each tile (typically 640)
        use_thumbnail: Whether to add a thumbnail view
    
    Returns:
        Tuple of (processed_images: List[Image], target_aspect_ratio: tuple)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    # Calculate possible tiling configurations
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    
    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    
    # Resize the image
    resized_img = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    processed_images = []
    
    # Split into tiles
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    
    # Add thumbnail if requested
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        processed_images.append(thumbnail_img)
    
    return processed_images, target_aspect_ratio


def preprocess_image(
    image: Image.Image,
    target_size: int,
    crop_mode: bool = False
) -> Image.Image:
    """
    Preprocess image according to resolution mode

    Args:
        image: PIL Image object
        target_size: Target resolution (e.g., 512, 640, 1024, 1280)
        crop_mode: Whether to use crop mode (for Gundam mode)

    Returns:
        Preprocessed PIL Image
    """
    if crop_mode:
        # For Gundam mode - will be handled by vLLM internally
        return image

    # For other modes - resize or pad to target size
    width, height = image.size

    if target_size <= 640:
        # Tiny/Small mode: direct resize
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    else:
        # Base/Large mode: padding to maintain aspect ratio
        max_dim = max(width, height)
        scale = target_size / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with padding
        new_image = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        new_image.paste(image, (paste_x, paste_y))
        image = new_image

    return image


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string
    """
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def parse_ocr_output(
    raw_output: str,
    output_format: str,
    parse_grounding: bool = True
) -> Union[str, Dict[str, Any]]:
    """
    Parse and format OCR output

    Args:
        raw_output: Raw output from the model
        output_format: Desired output format (text, markdown, json)
        parse_grounding: Whether to parse grounding tags into structured format

    Returns:
        Formatted output (string or dict)
    """
    if output_format == "text":
        # Extract plain text only
        # Remove grounding tags if present
        if parse_grounding and '<|ref|>' in raw_output:
            from .grounding_parser import parse_grounding_output
            _, clean_text = parse_grounding_output(raw_output)
            return clean_text
        return raw_output.strip()

    elif output_format == "markdown":
        # Return markdown as-is
        return raw_output.strip()

    elif output_format == "json":
        # Check if grounding tags are present
        has_grounding = '<|ref|>' in raw_output and '<|det|>' in raw_output

        if has_grounding and parse_grounding:
            # Parse grounding tags into structured format
            from .grounding_parser import parse_grounding_output, format_to_structured_json
            parsed_elements, clean_text = parse_grounding_output(raw_output)

            result = {
                "raw_content": raw_output.strip(),
                "clean_text": clean_text,
                "elements": [
                    {
                        "type": elem['type'],
                        "text": elem.get('text', ''),
                        "bounding_boxes": elem['bboxes']
                    }
                    for elem in parsed_elements
                ],
                "format": "grounding",
                "metadata": {
                    "has_grounding": True,
                    "element_count": len(parsed_elements),
                    "length": len(raw_output.strip())
                }
            }
        else:
            # Fallback to simple JSON format
            result = {
                "content": raw_output.strip(),
                "format": "markdown",
                "metadata": {
                    "has_grounding": False,
                    "length": len(raw_output.strip())
                }
            }

        return result

    else:
        raise ValueError(f"Unknown output format: {output_format}")


def save_output(
    output: Union[str, Dict[str, Any]],
    output_path: Union[str, Path],
    output_format: str
) -> None:
    """
    Save OCR output to file

    Args:
        output: Formatted output
        output_path: Path to save the output
        output_format: Output format (text, markdown, json)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    else:
        # text or markdown
        content = output if isinstance(output, str) else output.get("content", "")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)


def validate_paths(
    image_path: Union[str, Path],
    output_path: Union[str, Path]
) -> tuple[Path, Path]:
    """
    Validate input and output paths

    Args:
        image_path: Path to input image
        output_path: Path to output file

    Returns:
        Tuple of (validated image_path, validated output_path)

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If paths are invalid
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    # Check image extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if image_path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}. "
            f"Supported formats: {', '.join(valid_extensions)}"
        )

    return image_path, output_path


def calculate_valid_tokens(
    original_width: int,
    original_height: int,
    actual_tokens: int
) -> int:
    """
    Calculate valid vision tokens based on image aspect ratio

    Formula from spec.txt:
    N_valid = ceil(N_actual × [1 - ((max(w,h) - min(w,h)) / max(w,h))])

    Args:
        original_width: Original image width
        original_height: Original image height
        actual_tokens: Actual number of tokens (e.g., 256 for base mode)

    Returns:
        Number of valid tokens
    """
    max_dim = max(original_width, original_height)
    min_dim = min(original_width, original_height)

    ratio = (max_dim - min_dim) / max_dim
    valid_tokens = int(actual_tokens * (1 - ratio))

    return valid_tokens
