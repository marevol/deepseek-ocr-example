"""
Command-line interface for DeepSeek-OCR

Provides a simple CLI for running OCR on images using DeepSeek-OCR model.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import logging

# IMPORTANT: Disable vLLM V1 engine BEFORE importing vLLM
# Must be set before any vLLM imports occur
os.environ['VLLM_USE_V1'] = '0'

from .config import OCRConfig, ResolutionMode, OutputFormat
from .ocr_processor import OCRProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='DeepSeek-OCR: High-performance OCR using vLLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings (base mode, JSON output)
  ocr input.jpg output.json

  # Use different resolution mode
  ocr input.png output.json --mode small

  # Output as markdown
  ocr document.pdf output.md --format markdown

  # High-quality OCR for complex documents
  ocr complex_doc.jpg output.json --mode large

  # Ultra-high resolution with Gundam mode
  ocr newspaper.png output.json --mode gundam

  # Plain text output
  ocr simple.jpg output.txt --format text --mode tiny

  # Use specific GPU
  ocr input.jpg output.json --gpu 1

Resolution Modes:
  tiny   - 512x512,   64 tokens  (fastest, lowest quality)
  small  - 640x640,  100 tokens  (good for simple documents)
  base   - 1024x1024, 256 tokens (recommended, default)
  large  - 1280x1280, 400 tokens (high quality)
  gundam - Dynamic tiling       (ultra-high resolution)

Output Formats:
  text     - Plain text only
  markdown - Markdown with layout preservation
  json     - Structured JSON with metadata (default)
        """
    )

    # Required arguments
    parser.add_argument(
        'image',
        type=str,
        help='Path to input image file'
    )
    parser.add_argument(
        'output',
        type=str,
        help='Path to output file'
    )

    # Optional arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['tiny', 'small', 'base', 'large', 'gundam'],
        default='base',
        help='Resolution mode (default: base)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'markdown', 'json'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID (default: 0)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=8192,
        help='Maximum number of tokens to generate (default: 8192)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Custom prompt template (overrides format default). '
             'Common options: "OCR this image.", "Parse the figure.", "Free OCR."'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0'
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for CLI

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse arguments
        args = parse_args()

        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose/debug logging enabled")
            logger.debug(f"Arguments: {vars(args)}")

        # Convert string arguments to enums
        resolution_mode = ResolutionMode(args.mode)
        output_format = OutputFormat(args.format)

        # Validate input file
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return 1

        # Create configuration
        config = OCRConfig(
            default_resolution_mode=resolution_mode,
            default_output_format=output_format,
            max_tokens=args.max_tokens,
            gpu_id=args.gpu,
            custom_prompt=args.prompt
        )

        # Initialize processor
        logger.info("Initializing OCR processor...")
        processor = OCRProcessor(config=config, gpu_id=args.gpu)

        # Process image
        logger.info(f"Processing image: {args.image}")
        result = processor.process_image(
            image_path=args.image,
            output_path=args.output,
            resolution_mode=resolution_mode,
            output_format=output_format
        )

        # Check result
        if result['success']:
            logger.info("✓ Processing completed successfully")
            logger.info(f"Output saved to: {args.output}")

            # Print metadata if verbose
            if args.verbose:
                metadata = result['metadata']
                logger.info("\nMetadata:")
                logger.info(f"  Original size: {metadata['original_size']}")
                logger.info(f"  Resolution mode: {metadata['resolution_mode']}")
                logger.info(f"  Expected tokens: {metadata['expected_tokens']}")
                logger.info(f"  Valid tokens: {metadata['valid_tokens']}")
                logger.info(f"  Output length: {metadata['output_length']} chars")

            return 0
        else:
            logger.error(f"✗ Processing failed: {result.get('error', 'Unknown error')}")
            return 1

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
