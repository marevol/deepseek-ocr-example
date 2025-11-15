"""
OCR Processor using vLLM backend

This module provides the main OCR processing functionality using
DeepSeek-OCR model with vLLM for efficient inference.
"""

import os
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import logging

# Disable vLLM V1 engine - use legacy engine for better compatibility
os.environ['VLLM_USE_V1'] = '0'

from PIL import Image
from vllm import LLM, SamplingParams

from .config import OCRConfig, ResolutionMode, OutputFormat, RESOLUTION_CONFIGS
from .utils import (
    load_image,
    preprocess_image,
    parse_ocr_output,
    save_output,
    validate_paths,
    calculate_valid_tokens
)
from .logits_processor import NoRepeatNGramLogitsProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    OCR Processor using DeepSeek-OCR with vLLM backend

    This class handles model initialization, image preprocessing,
    inference, and output formatting.
    """

    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        gpu_id: int = 0
    ):
        """
        Initialize OCR Processor

        Args:
            config: OCR configuration (uses default if None)
            gpu_id: GPU device ID to use
        """
        self.config = config or OCRConfig()
        self.gpu_id = gpu_id

        # Set GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Initialize model
        logger.info(f"Initializing DeepSeek-OCR model on GPU {gpu_id}...")
        self.llm = self._initialize_model()
        logger.info("Model initialized successfully")

    def _initialize_model(self) -> LLM:
        """
        Initialize vLLM model with proper configuration

        Returns:
            Initialized LLM instance
        """
        try:
            llm = LLM(
                model=self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                dtype=self.config.dtype,
                enable_prefix_caching=self.config.enable_prefix_caching,
                mm_processor_cache_gb=self.config.mm_processor_cache_gb,
                max_model_len=self.config.max_model_len,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
            )
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _create_sampling_params(self) -> SamplingParams:
        """
        Create sampling parameters for inference

        Returns:
            SamplingParams instance
        """
        logger.debug(
            f"Creating sampling params: temperature={self.config.temperature}, "
            f"max_tokens={self.config.max_tokens}, "
            f"ngram_size={self.config.ngram_size}, "
            f"window_size={self.config.window_size}"
        )

        # Create custom logits processor for n-gram repetition prevention
        # Whitelist token IDs: 128821 (<td>), 128822 (</td>) for table parsing
        # NOTE: Temporarily disabled due to vLLM API compatibility issues
        # logits_processors = [
        #     NoRepeatNGramLogitsProcessor(
        #         ngram_size=self.config.ngram_size,
        #         window_size=self.config.window_size,
        #         whitelist_token_ids={128821, 128822}
        #     )
        # ]

        return SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            # logits_processors=logits_processors,  # Temporarily disabled
            skip_special_tokens=False,  # Important: keep special tokens for grounding
        )

    def process_image(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        resolution_mode: Optional[ResolutionMode] = None,
        output_format: Optional[OutputFormat] = None
    ) -> Dict[str, Any]:
        """
        Process a single image and save results

        Args:
            image_path: Path to input image
            output_path: Path to save output
            resolution_mode: Resolution mode to use (default: config.default_resolution_mode)
            output_format: Output format (default: config.default_output_format)

        Returns:
            Dictionary containing processing results and metadata
        """
        # Validate paths
        image_path, output_path = validate_paths(image_path, output_path)

        # Use defaults if not specified
        resolution_mode = resolution_mode or self.config.default_resolution_mode
        output_format = output_format or self.config.default_output_format

        logger.info("=" * 60)
        logger.info(f"🖼️  Processing image: {image_path}")
        logger.info(f"   Resolution mode: {resolution_mode.value}")
        logger.info(f"   Output format: {output_format.value}")
        logger.info("=" * 60)

        try:
            # Load and preprocess image
            logger.info("📂 Step 1/5: Loading image...")
            image = load_image(image_path)
            original_size = image.size
            logger.info(f"   ✓ Loaded - Original size: {original_size[0]}x{original_size[1]} pixels")

            # Get resolution configuration
            logger.info("🔧 Step 2/5: Preprocessing image...")
            res_config = RESOLUTION_CONFIGS[resolution_mode]
            logger.debug(
                f"Resolution config: native_resolution={res_config.native_resolution}, "
                f"expected_tokens={res_config.expected_tokens}, "
                f"crop_mode={res_config.crop_mode}"
            )

            # Preprocess image
            processed_image = preprocess_image(
                image,
                target_size=res_config.native_resolution,
                crop_mode=res_config.crop_mode
            )
            logger.info(f"   ✓ Preprocessed - Target size: {processed_image.size[0]}x{processed_image.size[1]} pixels")
            logger.info(f"   ✓ Expected tokens: {res_config.expected_tokens}")

            # Create prompt
            logger.info("📋 Step 3/5: Preparing prompt...")
            prompt = self.config.get_prompt(output_format)
            logger.debug(f"Using prompt: {prompt[:100]}...")
            logger.info(f"   ✓ Prompt prepared (length: {len(prompt)} chars)")

            # Prepare multimodal input
            # vLLM expects PIL Image objects directly (not file paths)
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": processed_image  # Pass PIL Image directly
                }
            }

            # Create sampling parameters
            sampling_params = self._create_sampling_params()

            # Run inference
            logger.info("⚙️  Step 4/5: Running DeepSeek-OCR inference...")
            logger.debug(f"Input prepared: prompt length={len(prompt)}, image size={processed_image.size}")

            outputs = self.llm.generate(
                prompts=[inputs],
                sampling_params=sampling_params
            )

            logger.info("   ✓ Inference completed successfully")

            # Extract output text and metadata
            output_obj = outputs[0].outputs[0]
            raw_output = output_obj.text

            # Log output metadata from DeepSeek-OCR
            logger.info(f"📊 DeepSeek-OCR Output Statistics:")
            logger.info(f"   - Output length: {len(raw_output)} characters")
            logger.info(f"   - Token IDs count: {len(output_obj.token_ids)}")
            if output_obj.cumulative_logprob is not None:
                logger.info(f"   - Cumulative logprob: {output_obj.cumulative_logprob:.4f}")
            logger.info(f"   - Finish reason: {output_obj.finish_reason}")

            logger.debug(f"Raw output preview: {raw_output[:200]}...")

            # Parse and format output
            logger.info("📝 Step 5/5: Parsing and formatting output...")
            formatted_output = parse_ocr_output(raw_output, output_format.value)

            # Calculate metadata
            valid_tokens = calculate_valid_tokens(
                original_size[0],
                original_size[1],
                res_config.expected_tokens
            )

            # Calculate compression ratio
            compression_ratio = len(output_obj.token_ids) / valid_tokens if valid_tokens > 0 else 0

            metadata = {
                "image_path": str(image_path),
                "output_path": str(output_path),
                "resolution_mode": resolution_mode.value,
                "output_format": output_format.value,
                "original_size": original_size,
                "processed_size": processed_image.size,
                "expected_tokens": res_config.expected_tokens,
                "valid_tokens": valid_tokens,
                "output_length": len(raw_output),
                "token_count": len(output_obj.token_ids),
                "compression_ratio": round(compression_ratio, 2),
                "finish_reason": output_obj.finish_reason
            }

            logger.info(f"📈 Processing Metrics:")
            logger.info(f"   - Expected tokens: {res_config.expected_tokens}")
            logger.info(f"   - Valid tokens (after padding): {valid_tokens}")
            logger.info(f"   - Actual output tokens: {len(output_obj.token_ids)}")
            logger.info(f"   - Compression ratio: {compression_ratio:.2f}x")

            # Add metadata to JSON output
            if output_format == OutputFormat.JSON and isinstance(formatted_output, dict):
                formatted_output["metadata"].update(metadata)

            # Generate visualization if grounding output detected
            visualized_image_path = None
            if formatted_output.get("metadata", {}).get("has_grounding") and isinstance(formatted_output, dict):
                if "elements" in formatted_output and formatted_output["elements"]:
                    logger.info(f"🎨 Generating visualization with bounding boxes...")
                    try:
                        from .grounding_parser import draw_bounding_boxes, parse_grounding_output

                        # Parse elements again for visualization
                        parsed_elements, _ = parse_grounding_output(raw_output)

                        # Create visualization output path
                        output_path_obj = Path(output_path)
                        visualized_image_path = output_path_obj.parent / f"{output_path_obj.stem}_annotated.jpg"

                        # Draw bounding boxes on original image
                        # Coordinates are relative to preprocessed image, need to scale back
                        draw_bounding_boxes(
                            image=image,
                            parsed_elements=parsed_elements,
                            output_path=visualized_image_path,
                            processed_image_size=processed_image.size,
                            original_image_size=original_size
                        )

                        logger.info(f"   ✓ Visualization saved to: {visualized_image_path}")

                        # Add visualization path to metadata
                        formatted_output["metadata"]["visualized_image"] = str(visualized_image_path)

                    except Exception as e:
                        logger.warning(f"   ⚠ Failed to generate visualization: {e}")

            # Save output
            logger.info(f"💾 Saving output...")
            save_output(formatted_output, output_path, output_format.value)
            logger.info("=" * 60)
            logger.info(f"✅ Processing completed successfully!")
            logger.info(f"📄 Output saved to: {output_path}")
            if visualized_image_path:
                logger.info(f"🎨 Annotated image saved to: {visualized_image_path}")
            logger.info("=" * 60)

            return {
                "success": True,
                "output": formatted_output,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "image_path": str(image_path)
            }

    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        resolution_mode: Optional[ResolutionMode] = None,
        output_format: Optional[OutputFormat] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch

        Args:
            image_paths: List of image paths
            output_dir: Directory to save outputs
            resolution_mode: Resolution mode to use
            output_format: Output format

        Returns:
            List of processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for image_path in image_paths:
            image_path = Path(image_path)
            output_path = output_dir / f"{image_path.stem}_ocr.json"

            result = self.process_image(
                image_path,
                output_path,
                resolution_mode,
                output_format
            )
            results.append(result)

        return results

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'llm'):
            # vLLM cleanup if needed
            pass
