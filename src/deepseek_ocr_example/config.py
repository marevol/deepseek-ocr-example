"""
Configuration for DeepSeek-OCR

Defines resolution modes, model settings, and other configuration parameters.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ResolutionMode(str, Enum):
    """Resolution modes for DeepSeek-OCR"""
    TINY = "tiny"       # 512x512, 64 tokens
    SMALL = "small"     # 640x640, 100 tokens
    BASE = "base"       # 1024x1024, 256 tokens
    LARGE = "large"     # 1280x1280, 400 tokens
    GUNDAM = "gundam"   # Dynamic: n×640×640 + 1024×1024


class OutputFormat(str, Enum):
    """Output format options"""
    TEXT = "text"           # Plain text only
    MARKDOWN = "markdown"   # Markdown with layout
    JSON = "json"          # JSON with structured data


@dataclass
class ResolutionConfig:
    """Configuration for a specific resolution mode"""
    mode: ResolutionMode
    native_resolution: int
    image_size: int
    crop_mode: bool
    expected_tokens: int
    description: str


# Resolution mode configurations based on spec.txt
RESOLUTION_CONFIGS = {
    ResolutionMode.TINY: ResolutionConfig(
        mode=ResolutionMode.TINY,
        native_resolution=512,
        image_size=512,
        crop_mode=False,
        expected_tokens=64,
        description="Tiny mode: 512x512, 64 tokens, fastest processing"
    ),
    ResolutionMode.SMALL: ResolutionConfig(
        mode=ResolutionMode.SMALL,
        native_resolution=640,
        image_size=640,
        crop_mode=False,
        expected_tokens=100,
        description="Small mode: 640x640, 100 tokens, good for simple documents"
    ),
    ResolutionMode.BASE: ResolutionConfig(
        mode=ResolutionMode.BASE,
        native_resolution=1024,
        image_size=1024,
        crop_mode=False,
        expected_tokens=256,
        description="Base mode: 1024x1024, 256 tokens, recommended for most documents"
    ),
    ResolutionMode.LARGE: ResolutionConfig(
        mode=ResolutionMode.LARGE,
        native_resolution=1280,
        image_size=1280,
        crop_mode=False,
        expected_tokens=400,
        description="Large mode: 1280x1280, 400 tokens, high quality"
    ),
    ResolutionMode.GUNDAM: ResolutionConfig(
        mode=ResolutionMode.GUNDAM,
        native_resolution=1024,
        image_size=640,
        crop_mode=True,
        expected_tokens=356,  # Variable: n×100 + 256
        description="Gundam mode: Dynamic tiling for ultra-high resolution images"
    ),
}


@dataclass
class OCRConfig:
    """Main configuration for OCR processing"""
    # Model configuration
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    trust_remote_code: bool = True
    dtype: str = "bfloat16"

    # vLLM configuration
    enable_prefix_caching: bool = False
    mm_processor_cache_gb: int = 0
    max_model_len: int = 8192
    max_num_batched_tokens: int = 4096

    # Sampling parameters
    temperature: float = 0.0
    max_tokens: int = 8192
    ngram_size: int = 30
    window_size: int = 90

    # Processing configuration
    default_resolution_mode: ResolutionMode = ResolutionMode.LARGE  # Changed from BASE to LARGE for better text extraction
    default_output_format: OutputFormat = OutputFormat.JSON
    save_results: bool = True
    test_compress: bool = True
    custom_prompt: str = None  # Custom prompt template (overrides format default)

    # GPU configuration
    gpu_id: int = 0

    def get_prompt(self, output_format: OutputFormat) -> str:
        """Get prompt template based on output format or custom prompt"""
        # Use custom prompt if provided
        if self.custom_prompt:
            # Auto-prepend <image> tag if not present
            if "<image>" not in self.custom_prompt:
                return f"<image>\n{self.custom_prompt}"
            return self.custom_prompt

        # Default prompts by format
        if output_format == OutputFormat.TEXT:
            return "<image>\nFree OCR."
        elif output_format == OutputFormat.MARKDOWN:
            return "<image>\n<|grounding|>Convert the document to markdown."
        elif output_format == OutputFormat.JSON:
            # JSON format with structured output
            return "<image>\n<|grounding|>Convert the document to markdown."
        else:
            raise ValueError(f"Unknown output format: {output_format}")

    def get_resolution_config(self, mode: ResolutionMode) -> ResolutionConfig:
        """Get resolution configuration for specified mode"""
        return RESOLUTION_CONFIGS[mode]
