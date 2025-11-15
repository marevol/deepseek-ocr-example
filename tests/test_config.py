"""
Tests for configuration module
"""

import pytest
from deepseek_ocr_example.config import (
    ResolutionMode,
    OutputFormat,
    OCRConfig,
    ResolutionConfig,
    RESOLUTION_CONFIGS
)


class TestResolutionMode:
    """Test ResolutionMode enum"""

    def test_all_modes_exist(self):
        """Test that all resolution modes are defined"""
        assert ResolutionMode.TINY == "tiny"
        assert ResolutionMode.SMALL == "small"
        assert ResolutionMode.BASE == "base"
        assert ResolutionMode.LARGE == "large"
        assert ResolutionMode.GUNDAM == "gundam"

    def test_mode_from_string(self):
        """Test creating mode from string"""
        assert ResolutionMode("tiny") == ResolutionMode.TINY
        assert ResolutionMode("base") == ResolutionMode.BASE


class TestOutputFormat:
    """Test OutputFormat enum"""

    def test_all_formats_exist(self):
        """Test that all output formats are defined"""
        assert OutputFormat.TEXT == "text"
        assert OutputFormat.MARKDOWN == "markdown"
        assert OutputFormat.JSON == "json"

    def test_format_from_string(self):
        """Test creating format from string"""
        assert OutputFormat("json") == OutputFormat.JSON
        assert OutputFormat("markdown") == OutputFormat.MARKDOWN


class TestResolutionConfigs:
    """Test resolution configurations"""

    def test_all_modes_configured(self):
        """Test that all modes have configurations"""
        for mode in ResolutionMode:
            assert mode in RESOLUTION_CONFIGS

    def test_tiny_config(self):
        """Test tiny mode configuration"""
        config = RESOLUTION_CONFIGS[ResolutionMode.TINY]
        assert config.native_resolution == 512
        assert config.image_size == 512
        assert config.expected_tokens == 64
        assert config.crop_mode is False

    def test_small_config(self):
        """Test small mode configuration"""
        config = RESOLUTION_CONFIGS[ResolutionMode.SMALL]
        assert config.native_resolution == 640
        assert config.image_size == 640
        assert config.expected_tokens == 100
        assert config.crop_mode is False

    def test_base_config(self):
        """Test base mode configuration"""
        config = RESOLUTION_CONFIGS[ResolutionMode.BASE]
        assert config.native_resolution == 1024
        assert config.image_size == 1024
        assert config.expected_tokens == 256
        assert config.crop_mode is False

    def test_large_config(self):
        """Test large mode configuration"""
        config = RESOLUTION_CONFIGS[ResolutionMode.LARGE]
        assert config.native_resolution == 1280
        assert config.image_size == 1280
        assert config.expected_tokens == 400
        assert config.crop_mode is False

    def test_gundam_config(self):
        """Test gundam mode configuration"""
        config = RESOLUTION_CONFIGS[ResolutionMode.GUNDAM]
        assert config.native_resolution == 1024
        assert config.image_size == 640
        assert config.crop_mode is True


class TestOCRConfig:
    """Test OCRConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        config = OCRConfig()
        assert config.model_name == "deepseek-ai/DeepSeek-OCR"
        assert config.trust_remote_code is True
        assert config.dtype == "bfloat16"
        assert config.enable_prefix_caching is False
        assert config.mm_processor_cache_gb == 0
        assert config.default_resolution_mode == ResolutionMode.BASE
        assert config.default_output_format == OutputFormat.JSON

    def test_custom_config(self):
        """Test custom configuration"""
        config = OCRConfig(
            default_resolution_mode=ResolutionMode.SMALL,
            default_output_format=OutputFormat.TEXT,
            max_tokens=16384,
            gpu_id=1
        )
        assert config.default_resolution_mode == ResolutionMode.SMALL
        assert config.default_output_format == OutputFormat.TEXT
        assert config.max_tokens == 16384
        assert config.gpu_id == 1

    def test_get_prompt_text(self):
        """Test getting prompt for text format"""
        config = OCRConfig()
        prompt = config.get_prompt(OutputFormat.TEXT)
        assert "<image>" in prompt
        assert "Free OCR" in prompt

    def test_get_prompt_markdown(self):
        """Test getting prompt for markdown format"""
        config = OCRConfig()
        prompt = config.get_prompt(OutputFormat.MARKDOWN)
        assert "<image>" in prompt
        assert "<|grounding|>" in prompt
        assert "markdown" in prompt.lower()

    def test_get_prompt_json(self):
        """Test getting prompt for JSON format"""
        config = OCRConfig()
        prompt = config.get_prompt(OutputFormat.JSON)
        assert "<image>" in prompt
        assert "<|grounding|>" in prompt

    def test_get_resolution_config(self):
        """Test getting resolution configuration"""
        config = OCRConfig()
        res_config = config.get_resolution_config(ResolutionMode.BASE)
        assert isinstance(res_config, ResolutionConfig)
        assert res_config.mode == ResolutionMode.BASE
        assert res_config.native_resolution == 1024

    def test_invalid_output_format(self):
        """Test handling invalid output format"""
        config = OCRConfig()
        with pytest.raises(ValueError):
            config.get_prompt("invalid_format")
