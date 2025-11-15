"""
Tests for utility functions
"""

import pytest
from pathlib import Path
from PIL import Image
import json

from deepseek_ocr_example.utils import (
    load_image,
    preprocess_image,
    image_to_base64,
    parse_ocr_output,
    save_output,
    validate_paths,
    calculate_valid_tokens
)


class TestLoadImage:
    """Test image loading functionality"""

    def test_load_valid_image(self, sample_image):
        """Test loading a valid image"""
        img = load_image(sample_image)
        assert isinstance(img, Image.Image)
        assert img.mode == 'RGB'

    def test_load_nonexistent_image(self, temp_dir):
        """Test loading a non-existent image"""
        with pytest.raises(FileNotFoundError):
            load_image(temp_dir / "nonexistent.jpg")

    def test_load_image_converts_to_rgb(self, temp_dir):
        """Test that images are converted to RGB"""
        # Create a grayscale image
        img_path = temp_dir / "grayscale.png"
        img = Image.new('L', (100, 100), color=128)
        img.save(img_path)

        loaded_img = load_image(img_path)
        assert loaded_img.mode == 'RGB'


class TestPreprocessImage:
    """Test image preprocessing"""

    def test_preprocess_tiny_mode(self, sample_image):
        """Test preprocessing for tiny mode"""
        img = load_image(sample_image)
        processed = preprocess_image(img, target_size=512, crop_mode=False)
        assert processed.size == (512, 512)

    def test_preprocess_small_mode(self, sample_image):
        """Test preprocessing for small mode"""
        img = load_image(sample_image)
        processed = preprocess_image(img, target_size=640, crop_mode=False)
        assert processed.size == (640, 640)

    def test_preprocess_base_mode_with_padding(self, sample_image):
        """Test preprocessing for base mode with padding"""
        img = load_image(sample_image)
        processed = preprocess_image(img, target_size=1024, crop_mode=False)
        assert processed.size == (1024, 1024)

    def test_preprocess_maintains_aspect_ratio(self, temp_dir):
        """Test that padding maintains aspect ratio"""
        # Create a wide image
        img_path = temp_dir / "wide.jpg"
        img = Image.new('RGB', (2000, 1000), color='white')
        img.save(img_path)

        loaded_img = load_image(img_path)
        processed = preprocess_image(loaded_img, target_size=1024, crop_mode=False)

        # Should be padded to 1024x1024
        assert processed.size == (1024, 1024)

    def test_preprocess_crop_mode(self, sample_image):
        """Test preprocessing with crop mode"""
        img = load_image(sample_image)
        # In crop mode, image should be returned as-is (handled by vLLM)
        processed = preprocess_image(img, target_size=1024, crop_mode=True)
        assert processed.size == img.size


class TestImageToBase64:
    """Test image to base64 conversion"""

    def test_image_to_base64(self, sample_image):
        """Test converting image to base64"""
        img = load_image(sample_image)
        base64_str = image_to_base64(img)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        # Base64 strings should only contain valid characters
        import base64
        try:
            base64.b64decode(base64_str)
        except Exception:
            pytest.fail("Invalid base64 string")


class TestParseOCROutput:
    """Test OCR output parsing"""

    def test_parse_text_format(self):
        """Test parsing text format output"""
        raw_output = "This is some extracted text.\nWith multiple lines."
        result = parse_ocr_output(raw_output, "text")

        assert isinstance(result, str)
        assert result == raw_output.strip()

    def test_parse_markdown_format(self):
        """Test parsing markdown format output"""
        raw_output = "# Title\n\nSome content"
        result = parse_ocr_output(raw_output, "markdown")

        assert isinstance(result, str)
        assert result == raw_output.strip()

    def test_parse_json_format(self):
        """Test parsing JSON format output"""
        raw_output = "# Document\n\nContent here"
        result = parse_ocr_output(raw_output, "json")

        assert isinstance(result, dict)
        assert "content" in result
        assert "format" in result
        assert "metadata" in result
        assert result["content"] == raw_output.strip()
        assert result["format"] == "markdown"

    def test_parse_json_with_grounding(self):
        """Test parsing JSON with grounding tags"""
        raw_output = "<|grounding|>Some content"
        result = parse_ocr_output(raw_output, "json")

        assert result["metadata"]["has_grounding"] is True

    def test_parse_json_without_grounding(self):
        """Test parsing JSON without grounding tags"""
        raw_output = "Some content"
        result = parse_ocr_output(raw_output, "json")

        assert result["metadata"]["has_grounding"] is False

    def test_parse_invalid_format(self):
        """Test parsing with invalid format"""
        with pytest.raises(ValueError):
            parse_ocr_output("content", "invalid")


class TestSaveOutput:
    """Test output saving functionality"""

    def test_save_text_output(self, temp_dir):
        """Test saving text output"""
        output_path = temp_dir / "output.txt"
        content = "This is text content"

        save_output(content, output_path, "text")

        assert output_path.exists()
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        assert saved_content == content

    def test_save_json_output(self, temp_dir):
        """Test saving JSON output"""
        output_path = temp_dir / "output.json"
        content = {
            "content": "Some content",
            "metadata": {"key": "value"}
        }

        save_output(content, output_path, "json")

        assert output_path.exists()
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_content = json.load(f)
        assert saved_content == content

    def test_save_creates_directory(self, temp_dir):
        """Test that save creates parent directories"""
        output_path = temp_dir / "subdir" / "output.txt"
        content = "Content"

        save_output(content, output_path, "text")

        assert output_path.exists()
        assert output_path.parent.exists()


class TestValidatePaths:
    """Test path validation"""

    def test_validate_valid_paths(self, sample_image, temp_dir):
        """Test validating valid paths"""
        output_path = temp_dir / "output.json"

        img_path, out_path = validate_paths(sample_image, output_path)

        assert isinstance(img_path, Path)
        assert isinstance(out_path, Path)
        assert img_path.exists()

    def test_validate_nonexistent_image(self, temp_dir):
        """Test validating non-existent image"""
        image_path = temp_dir / "nonexistent.jpg"
        output_path = temp_dir / "output.json"

        with pytest.raises(FileNotFoundError):
            validate_paths(image_path, output_path)

    def test_validate_invalid_extension(self, temp_dir):
        """Test validating invalid image extension"""
        # Create a text file with image extension
        image_path = temp_dir / "invalid.txt"
        image_path.write_text("not an image")
        output_path = temp_dir / "output.json"

        with pytest.raises(ValueError):
            validate_paths(image_path, output_path)

    def test_validate_directory_as_image(self, temp_dir):
        """Test validating directory as image path"""
        output_path = temp_dir / "output.json"

        with pytest.raises(ValueError):
            validate_paths(temp_dir, output_path)


class TestCalculateValidTokens:
    """Test valid token calculation"""

    def test_calculate_square_image(self):
        """Test calculation for square image"""
        tokens = calculate_valid_tokens(1024, 1024, 256)
        # Square image has no padding, so all tokens are valid
        assert tokens == 256

    def test_calculate_wide_image(self):
        """Test calculation for wide image"""
        tokens = calculate_valid_tokens(2048, 1024, 256)
        # Wide image has 50% padding vertically
        # ratio = (2048 - 1024) / 2048 = 0.5
        # valid = 256 * (1 - 0.5) = 128
        assert tokens == 128

    def test_calculate_tall_image(self):
        """Test calculation for tall image"""
        tokens = calculate_valid_tokens(1024, 2048, 256)
        # Tall image has 50% padding horizontally
        assert tokens == 128

    def test_calculate_slight_difference(self):
        """Test calculation with slight aspect ratio difference"""
        tokens = calculate_valid_tokens(1100, 1024, 256)
        # ratio = (1100 - 1024) / 1100 ≈ 0.069
        # valid = 256 * (1 - 0.069) ≈ 238
        assert tokens >= 230
        assert tokens <= 240
