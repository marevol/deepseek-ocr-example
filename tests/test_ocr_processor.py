"""
Tests for OCR processor module

Note: These are unit tests that mock vLLM. Integration tests with actual
model inference would require GPU and model weights.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from deepseek_ocr_example.ocr_processor import OCRProcessor
from deepseek_ocr_example.config import OCRConfig, ResolutionMode, OutputFormat


class TestOCRProcessor:
    """Test OCRProcessor class"""

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_initialization(self, mock_llm_class):
        """Test processor initialization"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        processor = OCRProcessor(gpu_id=0)

        assert processor.config is not None
        assert processor.gpu_id == 0
        mock_llm_class.assert_called_once()

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_initialization_with_custom_config(self, mock_llm_class):
        """Test processor initialization with custom config"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        config = OCRConfig(
            default_resolution_mode=ResolutionMode.SMALL,
            max_tokens=16384
        )
        processor = OCRProcessor(config=config, gpu_id=1)

        assert processor.config.default_resolution_mode == ResolutionMode.SMALL
        assert processor.config.max_tokens == 16384
        assert processor.gpu_id == 1

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_create_sampling_params(self, mock_llm_class):
        """Test sampling parameters creation"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        processor = OCRProcessor()
        sampling_params = processor._create_sampling_params()

        assert sampling_params.temperature == 0.0
        assert sampling_params.max_tokens == 8192

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_process_image_success(self, mock_llm_class, sample_image, temp_dir):
        """Test successful image processing"""
        # Mock LLM
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # Mock inference output
        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "# Document Title\n\nExtracted content"
        mock_llm.generate.return_value = [mock_output]

        processor = OCRProcessor()
        output_path = temp_dir / "output.json"

        result = processor.process_image(
            sample_image,
            output_path,
            resolution_mode=ResolutionMode.BASE,
            output_format=OutputFormat.JSON
        )

        assert result['success'] is True
        assert 'output' in result
        assert 'metadata' in result
        assert output_path.exists()

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_process_image_with_different_modes(self, mock_llm_class, sample_image, temp_dir):
        """Test processing with different resolution modes"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "Content"
        mock_llm.generate.return_value = [mock_output]

        processor = OCRProcessor()

        for mode in [ResolutionMode.TINY, ResolutionMode.SMALL, ResolutionMode.BASE]:
            output_path = temp_dir / f"output_{mode.value}.json"
            result = processor.process_image(
                sample_image,
                output_path,
                resolution_mode=mode,
                output_format=OutputFormat.JSON
            )
            assert result['success'] is True
            assert result['metadata']['resolution_mode'] == mode.value

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_process_image_with_different_formats(self, mock_llm_class, sample_image, temp_dir):
        """Test processing with different output formats"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "# Title\n\nContent"
        mock_llm.generate.return_value = [mock_output]

        processor = OCRProcessor()

        for fmt in [OutputFormat.TEXT, OutputFormat.MARKDOWN, OutputFormat.JSON]:
            output_path = temp_dir / f"output.{fmt.value}"
            result = processor.process_image(
                sample_image,
                output_path,
                resolution_mode=ResolutionMode.BASE,
                output_format=fmt
            )
            assert result['success'] is True
            assert result['metadata']['output_format'] == fmt.value

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_process_image_nonexistent(self, mock_llm_class, temp_dir):
        """Test processing non-existent image"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        processor = OCRProcessor()
        image_path = temp_dir / "nonexistent.jpg"
        output_path = temp_dir / "output.json"

        result = processor.process_image(
            image_path,
            output_path
        )

        assert result['success'] is False
        assert 'error' in result

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_process_image_inference_error(self, mock_llm_class, sample_image, temp_dir):
        """Test handling inference errors"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_llm.generate.side_effect = Exception("Inference failed")

        processor = OCRProcessor()
        output_path = temp_dir / "output.json"

        result = processor.process_image(
            sample_image,
            output_path
        )

        assert result['success'] is False
        assert 'error' in result

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_process_batch(self, mock_llm_class, sample_image, small_image, temp_dir):
        """Test batch processing"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "Content"
        mock_llm.generate.return_value = [mock_output]

        processor = OCRProcessor()
        image_paths = [sample_image, small_image]

        results = processor.process_batch(
            image_paths,
            temp_dir,
            resolution_mode=ResolutionMode.BASE,
            output_format=OutputFormat.JSON
        )

        assert len(results) == 2
        assert all(r['success'] for r in results)

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_metadata_includes_token_info(self, mock_llm_class, sample_image, temp_dir):
        """Test that metadata includes token information"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "Content"
        mock_llm.generate.return_value = [mock_output]

        processor = OCRProcessor()
        output_path = temp_dir / "output.json"

        result = processor.process_image(
            sample_image,
            output_path,
            resolution_mode=ResolutionMode.BASE
        )

        metadata = result['metadata']
        assert 'expected_tokens' in metadata
        assert 'valid_tokens' in metadata
        assert 'original_size' in metadata
        assert 'processed_size' in metadata
        assert metadata['expected_tokens'] == 256  # Base mode

    @patch('deepseek_ocr_example.ocr_processor.LLM')
    def test_uses_correct_prompt(self, mock_llm_class, sample_image, temp_dir):
        """Test that correct prompts are used for different formats"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        mock_output = Mock()
        mock_output.outputs = [Mock()]
        mock_output.outputs[0].text = "Content"
        mock_llm.generate.return_value = [mock_output]

        processor = OCRProcessor()

        # Test text format prompt
        output_path = temp_dir / "output.txt"
        processor.process_image(
            sample_image,
            output_path,
            output_format=OutputFormat.TEXT
        )

        call_args = mock_llm.generate.call_args
        prompt = call_args[1]['prompts'][0]['prompt']
        assert "Free OCR" in prompt

        # Test markdown format prompt
        mock_llm.generate.reset_mock()
        output_path = temp_dir / "output.md"
        processor.process_image(
            sample_image,
            output_path,
            output_format=OutputFormat.MARKDOWN
        )

        call_args = mock_llm.generate.call_args
        prompt = call_args[1]['prompts'][0]['prompt']
        assert "<|grounding|>" in prompt
