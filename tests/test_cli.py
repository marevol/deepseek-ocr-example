"""
Tests for CLI module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

from deepseek_ocr_example.cli import parse_args, main
from deepseek_ocr_example.config import ResolutionMode, OutputFormat


class TestParseArgs:
    """Test command-line argument parsing"""

    def test_parse_basic_args(self, monkeypatch):
        """Test parsing basic required arguments"""
        monkeypatch.setattr(sys, 'argv', ['ocr', 'input.jpg', 'output.json'])
        args = parse_args()

        assert args.image == 'input.jpg'
        assert args.output == 'output.json'
        assert args.mode == 'base'
        assert args.format == 'json'
        assert args.gpu == 0

    def test_parse_with_mode(self, monkeypatch):
        """Test parsing with resolution mode"""
        monkeypatch.setattr(sys, 'argv', ['ocr', 'input.jpg', 'output.json', '--mode', 'small'])
        args = parse_args()

        assert args.mode == 'small'

    def test_parse_with_format(self, monkeypatch):
        """Test parsing with output format"""
        monkeypatch.setattr(sys, 'argv', ['ocr', 'input.jpg', 'output.md', '--format', 'markdown'])
        args = parse_args()

        assert args.format == 'markdown'

    def test_parse_with_gpu(self, monkeypatch):
        """Test parsing with GPU specification"""
        monkeypatch.setattr(sys, 'argv', ['ocr', 'input.jpg', 'output.json', '--gpu', '1'])
        args = parse_args()

        assert args.gpu == 1

    def test_parse_with_max_tokens(self, monkeypatch):
        """Test parsing with max tokens"""
        monkeypatch.setattr(sys, 'argv', ['ocr', 'input.jpg', 'output.json', '--max-tokens', '16384'])
        args = parse_args()

        assert args.max_tokens == 16384

    def test_parse_with_verbose(self, monkeypatch):
        """Test parsing with verbose flag"""
        monkeypatch.setattr(sys, 'argv', ['ocr', 'input.jpg', 'output.json', '--verbose'])
        args = parse_args()

        assert args.verbose is True

    def test_parse_all_modes(self, monkeypatch):
        """Test parsing all resolution modes"""
        for mode in ['tiny', 'small', 'base', 'large', 'gundam']:
            monkeypatch.setattr(sys, 'argv', ['ocr', 'input.jpg', 'output.json', '--mode', mode])
            args = parse_args()
            assert args.mode == mode

    def test_parse_all_formats(self, monkeypatch):
        """Test parsing all output formats"""
        for fmt in ['text', 'markdown', 'json']:
            monkeypatch.setattr(sys, 'argv', ['ocr', 'input.jpg', 'output.json', '--format', fmt])
            args = parse_args()
            assert args.format == fmt


class TestMain:
    """Test main CLI function"""

    @patch('deepseek_ocr_example.cli.OCRProcessor')
    def test_main_success(self, mock_processor_class, sample_image, temp_dir, monkeypatch):
        """Test successful main execution"""
        output_path = temp_dir / "output.json"

        # Mock processor
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.process_image.return_value = {
            'success': True,
            'metadata': {
                'original_size': (1024, 768),
                'resolution_mode': 'base',
                'expected_tokens': 256,
                'valid_tokens': 230,
                'output_length': 1000
            }
        }

        # Set command-line arguments
        monkeypatch.setattr(sys, 'argv', ['ocr', str(sample_image), str(output_path)])

        # Run main
        exit_code = main()

        assert exit_code == 0
        mock_processor.process_image.assert_called_once()

    @patch('deepseek_ocr_example.cli.OCRProcessor')
    def test_main_nonexistent_file(self, mock_processor_class, temp_dir, monkeypatch):
        """Test main with non-existent input file"""
        image_path = temp_dir / "nonexistent.jpg"
        output_path = temp_dir / "output.json"

        monkeypatch.setattr(sys, 'argv', ['ocr', str(image_path), str(output_path)])

        exit_code = main()

        assert exit_code == 1

    @patch('deepseek_ocr_example.cli.OCRProcessor')
    def test_main_processing_failure(self, mock_processor_class, sample_image, temp_dir, monkeypatch):
        """Test main when processing fails"""
        output_path = temp_dir / "output.json"

        # Mock processor to return failure
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.process_image.return_value = {
            'success': False,
            'error': 'Processing failed'
        }

        monkeypatch.setattr(sys, 'argv', ['ocr', str(sample_image), str(output_path)])

        exit_code = main()

        assert exit_code == 1

    @patch('deepseek_ocr_example.cli.OCRProcessor')
    def test_main_with_verbose(self, mock_processor_class, sample_image, temp_dir, monkeypatch):
        """Test main with verbose logging"""
        output_path = temp_dir / "output.json"

        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.process_image.return_value = {
            'success': True,
            'metadata': {
                'original_size': (1024, 768),
                'resolution_mode': 'base',
                'expected_tokens': 256,
                'valid_tokens': 230,
                'output_length': 1000
            }
        }

        monkeypatch.setattr(sys, 'argv', [
            'ocr', str(sample_image), str(output_path), '--verbose'
        ])

        exit_code = main()

        assert exit_code == 0

    @patch('deepseek_ocr_example.cli.OCRProcessor')
    def test_main_exception_handling(self, mock_processor_class, sample_image, temp_dir, monkeypatch):
        """Test main handles exceptions properly"""
        output_path = temp_dir / "output.json"

        # Mock processor to raise exception
        mock_processor_class.side_effect = Exception("Initialization failed")

        monkeypatch.setattr(sys, 'argv', ['ocr', str(sample_image), str(output_path)])

        exit_code = main()

        assert exit_code == 1
