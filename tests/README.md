# Tests

This directory contains tests for the DeepSeek-OCR Example project.

## Test Structure

```
tests/
├── __init__.py           # Test package initialization
├── conftest.py           # Pytest fixtures and configuration
├── test_config.py        # Tests for configuration module
├── test_utils.py         # Tests for utility functions
├── test_cli.py           # Tests for CLI interface
└── test_ocr_processor.py # Tests for OCR processor (mocked)
```

## Running Tests

### Prerequisites

Make sure you have created the virtual environment:

```bash
# Create virtual environment (if not already done)
uv venv
```

**Note**: With `uv`, you don't need to activate the virtual environment. `uv pip` and `uv run` automatically use `.venv`.

### Install Development Dependencies

Install the development dependencies including pytest:

```bash
uv pip install -e ".[dev]"
```

### Run All Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run with coverage report
uv run pytest --cov=deepseek_ocr_example --cov-report=term-missing
```

### Run Specific Test Files

```bash
# Run only configuration tests
uv run pytest tests/test_config.py

# Run only utility tests
uv run pytest tests/test_utils.py

# Run only CLI tests
uv run pytest tests/test_cli.py
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
uv run pytest tests/test_config.py::TestResolutionMode

# Run a specific test function
uv run pytest tests/test_utils.py::TestLoadImage::test_load_valid_image
```

## Test Coverage

Generate an HTML coverage report:

```bash
uv run pytest --cov=deepseek_ocr_example --cov-report=html
```

Then open `htmlcov/index.html` in your browser.

## Test Types

### Unit Tests

Most tests are unit tests that test individual functions and classes in isolation:

- `test_config.py` - Tests configuration classes and enums
- `test_utils.py` - Tests utility functions for image processing and formatting
- `test_cli.py` - Tests CLI argument parsing (mocked)
- `test_ocr_processor.py` - Tests OCR processor logic (with mocked vLLM)

### Integration Tests

**Note**: The current tests mock the vLLM inference to avoid requiring:
- NVIDIA GPU with CUDA
- Model weights download (~7GB)
- Actual inference time

For true end-to-end testing with real model inference, you would need to:

1. Have a GPU with 16GB+ VRAM available
2. Install vLLM with CUDA support
3. Download the DeepSeek-OCR model weights
4. Create integration tests that skip mocking

Example integration test structure (not included):

```python
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_real_inference(sample_image):
    """Test with real model inference"""
    processor = OCRProcessor()  # No mocking
    result = processor.process_image(sample_image, output_path)
    assert result['success'] is True
```

## Fixtures

The `conftest.py` file provides reusable fixtures:

- `temp_dir` - Temporary directory for test outputs
- `sample_image` - Sample test image (1024×768)
- `small_image` - Small test image (512×512)
- `large_image` - Large test image (2048×1536)
- `output_path` - Output file path in temp directory

## Writing New Tests

When adding new functionality, please add corresponding tests:

1. Create test file: `tests/test_<module>.py`
2. Import the module to test
3. Create test classes using `Test<ClassName>` convention
4. Write test functions using `test_<description>` convention
5. Use fixtures from `conftest.py` where applicable
6. Mock external dependencies (vLLM, GPU operations)

Example:

```python
import pytest
from deepseek_ocr_example.my_module import my_function

class TestMyFunction:
    """Test my_function"""

    def test_basic_case(self):
        """Test basic functionality"""
        result = my_function(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge case"""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

## Continuous Integration

These tests are designed to run in CI/CD environments without requiring GPU access. All GPU-dependent operations are mocked.

## Troubleshooting

### Import Errors

If you get import errors, make sure the package is installed in editable mode:

```bash
uv pip install -e .
```

### Missing Dependencies

Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

### Fixture Not Found

Make sure `conftest.py` is in the `tests/` directory and pytest can find it.
