# DeepSeek-OCR Example

A high-performance command-line OCR tool using [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) with vLLM backend for efficient inference on NVIDIA GPUs.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

> **🚀 New to this project?** Check out the [Quick Start Guide](QUICKSTART.md) for a step-by-step walkthrough!

## Features

- **High Performance**: Powered by vLLM for fast batch inference (~2500 tokens/s on A100)
- **Multiple Resolution Modes**: Choose from 5 resolution modes (Tiny to Gundam) based on your needs
- **Flexible Output Formats**: Support for JSON, Markdown, and plain text outputs
- **Grounding Support**: Automatic text detection with bounding boxes and visualization
- **Custom Prompts**: Optimize detection accuracy by choosing task-specific prompts
- **16GB VRAM Compatible**: Optimized to run on NVIDIA GeForce RTX 5060 Ti (16GB)
- **Multiple Languages**: Support for ~100 languages including English, Chinese, and more
- **Easy to Use**: Simple command-line interface with sensible defaults

## Requirements

- Python 3.12+
- NVIDIA GPU with 16GB+ VRAM
- CUDA 11.8+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

> 📖 For detailed installation instructions and troubleshooting, see the [Installation Guide](docs/installation.md).

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the project

```bash
git clone https://github.com/codelibs/deepseek-ocr-example.git
cd deepseek-ocr-example
```

### 3. Install dependencies

```bash
# Create virtual environment
uv venv

# Install all dependencies (including vLLM from nightly index)
uv sync

# Optional: Install Flash Attention for better performance
uv sync --extra flash
```

**Notes**:
- `uv sync` automatically installs vLLM from the nightly build index
- Flash Attention (`--extra flash`) takes 5-10 minutes to compile but significantly improves speed
- If downloads timeout, set: `UV_HTTP_TIMEOUT=600 uv sync`
- No need to manually activate the virtual environment with uv!

## Quick Start

### Basic Usage

```bash
# Process an image with default settings (large mode, JSON output with grounding)
uv run ocr input.jpg output.json
```

**Output**: Creates both `output.json` (structured data) and `output_annotated.jpg` (visualization with bounding boxes)

### Optimizing for Different Content Types

**For the best detection accuracy**, use the appropriate prompt for your content type:

```bash
# Figures, charts, and graphs (RECOMMENDED for complex diagrams)
uv run ocr chart.png output.json --prompt "<|grounding|>OCR this image."

# Documents with layout (invoices, forms, structured documents)
uv run ocr invoice.jpg output.json --prompt "<|grounding|>Convert the document to markdown."

# Plain text extraction (no layout needed)
uv run ocr simple.jpg output.txt --format text --prompt "Free OCR."

# Analyze specific figures in documents
uv run ocr diagram.png output.json --prompt "Parse the figure."
```

💡 **Tip**: Different prompts work better for different content types. Experiment to find the best one for your use case.

### Resolution Modes

Choose the appropriate mode based on your image complexity and quality requirements:

| Mode | Resolution | Tokens | Use Case |
|------|-----------|--------|----------|
| `tiny` | 512×512 | 64 | Simple documents, fastest processing |
| `small` | 640×640 | 100 | General documents, good balance |
| `base` | 1024×1024 | 256 | Good for standard documents |
| `large` | 1280×1280 | 400 | **Recommended (Default)** - Best quality for most use cases |
| `gundam` | Dynamic | Variable | Ultra-high resolution (newspapers, posters) |

```bash
# Use small mode for faster processing
uv run ocr simple.jpg output.json --mode small

# Use large mode for high-quality OCR
uv run ocr complex_doc.jpg output.json --mode large

# Use gundam mode for ultra-high resolution images
uv run ocr newspaper.png output.json --mode gundam
```

### Output Formats

```bash
# JSON output with metadata (default)
uv run ocr input.jpg output.json --format json

# Markdown output with layout preservation
uv run ocr document.jpg output.md --format markdown

# Plain text output
uv run ocr simple.jpg output.txt --format text
```

### Advanced Options

```bash
# Use specific GPU
uv run ocr input.jpg output.json --gpu 1

# Increase max tokens for longer documents
uv run ocr long_doc.jpg output.json --max-tokens 16384

# Combine custom prompt with high resolution
uv run ocr complex_chart.png output.json --mode large --prompt "<|grounding|>OCR this image."

# Enable verbose logging
uv run ocr input.jpg output.json --verbose
```

### Grounding Output with Bounding Boxes

When using grounding prompts (e.g., `<|grounding|>OCR this image.`), the tool automatically generates:

1. **JSON output** with structured elements including:
   - Detected text content
   - Bounding box coordinates for each element
   - Metadata (element count, image dimensions, etc.)

2. **Annotated image** (`*_annotated.jpg`) showing:
   - Red bounding boxes around detected text regions
   - Color-coded by element type
   - Labels identifying each detected region

**Example output structure:**
```json
{
  "elements": [
    {
      "type": "title",
      "text": "Document Header",
      "bounding_boxes": [[100, 50, 500, 80]]
    }
  ],
  "metadata": {
    "has_grounding": true,
    "element_count": 80,
    "image_path": "input.png"
  }
}
```

## Command-Line Options

```
usage: ocr [-h] [--mode {tiny,small,base,large,gundam}]
           [--format {text,markdown,json}] [--gpu GPU]
           [--max-tokens MAX_TOKENS] [--prompt PROMPT]
           [--verbose] [--version]
           image output

positional arguments:
  image                 Path to input image file
  output                Path to output file

optional arguments:
  -h, --help            Show this help message and exit
  --mode {tiny,small,base,large,gundam}
                        Resolution mode (default: large)
  --format {text,markdown,json}
                        Output format (default: json)
  --gpu GPU             GPU device ID (default: 0)
  --max-tokens MAX_TOKENS
                        Maximum number of tokens to generate (default: 8192)
  --prompt PROMPT       Custom prompt template (overrides format default)
                        Common options:
                          "<|grounding|>OCR this image." (charts/figures)
                          "<|grounding|>Convert the document to markdown." (documents)
                          "Free OCR." (plain text)
                          "Parse the figure." (diagrams)
  --verbose             Enable verbose logging
  --version             Show program's version number and exit
```

## Output Examples

### JSON Format (with Grounding)

```json
{
  "raw_content": "<|ref|>title<|/ref|><|det|>[[100,50,500,80]]<|/det|>\n...",
  "clean_text": "Document Title\n...",
  "elements": [
    {
      "type": "title",
      "text": "Document Title",
      "bounding_boxes": [[100, 50, 500, 80]]
    },
    {
      "type": "paragraph",
      "text": "This is the extracted text...",
      "bounding_boxes": [[100, 100, 500, 200]]
    }
  ],
  "metadata": {
    "has_grounding": true,
    "element_count": 80,
    "image_path": "input.jpg",
    "resolution_mode": "large",
    "original_size": [2048, 1536],
    "expected_tokens": 400,
    "valid_tokens": 350,
    "compression_ratio": 9.87,
    "visualized_image": "output_annotated.jpg"
  }
}
```

### Markdown Format

```markdown
# Document Title

## Section 1

This is the extracted text with layout preserved...

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
```

### Text Format

```
Document Title
Section 1
This is the extracted text...
```

## Performance

- **Processing Speed**: ~2500 tokens/s on A100-40G, ~220 tokens/s on RTX 5060 Ti
- **Memory Usage**: 10-15GB VRAM depending on resolution mode
- **Compression Ratio**: Up to 10× with 97%+ accuracy
- **Supported Languages**: ~100 languages

### Compression Ratios vs Accuracy

| Text Tokens | Vision Tokens | Precision | Compression |
|-------------|--------------|-----------|-------------|
| 600-700 | 100 | 98.5% | 6.7× |
| 800-900 | 100 | 96.8% | 8.5× |
| 900-1000 | 100 | 96.8% | 9.7× |
| 1000-1100 | 100 | 91.5% | 10.6× |

## Documentation

See the `docs/` directory for detailed documentation:

- [model-architecture.md](docs/model-architecture.md) - Detailed model information and architecture
- [vllm-guide.md](docs/vllm-guide.md) - vLLM configuration and optimization guide
- [installation.md](docs/installation.md) - Detailed installation instructions

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors:

1. Reduce resolution mode (e.g., from `large` to `base` or `small`)
2. Lower `--max-tokens` value
3. Ensure no other processes are using GPU memory

### Slow Inference

1. Check that CUDA 11.8+ is installed
2. Install flash-attention for better performance
3. Use a lower resolution mode for faster processing

### Model Download Issues

The first run will download the model (~7GB). If download fails:

1. Check your internet connection
2. Ensure you have enough disk space (~10GB free)
3. Try using a mirror or VPN if needed

## Development

### Project Structure

```
deepseek-ocr-example/
├── src/
│   └── deepseek_ocr_example/
│       ├── __init__.py
│       ├── cli.py              # Command-line interface
│       ├── config.py           # Configuration and constants
│       ├── ocr_processor.py    # Main OCR processing logic
│       ├── grounding_parser.py # Grounding output parser & visualization
│       ├── logits_processor.py # N-gram repetition prevention
│       └── utils.py            # Utility functions
├── docs/
│   ├── model-architecture.md  # Model architecture details
│   ├── vllm-guide.md          # vLLM configuration guide
│   └── installation.md        # Detailed installation guide
├── tests/                     # Test suite
├── pyproject.toml             # Project configuration
├── README.md                  # This file
└── LICENSE                    # Apache 2.0 License
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=deepseek_ocr_example --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_config.py
```

**Note**: All tests mock vLLM/GPU operations to run without requiring actual GPU hardware.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) - The amazing OCR model
- [vLLM](https://github.com/vllm-project/vllm) - Fast inference engine
- [HuggingFace](https://huggingface.co/) - Model hosting and transformers library

## Citation

If you use this tool in your research, please cite the original DeepSeek-OCR paper:

```bibtex
@article{deepseek-ocr,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/codelibs/deepseek-ocr-example/issues)
3. Open a new issue if needed

---

**Note**: This project is an example implementation and is not officially affiliated with DeepSeek-AI.
