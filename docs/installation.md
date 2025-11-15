# Installation Guide

Detailed installation instructions for DeepSeek-OCR Example.

## Prerequisites

Before installing, ensure you have:

- **Python 3.12+** installed
- **NVIDIA GPU** with 16GB+ VRAM (e.g., RTX 4090, RTX 5060 Ti, A100)
- **CUDA 11.8+** or **CUDA 12.1+** installed
- **20GB+ free disk space** (for model weights and dependencies)
- **uv package manager** installed

### Check Your System

```bash
# Check Python version
python --version  # Should be 3.12 or higher

# Check CUDA version
nvcc --version    # Should be 11.8 or higher

# Check GPU
nvidia-smi        # Should show your GPU with 16GB+ memory

# Check disk space
df -h .           # Should have 20GB+ available
```

## Installation Methods

### Method 1: Simple Installation (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/codelibs/deepseek-ocr-example.git
cd deepseek-ocr-example

# 2. Create virtual environment
uv venv

# 3. Install all dependencies
uv sync

# 4. (Optional) Install Flash Attention for 2-3x speedup
uv sync --extra flash
```

That's it! The `uv sync` command automatically:
- Installs vLLM from the nightly build index
- Installs all required dependencies (torch, transformers, etc.)
- Sets up the CLI command (`ocr`)

### Method 2: Development Installation

For contributors or developers:

```bash
# Clone and enter directory
git clone https://github.com/codelibs/deepseek-ocr-example.git
cd deepseek-ocr-example

# Create virtual environment
uv venv

# Install with development dependencies
uv sync --extra dev

# Install with both dev and flash
uv sync --extra dev --extra flash
```

## Configuration Details

All dependencies are managed in `pyproject.toml`:

### Core Dependencies (automatically installed with `uv sync`)

- **torch** (≥2.6.0): Deep learning framework
- **transformers** (≥4.46.3): HuggingFace transformers
- **vllm** (≥0.7.0): Fast inference engine (from nightly index)
- **pillow**: Image processing
- **einops**, **addict**, **easydict**: Utilities

### Optional Dependencies

Install with `uv sync --extra <name>`:

- **`--extra dev`**: Testing and development tools (pytest, coverage)
- **`--extra flash`**: Flash Attention for faster inference

### How vLLM Nightly Index Works

The `pyproject.toml` includes special uv configuration:

```toml
[tool.uv]
extra-index-url = ["https://wheels.vllm.ai/nightly"]

[tool.uv.sources]
vllm = { index = "https://wheels.vllm.ai/nightly" }
```

This tells uv to fetch vLLM from the nightly build index instead of PyPI.

## Troubleshooting

### Timeout Errors During Installation

Large downloads (torch, vLLM) may timeout on slow connections:

```bash
# Increase timeout to 10 minutes
UV_HTTP_TIMEOUT=600 uv sync

# Or 15 minutes for very slow connections
UV_HTTP_TIMEOUT=900 uv sync
```

### Flash Attention Build Errors

Flash Attention requires CUDA and a C++ compiler. Common issues:

**Error: "No module named 'psutil'"**
- This is now handled automatically by `pyproject.toml`
- Try: `uv sync --extra flash` again

**Error: "CUDA not found"**
- Ensure CUDA is installed: `nvcc --version`
- Set CUDA_HOME: `export CUDA_HOME=/usr/local/cuda`

**Error: "Compilation failed"**
- Skip Flash Attention: just run `uv sync` without `--extra flash`
- The system works fine without it, just 2-3x slower

### vLLM Installation Issues

**Error: "Failed to download vllm"**
- Check internet connection
- Try increasing timeout: `UV_HTTP_TIMEOUT=900 uv sync`
- Verify index URL: `https://wheels.vllm.ai/nightly`

### CUDA Compatibility Issues

**Error: "forward compatibility was attempted on non supported HW"**
- Your GPU/driver doesn't support the CUDA version in torch
- Check compatibility: `nvidia-smi` and look for "CUDA Version"
- You may need to install a different torch version

## Verifying Installation

After installation, verify everything works:

```bash
# Check vLLM installation
uv run python -c "import vllm; print(vllm.__version__)"

# Check torch with CUDA
uv run python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check Flash Attention (if installed)
uv run python -c "import flash_attn; print(flash_attn.__version__)"

# Test the CLI
uv run ocr --version
```

Expected output:
```
vllm: 0.11.x
Torch: 2.9.x, CUDA: True
flash_attn: 2.7.x (if installed)
ocr 0.1.0
```

## Updating

To update to the latest version:

```bash
# Pull latest code
git pull

# Update dependencies
uv sync --upgrade

# Update with extras
uv sync --upgrade --extra flash
```

## Uninstallation

To completely remove:

```bash
# Remove virtual environment
rm -rf .venv

# Remove project directory
cd ..
rm -rf deepseek-ocr-example
```

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](../QUICKSTART.md)
2. Check [Usage Examples](../README.md#quick-start)
3. Review [vLLM Configuration](vllm-guide.md)
4. Explore [Model Architecture](model-architecture.md)

## Getting Help

If you encounter issues:

1. Check this guide and troubleshooting section
2. Search [GitHub Issues](https://github.com/codelibs/deepseek-ocr-example/issues)
3. Open a new issue with:
   - Error message (full traceback)
   - System info (`python --version`, `nvcc --version`, `nvidia-smi`)
   - Installation command used
