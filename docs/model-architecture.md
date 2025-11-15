# DeepSeek-OCR Model Architecture

## Overview

- **Model**: deepseek-ai/DeepSeek-OCR
- **Total Parameters**: ~3.3B
  - DeepEncoder: 380M
  - DeepSeek-3B-MoE: 570M activated parameters
- **Model Size**: ~6.7GB (BF16 format)
- **Recommended VRAM**: 16GB+

## Architecture Components

### DeepEncoder (380M parameters)

The DeepEncoder is the vision component responsible for extracting and compressing image features:

1. **SAM-base (80M)**: Window attention-based visual perception
   - Patch size: 16×16
   - Processes high-resolution inputs efficiently

2. **CLIP-large (300M)**: Dense global attention for visual knowledge
   - Provides semantic understanding
   - Pre-trained vision-language alignment

3. **16× Token Compressor**: 2-layer convolutional downsampling
   - Reduces vision tokens by 16× before global attention
   - Kernel size: 3, Stride: 2, Padding: 1
   - Channels: 256 → 1024

**Token Flow Example** (1024×1024 input):
```
Input: 1024×1024 image
  ↓ Patch embedding (16×16)
4096 tokens (SAM processing)
  ↓ 16× Compressor
256 tokens (CLIP processing)
  ↓ Output to decoder
```

### DeepSeek-3B-MoE Decoder (570M activated)

Mixture of Experts (MoE) architecture optimized for OCR:

- **Total Experts**: 64 routed experts + 2 shared experts
- **Active Experts**: 6 routed + 2 shared per token
- **Efficiency**: 3B model expressiveness with 570M inference cost
- **Sequence Length**: 8192 tokens

## Resolution Modes

### Native Resolution Modes

| Mode | Resolution | Tokens | Processing | Memory | Use Case |
|------|-----------|--------|-----------|---------|----------|
| Tiny | 512×512 | 64 | Resize | ~2-3GB | Fast processing, simple docs |
| Small | 640×640 | 100 | Resize | ~2-3GB | General documents |
| Base | 1024×1024 | 256 | Padding | ~3-4GB | Recommended default |
| Large | 1280×1280 | 400 | Padding | ~4-5GB | High-quality requirements |

### Dynamic Resolution Modes

| Mode | Configuration | Tokens | Memory | Use Case |
|------|--------------|--------|---------|----------|
| Gundam | n×640 + 1024 global | n×100 + 256 | ~5-8GB | Ultra-high res (newspapers) |
| Gundam-M | n×1024 + 1280 global | n×256 + 400 | ~8-12GB | Maximum quality |

**Gundam Mode Details**:
- Splits large images into 640×640 tiles (local views)
- Adds a 1024×1024 global view
- Number of tiles (n) ranges from 2-9
- For images < 640×640, degrades to Base mode

## Memory Requirements (16GB VRAM)

### Memory Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| Model Weights | ~7GB | BF16 precision |
| Activations (Tiny/Small) | 2-3GB | Varies by resolution |
| Activations (Base) | 3-4GB | Recommended mode |
| Activations (Large) | 4-5GB | High quality |
| Activations (Gundam) | 5-8GB | Depends on tile count |

**Total**: 10-15GB → **Compatible with 16GB VRAM**

### Token Calculation for Padded Images

For images requiring padding (Base/Large modes):

```
Valid Tokens = ⌈Actual Tokens × [1 - ((max(w,h) - min(w,h)) / max(w,h))]⌉
```

Example: 2048×1024 image in Base mode (1024×1024)
- Actual tokens: 256
- Aspect ratio penalty: (2048-1024)/2048 = 0.5
- Valid tokens: 256 × (1 - 0.5) = 128

## Performance Characteristics

### Compression Ratios vs Accuracy

Based on Fox benchmark (English documents, 600-1300 tokens):

| Text Tokens | Vision Tokens (64) | Vision Tokens (100) |
|-------------|-------------------|---------------------|
| 600-700 | 96.5% (10.5×) | 98.5% (6.7×) |
| 700-800 | 93.8% (11.8×) | 97.3% (7.5×) |
| 800-900 | 83.8% (13.2×) | 96.8% (8.5×) |
| 900-1000 | 85.9% (15.1×) | 96.8% (9.7×) |
| 1000-1100 | 79.3% (16.5×) | 91.5% (10.6×) |

**Key Findings**:
- **10× compression**: 97%+ accuracy
- **20× compression**: ~60% accuracy
- Optimal range: 6-10× compression for production use

### Inference Speed

- **A100-40G**: ~2500 tokens/s
- **Production Scale**: 33M pages/day with 20 nodes (160× A100-40G)
- **Batch Processing**: Recommended for throughput optimization

## Supported Capabilities

### Languages

- **Total**: ~100 languages
- **Primary**: Chinese, English (25M pages training each)
- **Others**: 5M pages across 98 languages
- All languages support layout-aware and layout-free modes

### Output Modes

1. **Free OCR** (Layout-free): Plain text extraction
2. **Grounding OCR** (Layout-aware): Structured with coordinates
3. **Deep Parsing**: Charts (HTML), Chemistry (SMILES), Geometry

### Prompt Templates

```python
# Plain text
"<image>\nFree OCR."

# Markdown with layout
"<image>\n<|grounding|>Convert the document to markdown."

# Deep parsing (charts)
"<image>\n<|grounding|>Extract chart as HTML table."
```

## References

- **Paper**: DeepSeek-OCR: Contexts Optical Compression
- **Model**: [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Code**: [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)
