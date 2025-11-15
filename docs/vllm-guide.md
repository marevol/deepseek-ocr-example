# vLLM Configuration Guide

This guide covers vLLM setup, configuration, and optimization for DeepSeek-OCR.

## Installation

### Prerequisites

- Python 3.12+
- CUDA 11.8+ or CUDA 12.1+
- NVIDIA GPU with 16GB+ VRAM
- [uv](https://github.com/astral-sh/uv) package manager

### Setup Steps

#### 1. Create Virtual Environment

```bash
# Create a virtual environment with Python 3.12
uv venv
```

**Note**: With `uv`, you don't need to manually activate the virtual environment. Commands like `uv pip` and `uv run` automatically detect and use `.venv`.

#### 2. Install vLLM

DeepSeek-OCR requires vLLM nightly build (officially supported since October 23, 2025):

```bash
UV_HTTP_TIMEOUT=600 uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

**Timeout handling**: Large downloads may require extended timeouts. If the default 30s is insufficient, increase as needed:
```bash
UV_HTTP_TIMEOUT=900 uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

#### 3. Install Dependencies

Required dependencies (tested with Python 3.12.9 + CUDA 11.8):

```bash
# Core dependencies (handled by uv sync if using the project)
uv pip install torch>=2.6.0
uv pip install transformers>=4.46.3
uv pip install tokenizers>=0.20.3
uv pip install einops
uv pip install addict
uv pip install easydict
uv pip install pillow>=10.0.0

# Optional but recommended for performance
uv pip install flash-attn==2.7.3 --no-build-isolation
```

**Note**: If you're using this project's `pyproject.toml`, simply run:
```bash
uv sync
```
This will install all required dependencies automatically.

## vLLM Configuration

### Critical Configuration Parameters

These settings are **required** for optimal DeepSeek-OCR performance:

#### 1. Disable Prefix Caching

```python
enable_prefix_caching=False
```

**Why**: OCR tasks don't benefit from image reuse, as each image is unique.

#### 2. Disable Multimodal Processor Cache

```python
mm_processor_cache_gb=0
```

**Why**: No advantage for OCR workloads; saves memory.

#### 3. Enable Custom Logits Processor

```python
--logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor
```

**Why**: Essential for optimal OCR accuracy and markdown generation quality.

## Usage Examples

### Offline Inference (Python API)

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=8192,
    max_num_batched_tokens=4096,  # Adjust for 16GB VRAM
)

# Configure sampling
sampling_params = SamplingParams(
    temperature=0.0,           # Deterministic output
    max_tokens=8192,           # Maximum output length
    extra_args={               # Custom parameters for DeepSeek-OCR
        "ngram_size": 30,      # N-gram repetition penalty size
        "window_size": 90      # Look-back window for repetition detection
    }
)

# Prepare input
# Load and preprocess image
from PIL import Image
image = Image.open("/path/to/image.jpg")
# Note: preprocess image according to your resolution mode

prompt = "<image>\n<|grounding|>Convert the document to markdown."
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image  # Pass PIL Image object, not file path string
    }
}

# Run inference
outputs = llm.generate(
    prompts=[inputs],
    sampling_params=sampling_params
)

# Extract result
text = outputs[0].outputs[0].text
print(text)
```

### Online Inference (API Server)

#### Start Server

```bash
vllm serve deepseek-ai/DeepSeek-OCR \
  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
  --enable-prefix-caching False \
  --mm-processor-cache-gb 0 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9
```

#### Client Request (OpenAI-Compatible API)

```python
from openai import OpenAI
import base64

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used but required
)

# Encode image
with open("document.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Make request
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-OCR",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "<|grounding|>Convert the document to markdown."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ]
    }],
    max_tokens=8192,
    temperature=0.0
)

result = response.choices[0].message.content
print(result)
```

## Performance Optimization

### Memory Optimization for 16GB VRAM

```python
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    trust_remote_code=True,

    # Memory-critical settings
    dtype="bfloat16",                 # Half precision
    max_model_len=8192,               # Limit sequence length
    max_num_batched_tokens=4096,      # Reduce batch size
    gpu_memory_utilization=0.9,       # Leave some VRAM headroom
    swap_space=4,                     # 4GB CPU RAM for swapping
)
```

### Batch Size Guidelines

| Resolution Mode | Recommended Batch Size | Memory Usage |
|----------------|----------------------|--------------|
| Tiny (512×512) | 4-8 | ~10-12GB |
| Small (640×640) | 4-6 | ~11-13GB |
| Base (1024×1024) | 2-4 | ~12-14GB |
| Large (1280×1280) | 1-2 | ~13-15GB |
| Gundam (Dynamic) | 1 | ~14-16GB |

### GPU Selection

```python
import os

# Use specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Or in vLLM server
vllm serve ... --tensor-parallel-size 1 --gpu-ids 0
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: CUDA OOM, allocation failed

**Solutions** (try in order):
1. Reduce `max_num_batched_tokens` (e.g., 4096 → 2048)
2. Lower `max_model_len` (e.g., 8192 → 4096)
3. Use smaller resolution mode (Large → Base → Small)
4. Reduce batch size to 1
5. Enable CPU swap: `swap_space=4`

### Slow Inference Speed

**Symptoms**: <100 tokens/s on RTX 5060 Ti

**Solutions**:
1. Verify Flash Attention 2 is installed:
   ```bash
   python -c "import flash_attn; print(flash_attn.__version__)"
   ```

2. Check CUDA version (11.8+ or 12.1+):
   ```bash
   nvcc --version
   ```

3. Ensure `dtype="bfloat16"` is set

4. Profile with:
   ```bash
   VLLM_TRACE=1 vllm serve ...
   ```

### Low Accuracy

**Symptoms**: Poor OCR quality, garbled output

**Solutions**:
1. Verify custom logits processor is enabled:
   ```python
   # Check server logs for:
   # "Using logits processor: NGramPerReqLogitsProcessor"
   ```

2. Use higher resolution mode (Small → Base → Large)

3. Include `<|grounding|>` tag for layout-aware OCR:
   ```python
   prompt = "<image>\n<|grounding|>Convert the document to markdown."
   ```

4. Increase `max_tokens` for long documents:
   ```python
   sampling_params = SamplingParams(max_tokens=16384)
   ```

### Model Download Issues

**Symptoms**: Timeout, connection errors

**Solutions**:
1. Check internet connection and disk space (~10GB free)

2. Use HuggingFace mirror:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. Manual download:
   ```bash
   git lfs install
   git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
   ```

4. Point to local path:
   ```python
   llm = LLM(model="/path/to/DeepSeek-OCR", ...)
   ```

## Best Practices

### 1. First Run

On first execution, expect:
- Model download: ~7GB (5-30 minutes depending on connection)
- Model loading: 30-60 seconds
- Compilation (CUDA kernels): 1-2 minutes

Subsequent runs skip download and are much faster.

### 2. Prompt Engineering

**For layout preservation**:
```python
prompt = "<image>\n<|grounding|>Convert the document to markdown."
```

**For plain text**:
```python
prompt = "<image>\nFree OCR."
```

**For specific extraction**:
```python
prompt = "<image>\n<|grounding|>Extract all tables as HTML."
```

### 3. Image Preprocessing

- **Aspect ratio**: Maintain original aspect ratio with padding
- **Format**: Convert to RGB (JPEG or PNG)
- **Size**: Match resolution mode (e.g., 1024×1024 for Base)
- **Quality**: Use high-quality source images for best results

### 4. Error Handling

```python
try:
    outputs = llm.generate(prompts=[inputs], sampling_params=params)
    text = outputs[0].outputs[0].text
except torch.cuda.OutOfMemoryError:
    # Reduce batch size or resolution
    print("OOM: Reduce max_num_batched_tokens or use lower resolution mode")
except Exception as e:
    # Log and retry
    print(f"Inference failed: {e}")
```

### 5. Production Deployment

- Use **API server mode** for multiple clients
- Enable **monitoring** (Prometheus metrics available)
- Set **timeout limits** (typical: 30-120s per image)
- Implement **retry logic** with exponential backoff
- Use **batch processing** for high throughput

## Advanced Configuration

### Custom Logits Processor Implementation

```python
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

# Already included in vLLM, but you can customize if needed
processor = NGramPerReqLogitsProcessor(
    ngram_size=30,
    window_size=90
)
```

### Multi-GPU Setup

For models requiring >16GB or higher throughput:

```bash
vllm serve deepseek-ai/DeepSeek-OCR \
  --tensor-parallel-size 2 \  # Use 2 GPUs
  --gpu-ids 0,1 \
  ...
```

**Note**: DeepSeek-OCR (3.3B) typically doesn't need tensor parallelism for inference, but can benefit from pipeline parallelism for batch processing.

## Benchmarking

### Single Image Inference

```bash
time uv run ocr input.jpg output.json --mode base
```

Typical timings (RTX 5060 Ti 16GB):
- Model loading: 30-45s (first run only)
- Tiny mode: 1-2s
- Small mode: 2-3s
- Base mode: 3-5s
- Large mode: 5-8s
- Gundam mode: 10-20s (varies with image size)

### Batch Processing

```python
import time

images = ["img1.jpg", "img2.jpg", ..., "img100.jpg"]
start = time.time()

# Process batch
results = processor.process_batch(images, output_dir)

elapsed = time.time() - start
print(f"Throughput: {len(images)/elapsed:.2f} images/s")
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [DeepSeek-OCR vLLM Recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
