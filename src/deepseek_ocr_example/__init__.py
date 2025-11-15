"""
DeepSeek-OCR Example - High-performance OCR using vLLM

A command-line tool for optical character recognition using DeepSeek-OCR model
with vLLM backend for efficient inference on NVIDIA GPUs.
"""

import os

# CRITICAL: Disable vLLM V1 engine at the earliest possible point
# Must be set BEFORE any vLLM imports
os.environ['VLLM_USE_V1'] = '0'

# Enable Hugging Face offline mode to prevent external network access
# Assumes models are already downloaded to cache
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

__version__ = "0.1.0"

from .ocr_processor import OCRProcessor
from .config import ResolutionMode, OCRConfig

__all__ = ["OCRProcessor", "ResolutionMode", "OCRConfig"]
