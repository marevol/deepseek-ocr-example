"""
Pytest configuration and fixtures
"""

import pytest
from pathlib import Path
from PIL import Image
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image(temp_dir):
    """Create a simple test image"""
    image_path = temp_dir / "test_image.jpg"

    # Create a simple white image with some text-like pattern
    img = Image.new('RGB', (1024, 768), color='white')

    # Draw some simple patterns to simulate text
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Draw some rectangles to simulate text blocks
    for i in range(5):
        y = 100 + i * 100
        draw.rectangle([(100, y), (900, y + 50)], outline='black', width=2)

    img.save(image_path, 'JPEG')
    return image_path


@pytest.fixture
def small_image(temp_dir):
    """Create a small test image"""
    image_path = temp_dir / "small_test.jpg"
    img = Image.new('RGB', (512, 512), color='white')
    img.save(image_path, 'JPEG')
    return image_path


@pytest.fixture
def large_image(temp_dir):
    """Create a large test image"""
    image_path = temp_dir / "large_test.jpg"
    img = Image.new('RGB', (2048, 1536), color='white')
    img.save(image_path, 'JPEG')
    return image_path


@pytest.fixture
def output_path(temp_dir):
    """Generate an output file path"""
    return temp_dir / "output.json"
