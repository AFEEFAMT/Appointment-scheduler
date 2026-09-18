from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "tests" / "fixtures" / "images"

APPOINTMENT_TEXT = "Book dentist next Friday at 3pm"


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a common font while remaining cross-platform."""

    font_names = [
        "arial.ttf",
        "DejaVuSans.ttf",
    ]

    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue

    return ImageFont.load_default(size=size)


def create_base_image() -> Image.Image:
    """Create a clean appointment-request image."""

    image = Image.new(
        mode="RGB",
        size=(1200, 300),
        color="white",
    )

    draw = ImageDraw.Draw(image)
    font = load_font(48)

    heading_font = load_font(30)

    draw.text(
        (60, 45),
        "Appointment Request",
        font=heading_font,
        fill=(80, 80, 80),
    )

    draw.text(
        (60, 125),
        APPOINTMENT_TEXT,
        font=font,
        fill="black",
    )

    return image


def create_clean_image(base_image: Image.Image) -> None:
    """Save the original high-quality image."""

    base_image.save(
        OUTPUT_DIRECTORY / "appointment_clean.png"
    )


def create_rotated_image(base_image: Image.Image) -> None:
    """Create a moderately rotated image for deskew testing."""

    rotated = base_image.rotate(
        5,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor="white",
    )

    rotated.save(
        OUTPUT_DIRECTORY / "appointment_rotated.png"
    )


def create_noisy_image(base_image: Image.Image) -> None:
    """Create a low-contrast, blurred and noisy image."""

    low_contrast = ImageEnhance.Contrast(
        base_image
    ).enhance(0.45)

    blurred = low_contrast.filter(
        ImageFilter.GaussianBlur(radius=1.1)
    )

    image_array = np.asarray(
        blurred,
        dtype=np.int16,
    )

    random_generator = np.random.default_rng(seed=42)

    noise = random_generator.normal(
        loc=0,
        scale=12,
        size=image_array.shape,
    )

    noisy_array = np.clip(
        image_array + noise,
        0,
        255,
    ).astype(np.uint8)

    noisy_image = Image.fromarray(noisy_array)

    noisy_image.save(
        OUTPUT_DIRECTORY / "appointment_noisy.png"
    )


def create_compressed_image(base_image: Image.Image) -> None:
    """Create a strongly compressed JPEG image."""

    memory_buffer = BytesIO()

    base_image.save(
        memory_buffer,
        format="JPEG",
        quality=28,
    )

    memory_buffer.seek(0)

    with Image.open(memory_buffer) as compressed:
        compressed.convert("RGB").save(
            OUTPUT_DIRECTORY / "appointment_compressed.jpg",
            format="JPEG",
            quality=28,
        )


def main() -> None:
    """Generate every deterministic OCR fixture."""

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_image = create_base_image()

    create_clean_image(base_image)
    create_rotated_image(base_image)
    create_noisy_image(base_image)
    create_compressed_image(base_image)

    print(
        f"Created 4 images in: {OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()