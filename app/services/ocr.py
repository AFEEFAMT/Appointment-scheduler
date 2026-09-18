from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError
from pytesseract import Output

from app.config import settings
from app.schemas import OCRResult


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_IMAGE_PIXELS = 20_000_000


class OCRProcessingError(ValueError):
    """Raised when an image cannot be safely processed."""


@dataclass(frozen=True)
class OCRCandidate:
    """One OCR result from a preprocessing/PSM combination."""

    text: str
    confidence: float
    variant: str
    page_segmentation_mode: int

    @property
    def selection_score(self) -> float:
        """Balance OCR confidence with extracted text completeness."""

        completeness = min(len(self.text) / 100, 1.0)

        return (self.confidence * 0.90) + (
            completeness * 0.10
        )


def tesseract_is_available() -> bool:
    """Check whether the Tesseract executable is accessible."""

    try:
        pytesseract.get_tesseract_version()
        return True
    except (
        pytesseract.TesseractNotFoundError,
        RuntimeError,
    ):
        return False


def create_typed_text_result(text: str) -> OCRResult:
    """Create an OCR-compatible result for directly typed text."""

    cleaned_text = " ".join(text.split())

    if not cleaned_text:
        raise OCRProcessingError("Input text is empty")

    return OCRResult(
        raw_text=cleaned_text,
        confidence=1.0,
        engine="typed_text",
        preprocessing_variant=None,
    )


def validate_image_input(
    image_bytes: bytes,
    content_type: str | None,
) -> None:
    """Validate image type and upload size before decoding."""

    normalized_type = (
        content_type.split(";")[0].strip().lower()
        if content_type
        else ""
    )

    if normalized_type not in ALLOWED_IMAGE_TYPES:
        raise OCRProcessingError(
            "Only PNG, JPEG and WebP images are supported"
        )

    if not image_bytes:
        raise OCRProcessingError("Uploaded image is empty")

    maximum_bytes = settings.max_image_size_mb * 1024 * 1024

    if len(image_bytes) > maximum_bytes:
        raise OCRProcessingError(
            f"Image exceeds the {settings.max_image_size_mb} MB limit"
        )


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode an image safely and correct its EXIF orientation."""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")

            width, height = image.size

            if width * height > MAX_IMAGE_PIXELS:
                raise OCRProcessingError(
                    "Image dimensions are too large"
                )

            rgb_image = np.array(image)

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as error:
        raise OCRProcessingError(
            "Uploaded file is not a readable image"
        ) from error

    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def resize_small_image(image: np.ndarray) -> np.ndarray:
    """Upscale small images to improve character recognition."""

    height, width = image.shape[:2]
    largest_dimension = max(height, width)

    if largest_dimension >= 1400:
        return image

    scale = min(1400 / largest_dimension, 3.0)

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def deskew_image(grayscale: np.ndarray) -> np.ndarray:
    """Correct moderate image rotation before OCR."""

    _, binary = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    coordinates = cv2.findNonZero(binary)

    if coordinates is None or len(coordinates) < 20:
        return grayscale

    raw_angle = cv2.minAreaRect(coordinates)[-1]

    if raw_angle > 45:
        skew_angle = raw_angle - 90
    elif raw_angle < -45:
        skew_angle = raw_angle + 90
    else:
        skew_angle = raw_angle

    # Avoid damaging images with unreliable large-angle estimates.
    if abs(skew_angle) < 0.3 or abs(skew_angle) > 15:
        return grayscale

    height, width = grayscale.shape
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        -skew_angle,
        1.0,
    )

    return cv2.warpAffine(
        grayscale,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def create_preprocessing_variants(
    image: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create complementary image variants for OCR voting."""

    resized = resize_small_image(image)
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    grayscale = deskew_image(grayscale)

    grayscale = cv2.copyMakeBorder(
        grayscale,
        25,
        25,
        25,
        25,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(grayscale)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    _, otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return {
        "grayscale": grayscale,
        "enhanced": denoised,
        "otsu": otsu,
        "adaptive": adaptive,
    }


def run_tesseract_candidate(
    image: np.ndarray,
    variant: str,
    page_segmentation_mode: int,
) -> OCRCandidate | None:
    """Run one Tesseract configuration and calculate real confidence."""

    configuration = (
        f"--oem 3 --psm {page_segmentation_mode} "
        "-c preserve_interword_spaces=1"
    )

    try:
        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config=configuration,
            output_type=Output.DICT,
            timeout=15,
        )
    except (
        pytesseract.TesseractError,
        pytesseract.TesseractNotFoundError,
        RuntimeError,
    ):
        return None

    words: list[str] = []
    weighted_confidence = 0.0
    total_characters = 0

    for text, confidence_value in zip(
        data["text"],
        data["conf"],
        strict=False,
    ):
        word = text.strip()

        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            continue

        if not word or confidence < 0:
            continue

        words.append(word)

        character_count = max(len(word), 1)
        weighted_confidence += confidence * character_count
        total_characters += character_count

    if not words or total_characters == 0:
        return None

    extracted_text = " ".join(words)
    normalized_confidence = (
        weighted_confidence / total_characters
    ) / 100

    normalized_confidence = max(
        0.0,
        min(normalized_confidence, 1.0),
    )

    return OCRCandidate(
        text=extracted_text,
        confidence=normalized_confidence,
        variant=variant,
        page_segmentation_mode=page_segmentation_mode,
    )


def extract_text_from_image(
    image_bytes: bytes,
    content_type: str | None,
) -> OCRResult:
    """Run multi-pass OCR and return the strongest candidate."""

    validate_image_input(image_bytes, content_type)

    if not tesseract_is_available():
        raise OCRProcessingError(
            "Tesseract OCR is not available"
        )

    image = decode_image(image_bytes)
    variants = create_preprocessing_variants(image)

    candidates: list[OCRCandidate] = []

    # PSM 6: one text block. PSM 11: sparse text.
    configurations = [
        ("grayscale", 6),
        ("grayscale", 11),
        ("enhanced", 6),
        ("enhanced", 11),
        ("otsu", 6),
        ("adaptive", 6),
    ]

    for variant_name, psm in configurations:
        candidate = run_tesseract_candidate(
            variants[variant_name],
            variant_name,
            psm,
        )

        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        raise OCRProcessingError(
            "No readable text was found in the image"
        )

    best_candidate = max(
        candidates,
        key=lambda candidate: candidate.selection_score,
    )

    return OCRResult(
        raw_text=best_candidate.text,
        confidence=best_candidate.confidence,
        engine="tesseract",
        preprocessing_variant=(
            f"{best_candidate.variant}-psm"
            f"{best_candidate.page_segmentation_mode}"
        ),
    )