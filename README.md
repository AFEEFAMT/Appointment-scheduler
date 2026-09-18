# AI-Powered Appointment Scheduler

A FastAPI backend that converts typed or image-based appointment requests into structured scheduling data.

## Live Demo

- **Interactive API Documentation:** https://appointment-scheduler-x2hc.onrender.com/docs
- **Health Check:** https://appointment-scheduler-x2hc.onrender.com/health
- **Base API URL:** https://appointment-scheduler-x2hc.onrender.com

> The service runs on a free Render instance and may require up to one minute to wake after inactivity.

The service implements the complete pipeline required by the assignment:

```text
Text or Image
      ↓
OCR / Text Extraction
      ↓
Gemini Entity Extraction
      ↓
Department Validation
      ↓
Deterministic Date-Time Normalization
      ↓
Structured Appointment JSON
```

## Features

- Accepts typed text and uploaded appointment images
- Multi-pass Tesseract OCR
- Image deskewing, denoising, contrast enhancement and thresholding
- Gemini structured entity extraction
- Model retry and fallback chain
- In-memory TTL cache for repeated extraction requests
- Department alias and OCR typo correction
- Relative-date normalization in `Asia/Kolkata`
- Guardrails for ambiguous, missing or past appointment details
- Consistent error responses
- Interactive Swagger documentation
- Automated unit, integration and API tests

## Example

Input:

```text
Book dentist next Friday at 3pm
```

Output:

```json
{
  "appointment": {
    "department": "Dentistry",
    "date": "2026-09-25",
    "time": "15:00",
    "tz": "Asia/Kolkata"
  },
  "status": "ok"
}
```

Incomplete input:

```text
Book a dentist appointment
```

Output:

```json
{
  "status": "needs_clarification",
  "message": "Please provide the appointment date and appointment time."
}
```

## Architecture

### Input layer

The API exposes separate endpoints for typed text and image uploads. Both inputs are converted into the same `OCRResult` structure before entering the processing pipeline.

### OCR layer

Image requests are processed using multiple OpenCV preprocessing variants:

- EXIF orientation correction
- Resizing
- Deskewing
- Grayscale conversion
- CLAHE contrast enhancement
- Denoising
- Otsu thresholding
- Adaptive thresholding
- Multiple Tesseract page-segmentation modes

The candidate with the best combination of OCR confidence and appointment-field completeness is selected.

### Entity extraction

Gemini extracts only:

- `date_phrase`
- `time_phrase`
- `department`
- `ambiguity_reason`

Gemini is intentionally not responsible for calculating final dates or times. Its response is constrained using a structured schema and validated again using Pydantic.

Configured model chain:

```text
gemini-3.6-flash
    ↓
gemini-3.1-flash-lite
    ↓
gemini-3.5-flash-lite
```

Temporary service failures are retried with exponential backoff. Timeouts move directly to the next model.

### Validation

The validation layer:

- Detects missing appointment information
- Rejects generic departments such as `doctor` or `specialist`
- Maps aliases such as `dentist` to `Dentistry`
- Corrects minor OCR errors using conservative fuzzy matching
- Accepts clear specialties that are not in the built-in alias catalogue
- Generates a focused clarification question when required

### Normalization

Date and time normalization is deterministic and independent of Gemini.

Supported examples include:

- `today`
- `tomorrow`
- `day after tomorrow`
- `next Friday`
- `25 Sep 2026`
- `September 25, 2026`
- `3pm`
- `3:30pm`
- `15:00`
- `noon`
- `midnight`

Ambiguous expressions such as `next week`, `morning` and numeric dates such as `09/10/2026` are rejected instead of being guessed.

## Project Structure

```text
Appointment-scheduler/
├── app/
│   ├── services/
│   │   ├── extractor.py
│   │   ├── normalizer.py
│   │   ├── ocr.py
│   │   ├── pipeline.py
│   │   └── validator.py
│   ├── config.py
│   ├── main.py
│   └── schemas.py
├── scripts/
│   ├── benchmark.py
│   └── generate_test_images.py
├── tests/
│   ├── fixtures/
│   │   └── images/
│   ├── test_api.py
│   ├── test_normalizer.py
│   ├── test_ocr.py
│   ├── test_pipeline.py
│   └── test_validator.py
├── .env.example
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

## Local Setup

### Requirements

- Python 3.13
- Tesseract OCR 5
- Gemini API key

Verify Tesseract:

```bash
tesseract --version
```

### Create the virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Configure environment variables

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Add the Gemini API key to `.env`:

```env
GEMINI_API_KEY=your_real_api_key
```

Never commit the `.env` file.

## Run the API

```powershell
uvicorn app.main:app --reload
```

Available URLs:

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health: `http://127.0.0.1:8000/health`

## API Usage

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Typed appointment

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/appointments/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Book dentist next Friday at 3pm",
    "reference_datetime": "2026-09-18T12:00:00+05:30"
  }'
```

### Image appointment

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/appointments/image" \
  -F "image=@tests/fixtures/images/appointment_noisy.png;type=image/png" \
  -F "reference_datetime=2026-09-18T12:00:00+05:30"
```

### Clarification guardrail

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/appointments/text" \
  -H "Content-Type: application/json" \
  -d '{"text":"Book a dentist appointment"}'
```

## HTTP Responses

| Code | Meaning |
|---|---|
| `200` | Appointment created or clarification required |
| `413` | Uploaded image exceeds the size limit |
| `415` | Unsupported image type |
| `422` | Invalid request or unreadable image |
| `503` | AI extraction service temporarily unavailable |

## Testing

Run the complete suite:

```powershell
pytest -v
```

Current result:

```text
86 passed
```

The tests cover:

- API success and failure responses
- Typed-text processing
- Clean, noisy, rotated and compressed images
- Corrupted and unsupported images
- Image size limits
- Date and time normalization
- Ambiguity detection
- Past appointment rejection
- Department aliases and OCR typo correction
- Pipeline integration
- Gemini service failure handling

## Generate OCR Fixtures

```powershell
python scripts/generate_test_images.py
```

## Docker

Build:

```bash
docker build -t appointment-scheduler .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY="your_api_key" \
  appointment-scheduler
```

## Reliability and Guardrails

- The Gemini API key is loaded only from environment variables.
- Raw appointment text is not written to application logs.
- Model output is treated as untrusted and validated using Pydantic.
- Prompt instructions prohibit invented appointment information.
- Relative dates are resolved using an explicit timezone.
- Ambiguous dates and times are never silently guessed.
- Past appointments are rejected.
- Upload type, size and image integrity are validated.
- Repeated successful extractions are cached temporarily.
- Internal exceptions are not returned to API clients.

## Design Decision

The service uses a hybrid AI and deterministic architecture.

Gemini handles the language-understanding task of extracting phrases and departments. Deterministic Python code performs validation and date-time normalization. This prevents the language model from inventing calendar values while retaining flexibility for natural-language and OCR inputs.