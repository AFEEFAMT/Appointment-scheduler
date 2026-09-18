# AI-Powered Appointment Scheduler

A FastAPI backend that converts typed or image-based appointment requests into validated scheduling JSON using Tesseract OCR, Gemini entity extraction, source-grounding checks, and deterministic date/time normalization.

The service produces scheduling data; it does not reserve slots or persist bookings.

## Live Demo

- [Interactive API documentation](https://appointment-scheduler-x2hc.onrender.com/docs)
- [Health check](https://appointment-scheduler-x2hc.onrender.com/health)
- [GitHub repository](https://github.com/AFEEFAMT/Appointment-scheduler)

**Base URL:** `https://appointment-scheduler-x2hc.onrender.com`

The cloud deployment runs independently of the developer's computer. The hosting instance may have a cold-start delay after inactivity.

## Features

- Typed-text and image-upload endpoints
- PNG, JPEG, and WebP support
- Multi-pass Tesseract OCR with image preprocessing
- Structured Gemini extraction with Pydantic validation
- Configurable model fallback, timeouts, and retries
- Source-grounding checks for extracted date/time phrases
- Department alias resolution and conservative typo correction
- Deterministic normalization in `Asia/Kolkata`
- Clarification for missing, ambiguous, or past scheduling details
- Upload size, pixel-count, and image-integrity checks
- Expiring, process-local extraction cache
- Automated API, OCR, extractor, pipeline, and normalization tests

## Example

Request:

```json
{
  "text": "Book dentist next Friday at 3pm",
  "reference_datetime": "2026-09-18T12:00:00+05:30"
}
```

Response:

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

Incomplete input such as `Book a dentist appointment` returns clarification instead of invented scheduling details:

```json
{
  "status": "needs_clarification",
  "message": "Please provide the appointment date and appointment time."
}
```

Clarification wording can vary depending on the extracted ambiguity.

## Architecture

Both endpoints share the following processing stages:

1. Convert typed text or an image into an `OCRResult`.
2. For images, check OCR confidence before calling Gemini.
3. Extract department, original date/time phrases, and ambiguity reason.
4. Check extracted date/time phrases against the source text.
5. Validate required fields and resolve department aliases.
6. Normalize date/time using deterministic Python logic.
7. Return appointment JSON or a clarification response.

### OCR

Image preprocessing includes EXIF orientation correction, resizing, deskewing, grayscale conversion, contrast enhancement, denoising, and thresholding.

Tesseract runs across multiple preprocessing variants and page-segmentation modes. Candidates are ranked using character-weighted Tesseract confidence and a capped text-length bonus. This ranking does not measure appointment-field correctness.

Images below the configured confidence threshold request clarification before Gemini extraction.

### Entity extraction

Gemini extracts four fields:

- `date_phrase`
- `time_phrase`
- `department`
- `ambiguity_reason`

Structured model output is validated using Pydantic. Gemini extracts original date/time wording rather than calculating calendar values.

Default model chain:

1. `gemini-3.6-flash`
2. `gemini-3.1-flash-lite`
3. `gemini-3.5-flash-lite`

Temporary API and network failures receive bounded retries with exponential backoff and jitter. Recognized client timeouts move directly to the next model. Authentication and quota errors stop the chain.

Model access and availability depend on the configured Gemini account.

### Source grounding

Date/time phrases must match a contiguous span in the source text after conservative casing, spacing, and punctuation normalization.

The check also rejects some dropped leading modifiers, such as extracting `Friday` from `next Friday`. Unsupported schedule phrases are removed and require clarification.

Grounding verifies source support, not complete semantic correctness.

### Validation and normalization

Validation checks required fields, resolves department aliases, and respects extraction ambiguity.

Date/time normalization is independent of Gemini. Supported examples include:

- `today`, `tomorrow`, and `day after tomorrow`
- `next Friday`
- `2026-09-25`
- `25 Sep 2026`
- `September 25, 2026`
- `3pm`, `3:30pm`, and `15:00`
- `noon` and `midnight`

Covered ambiguous expressions, including `next week`, `morning`, and `09/10/2026`, request clarification. Past appointments are rejected.

## Project Organization

| File or directory | Responsibility |
| --- | --- |
| `app/main.py` | API routes, request validation, and error handling |
| `app/config.py` | Environment-based configuration |
| `app/schemas.py` | Response and pipeline data models |
| `app/services/ocr.py` | Typed-text handling and image OCR |
| `app/services/extractor.py` | Gemini extraction, retries, fallback, and caching |
| `app/services/grounding.py` | Source checks for extracted date/time phrases |
| `app/services/validator.py` | Required-field and department validation |
| `app/services/normalizer.py` | Deterministic date/time normalization |
| `app/services/pipeline.py` | Shared processing orchestration |
| `tests/` | Automated tests |
| `tests/fixtures/images/` | Synthetic OCR fixtures |
| `scripts/generate_test_images.py` | Fixture generation |
| `Dockerfile` | Container setup, including Tesseract |

## Local Setup

### Requirements

- Python 3.13
- Tesseract OCR 5, available on `PATH`
- Gemini API key with access to the configured models

Verify:

```powershell
python --version
tesseract --version
```

### Install dependencies

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Configure environment variables

For a fresh setup:

```powershell
Copy-Item .env.example .env
```

Do not overwrite an existing `.env` containing your settings.

Edit `.env`:

```env
GEMINI_API_KEY=your_real_api_key

GEMINI_PRIMARY_MODEL=gemini-3.6-flash
GEMINI_FALLBACK_MODELS=gemini-3.1-flash-lite,gemini-3.5-flash-lite

APP_TIMEZONE=Asia/Kolkata
MAX_IMAGE_SIZE_MB=5
OCR_CONFIDENCE_THRESHOLD=0.60

LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=1
LLM_RETRY_BASE_SECONDS=1.0
CACHE_TTL_SECONDS=600
```

`.env` contains local configuration and secrets and must not be committed. `.env.example` documents settings without real credentials.

The current scheduling schema supports `Asia/Kolkata`; changing the configuration alone does not add multi-timezone support.

### Run

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## API Usage

| Method | Endpoint | Input |
| --- | --- | --- |
| GET | `/health` | None |
| POST | `/api/v1/appointments/text` | JSON |
| POST | `/api/v1/appointments/image` | Multipart upload |

### Reference datetime

`reference_datetime` is optional.

- Omit it or use JSON `null` to resolve relative dates against the current date/time in `Asia/Kolkata`.
- Supply an ISO-8601 datetime with an explicit timezone offset for reproducible tests.
- In Swagger's image endpoint, leave it blank and disable “Send empty value” when omitting it.
- Do not send the literal placeholder `string`.

The fixed reference datetime below makes the example output reproducible.

### cURL examples

These commands use Bash syntax. PowerShell examples follow.

#### Health

```bash
curl "https://appointment-scheduler-x2hc.onrender.com/health"
```

#### Typed request

```bash
curl -X POST \
  "https://appointment-scheduler-x2hc.onrender.com/api/v1/appointments/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Book dentist next Friday at 3pm",
    "reference_datetime": "2026-09-18T12:00:00+05:30"
  }'
```

#### Clarification

```bash
curl -X POST \
  "https://appointment-scheduler-x2hc.onrender.com/api/v1/appointments/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Book a dentist appointment"}'
```

#### Image upload

Run from the repository root:

```bash
curl -X POST \
  "https://appointment-scheduler-x2hc.onrender.com/api/v1/appointments/image" \
  -F "image=@tests/fixtures/images/appointment_noisy.png;type=image/png" \
  -F "reference_datetime=2026-09-18T12:00:00+05:30"
```

For local testing, replace the deployed base URL with `http://127.0.0.1:8000`.

### Windows PowerShell

#### Typed request

```powershell
$body = @{
    text = "Book dentist next Friday at 3pm"
    reference_datetime = "2026-09-18T12:00:00+05:30"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "https://appointment-scheduler-x2hc.onrender.com/api/v1/appointments/text" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body | ConvertTo-Json -Depth 5
```

#### Image upload

```powershell
curl.exe -X POST `
    "https://appointment-scheduler-x2hc.onrender.com/api/v1/appointments/image" `
    -F "image=@tests/fixtures/images/appointment_noisy.png;type=image/png" `
    -F "reference_datetime=2026-09-18T12:00:00+05:30"
```

### Postman

For text requests, use **POST → Body → raw → JSON** and paste the example request.

For images, use **POST → Body → form-data**:

- `image`: File
- `reference_datetime`: optional Text

Let Postman generate the multipart content-type header.

## Response Contract

| HTTP code | Meaning |
| --- | --- |
| `200` | Scheduling data or clarification required |
| `413` | Image exceeds byte-size or pixel-count limit |
| `415` | Unsupported declared image type |
| `422` | Invalid request, extraction input error, or unreadable image |
| `503` | Gemini or OCR service unavailable |
| `500` | Unexpected internal failure |

Clients must inspect the JSON `status` field. A `200` clarification response is not a successful scheduling result.

Input errors use `status: "error"`. Service-unavailability responses use `status: "service_unavailable"`.

The health endpoint reports process liveness; it does not verify Gemini credentials, quota, or OCR availability.

## Testing

Latest verified local result:

```text
152 passed, 2 warnings in 10.50s
```

The warnings concern dependency deprecations in the test-client stack. No tests failed in that run.

Run all tests:

```powershell
python -m pytest -q
```

Run extractor and grounding tests:

```powershell
python -m pytest tests/test_grounding.py tests/test_extractor.py -q
```

Check installed dependency compatibility:

```powershell
python -m pip check
```

Coverage includes:

- API success, clarification, and error paths
- Real Tesseract extraction from synthetic printed-text fixtures
- Clean, noisy, rotated, and compressed images
- Invalid uploads, size limits, and unavailable OCR
- Configurable low-confidence OCR guardrails
- Department aliases and typo correction
- Date/time normalization and past-appointment rejection
- Gemini response parsing, retries, fallback, timeout, and quota handling
- Cache expiration and result-copy isolation
- Source-grounding regression cases

Extractor tests mock Gemini. Passing tests does not establish live provider availability or universal OCR accuracy.

### Generate fixtures

```powershell
python scripts/generate_test_images.py
```

## Docker

The image installs Tesseract and the Python dependencies.

```powershell
docker build -t appointment-scheduler .
docker run --rm -p 8000:8000 --env-file .env appointment-scheduler
```

Open `http://127.0.0.1:8000/docs`.

The container uses the `PORT` environment variable when supplied, otherwise port `8000`.

## Deployment

The backend is hosted on Render using the Docker configuration.

Configure `GEMINI_API_KEY` in the hosting environment. Local `.env` files are not automatically transferred to the deployed service.

After each deployment, verify:

1. Health
2. Complete typed request
3. Missing-details clarification
4. Image request using an included fixture

Local tests do not substitute for deployed endpoint checks.

## Design Decisions

- Gemini handles language-level extraction; deterministic code handles calendar arithmetic.
- Text and image inputs reuse one processing pipeline.
- Missing or uncertain details request clarification rather than guessed scheduling values.
- Date/time source checks run before normalization.
- Configurable timeouts and retry counts bound individual model attempts.
- Short-lived caching reduces duplicate extraction calls.

## Limitations

- Extraction confidence measures field completeness, not calibrated correctness.
- Tesseract confidence and candidate ranking are heuristics.
- OCR fixtures cover synthetic printed text. Reliable messy-handwriting recognition has not been established.
- Grounding checks phrase support, not complete intent understanding. Alternatives, negation, and conflicting details remain challenging.
- The date/time grounding module does not independently verify department extraction.
- Gemini requires connectivity, valid credentials, available models, and sufficient quota.
- Caching is process-local, resets on restart, and has no global capacity bound.
- Multiple model attempts can make total latency longer than one model's timeout.
- Authentication, application-level rate limiting, persistent bookings, and slot-availability checks are outside this implementation.
- Avoid submitting sensitive patient information to the public demo.