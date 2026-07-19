# ML API Conventions

This document describes the conventions all ML API services follow.
When adding or modifying endpoints, follow these rules so all services feel identical.

## Universal Endpoint Patterns

Every ML service exposes these standardized endpoints:

| Pattern                       | Method | Purpose                                                              |
| ----------------------------- | ------ | -------------------------------------------------------------------- |
| `GET /status`                 | GET    | Health check. Always returns `{"status": "ok", "service": "<name>"}` |
| `GET /jobs`                   | GET    | List all jobs. Returns `{"jobs": [...]}`                             |
| `GET /jobs/{job_id}`          | GET    | Get status of a specific job                                         |
| `DELETE /jobs/{job_id}`       | DELETE | Cancel a job (only if it hasn't completed/failed)                    |
| `GET /jobs/{job_id}/download` | GET    | Download the result file for a completed job                         |

Domain-specific create endpoints (e.g. `POST /tts`, `POST /train`, `POST /generate/video`) vary per service but all return `{"job_id": "...", "status": "queued|pending"}`.

## Workflow

All ML workloads follow the same async pattern:

1. **Submit job** — `POST` to the domain-specific create endpoint (e.g. `/generate/video`, `/train`, `/tts`)
   - Returns `{"job_id": "...", "status": "pending|queued"}` immediately
2. **Poll status** — `GET /jobs/{job_id}`
   - Check `status` field. If `"processing"` or `"preprocessing"` or `"queued"`, keep polling.
   - Check `progress` field (0-100) for progress indication.
3. **Download result** — Once `status` is `"completed"`, call `GET /jobs/{job_id}/download`
   - Returns the result file (video, audio, or model weights depending on service)
4. **Cancel (optional)** — `DELETE /jobs/{job_id}`
   - Only works if job is still queued or processing. Returns error if already completed/failed.

**Exception:** TTS in standalone mode returns audio directly from the create endpoint (no polling needed).

## HTTP Method Rules

- `GET` = read-only (query, list, health check, download)
- `POST` = create jobs or mutate state (set weights, upload files)
- `DELETE` = cancel jobs

**Never use GET for state mutations.** This prevents accidental triggering from browser prefetch, crawlers, etc.

## Error Response Format

All errors use `{"detail": "message"}` (FastAPI HTTPException convention).

## Authorization Model

These are **internal worker APIs** called by the app server, not by end users.
The app server handles authentication. Workers receive `user_id` as a trusted parameter for file organization only.

## Documentation Rules

When you add or modify any endpoint:

1. Update the FastAPI `description=` on the app to describe the sync/async workflow
2. Add a docstring to each endpoint clearly marking it as **ASYNC** or **SYNC**
3. List all possible `status` values and what they mean
4. Describe the step-by-step workflow: submit → poll → download
