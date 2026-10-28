"""HTTP API layer over the existing RAG pipeline.

This module only adapts the verified pipeline to HTTP.  It contains no
retrieval, reranking, embedding, or prompting logic -- every request is handed
to :func:`generation.answer`, which is the single source of truth.

Layer position::

    React (future) -> Node/Express -> this FastAPI app -> generation.py
                   -> rerank.py -> retrieval.py

Configuration is inherited from ``generation.py`` (``LLM_API_KEY``,
``LLM_BASE_URL``, ``LLM_MODEL``).  Secrets are never echoed in any response.
"""

from __future__ import annotations

import logging
import os
import re
import traceback
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

import generation
import retrieval
from retrieval import DEFAULT_DATABASE, DEFAULT_INDEX, DEFAULT_MANIFEST, EXPECTED_COUNT

logger = logging.getLogger("scheme_api")

MAX_QUESTION_LENGTH = 1000
DEFAULT_CANDIDATE_K = 30
DEFAULT_FINAL_K = 5
MAX_FINAL_K = 50


class AnswerRequest(BaseModel):
    """Request body for ``POST /answer``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="Natural-language question about the schemes.",
        examples=["What financial assistance is available for old age pension?"],
    )
    candidate_k: int = Field(DEFAULT_CANDIDATE_K, ge=1, le=EXPECTED_COUNT)
    final_k: int = Field(DEFAULT_FINAL_K, ge=1, le=MAX_FINAL_K)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty or whitespace")
        return stripped

    @model_validator(mode="after")
    def _final_within_candidates(self) -> "AnswerRequest":
        if self.final_k > self.candidate_k:
            raise ValueError("final_k must not exceed candidate_k")
        return self


class CitationOut(BaseModel):
    chunk_id: str
    scheme_name: str
    source_file: str
    page_start: int
    page_end: int
    label: str


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    scheme_name: str
    source_file: str
    page_start: int
    page_end: int
    final_rank: int
    faiss_similarity: float
    reranker_score: float


class AnswerResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    citation_labels: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_chunks: list[RetrievedChunkOut]
    insufficient_context: bool
    model: str
    candidate_k: int
    final_k: int


class HealthResponse(BaseModel):
    status: str
    service: str
    llm: dict[str, Any]
    artifacts: dict[str, Any]


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{6,}")


def _redact(text: str) -> str:
    """Strip credentials from log lines (the configured key and ``sk-`` tokens)."""

    cleaned = str(text)
    secret = (os.environ.get("LLM_API_KEY") or "").strip()
    if secret:
        cleaned = cleaned.replace(secret, "***")
    return _SECRET_PATTERN.sub("sk-***", cleaned)


def _artifact_status() -> tuple[dict[str, Any], bool]:
    checks = {
        "vector_index": DEFAULT_INDEX,
        "metadata_database": DEFAULT_DATABASE,
        "index_manifest": DEFAULT_MANIFEST,
    }
    state = {name: path.is_file() for name, path in checks.items()}
    return state, all(state.values())


def _llm_status() -> dict[str, Any]:
    """Describe LLM configuration without ever reading back the key value."""

    return {
        "model": (os.environ.get("LLM_MODEL") or generation.DEFAULT_MODEL).strip(),
        "configured": bool((os.environ.get("LLM_API_KEY") or "").strip()),
        "base_url_configured": bool((os.environ.get("LLM_BASE_URL") or "").strip()),
    }


app = FastAPI(
    title="Scheme Voice Assistant API",
    version="0.1.0",
    description=(
        "Grounded question answering over the indexed government scheme corpus. "
        "Thin HTTP adapter over generation.answer()."
    ),
)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return one stable error shape instead of FastAPI's default ``detail`` list."""

    details = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
        details.append(
            {"field": location or "body", "message": error.get("msg", "invalid value")}
        )
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "invalid_request",
                "message": "Request validation failed.",
                "details": details,
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def _http_error_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    if exc.status_code == 404:
        return _error(404, "not_found", "Unknown endpoint.")
    return _error(exc.status_code, "http_error", str(exc.detail))


@app.get("/health", response_model=HealthResponse)
def health() -> JSONResponse:
    """Liveness/readiness without loading any model."""

    artifacts, ok = _artifact_status()
    payload = HealthResponse(
        status="ok" if ok else "degraded",
        service="scheme-voice-assistant-api",
        llm=_llm_status(),
        artifacts=artifacts,
    )
    return JSONResponse(status_code=200 if ok else 503, content=payload.model_dump())


@app.post(
    "/answer",
    response_model=AnswerResponse,
    responses={
        422: {"description": "Invalid or empty question"},
        502: {"description": "LLM failed or returned unusable output"},
        503: {"description": "LLM not configured, or retrieval artifacts unavailable"},
        500: {"description": "Unexpected server error"},
    },
)
def answer_endpoint(payload: AnswerRequest) -> JSONResponse:
    """Answer a question using the verified pipeline."""

    try:
        result = generation.answer(
            payload.question,
            candidate_k=payload.candidate_k,
            final_k=payload.final_k,
        )
    except generation.ConfigurationError as error:
        logger.warning("LLM not configured: %s", _redact(f"{type(error).__name__}: {error}"))
        return _error(
            503,
            "llm_not_configured",
            "The answer service is not configured (LLM_API_KEY is missing or invalid).",
        )
    except generation.CitationError as error:
        logger.warning(
            "Rejected model citations: %s", _redact(f"{type(error).__name__}: {error}")
        )
        return _error(
            502,
            "invalid_citations",
            "The language model cited sources that were not provided; the answer was discarded.",
        )
    except generation.LLMRequestError as error:
        logger.warning(
            "LLM request failed: %s", _redact(f"{type(error).__name__}: {error}")
        )
        return _error(502, "llm_unavailable", "The language model service is unavailable.")
    except generation.LLMResponseError as error:
        logger.warning(
            "Unusable LLM response: %s", _redact(f"{type(error).__name__}: {error}")
        )
        return _error(502, "llm_invalid_response", "The language model returned an unusable response.")
    except generation.GenerationError as error:
        logger.warning("Generation failed: %s", _redact(f"{type(error).__name__}: {error}"))
        return _error(500, "generation_failed", "Answer generation failed.")
    except retrieval.RetrievalError as error:
        logger.error("Retrieval failed: %s", _redact(f"{type(error).__name__}: {error}"))
        return _error(503, "retrieval_unavailable", "The retrieval index is unavailable.")
    except Exception as error:  # pragma: no cover - defensive
        logger.error(
            "Unexpected error while answering:\n%s", _redact(traceback.format_exc())
        )
        return _error(500, "internal_error", "Unexpected server error.")

    body = AnswerResponse(
        answer=result.answer,
        citations=[
            CitationOut(
                chunk_id=citation.chunk_id,
                scheme_name=citation.scheme_name,
                source_file=citation.source_file,
                page_start=citation.page_start,
                page_end=citation.page_end,
                label=citation.label,
            )
            for citation in result.citations
        ],
        citation_labels=[citation.label for citation in result.citations],
        retrieved_chunk_ids=[chunk.chunk_id for chunk in result.chunks],
        retrieved_chunks=[
            RetrievedChunkOut(
                chunk_id=chunk.chunk_id,
                scheme_name=chunk.scheme_name,
                source_file=chunk.source_file,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                final_rank=chunk.final_rank,
                faiss_similarity=chunk.faiss_similarity,
                reranker_score=chunk.reranker_score,
            )
            for chunk in result.chunks
        ],
        insufficient_context=result.insufficient_context,
        model=result.model,
        candidate_k=result.candidate_k,
        final_k=result.final_k,
    )
    return JSONResponse(status_code=200, content=body.model_dump())


def main() -> int:  # pragma: no cover - convenience runner
    import uvicorn

    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
