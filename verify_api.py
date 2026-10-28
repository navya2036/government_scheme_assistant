"""Verification tests for the HTTP API layer.

The LLM, reranker, and FAISS index are mocked out at the ``generation.answer``
boundary, so these tests need no API key, no models, and no network.  What is
actually checked:

* the FastAPI contract (routes, status codes, request validation)
* that ``POST /answer`` really calls ``generation.answer`` with the question
* that the response keeps answer, citations, chunk ids, source/page metadata,
  and reranker scores
* that pipeline failures map to stable, sanitized error responses
* that no secret material ever appears in a response
* that ``api.py`` contains no retrieval/rerank/embedding logic
* that the Node gateway never touches FAISS, embeddings, or the LLM key
* that the frozen RAG artifacts and pipeline modules are byte-identical
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi.testclient import TestClient

import api
import generation
import retrieval

EXAMPLE_QUESTION = "What financial assistance is available for old age pension?"

# Baselines captured when the API stage was added.  These files are frozen by
# project convention: any change here is a regression, not a refactor.
FROZEN_CODE_SHA256 = {
    "generate_embeddings.py": "59ea4bd89197a2ed14a94d9c783dfd828a31911329a4ef652ad0a75f378f8893",
    "build_vector_index.py": "0b3cf6b430c956a65942c611b32657b77ee7a88530f17fad0f0b27eed67fb96b",
    "verify_vector_index.py": "5ed1ec768157fd9c97d14260e4d8dd6d8a518319eebdaeee4d6581f1b6bb0acf",
    "retrieval.py": "3460e4af4579a9e31eb79ea883fe65de61efc1e104f63a49e6ee058caaba5003",
    "rerank.py": "1d346a219bf0fab4e2af17d10ac803c446456190d59a7de91ce434c927064ee2",
    "verify_retrieval.py": "f4d9f9e9c8653937974005400850645f5526be2a4ed13cfefe5a438f2f44b305",
    "verify_rerank.py": "6896587a84eb1f8a4abd1a51143fe8a2a2d58d9efc1fa35b6f86ba716db740e5",
    "generation.py": "16abacbbd0cfe49eeae12576b6bc9a3d683bcc2e1d764c3c26bc7c5f39da9ab0",
    "verify_generation.py": "7bfbb621fa327fbe7107d9124768ef54858d12365ba7d900606a8f26fda3721a",
    "data/embeddings_config.json": "06772d0de3c088ff351326277a1862b1895b303e769515764f04d6bff300f58f",
    "data/vector_index_config.json": "e90dfb6324d8b3d6c0f6259e14b4ddc2d2778e46003e6e71bb1d5967830cf252",
    "data/vector_index_metadata.sqlite": "c454104c164a625cbe0d2db74db823b5df718e1fd8235267b9367300619e7252",
}

FORBIDDEN_API_TOKENS = (
    "import faiss",
    "faiss.",
    "sentence_transformers",
    "SentenceTransformer",
    "IndexFlatIP",
    "np.load",
    "numpy",
    "embedding_model",
    "rerank.retrieve",
    "Retriever(",
    "build_vector_index",
    "chat/completions",
    "Authorization",
)

# Matched against the gateway source.  Patterns (not bare words) so that prose
# comments such as "never touches FAISS, embeddings, SQLite" do not trip them.
NODE_FORBIDDEN_PATTERNS = (
    (
        r"""(?:require|import)\s*\(\s*['"](?:faiss|numpy|sentence-transformers|better-sqlite3)""",
        "no FAISS/embedding/SQLite imports",
    ),
    (
        r"vector_index\.faiss|embeddings\.npy|\.sqlite\b|IndexFlatIP|sentence_transformers",
        "no vector-index file access",
    ),
    (
        r"\bLLM_API_KEY\b|\bOPENAI_API_KEY\b|sk-[A-Za-z0-9]",
        "no LLM credentials",
    ),
)


class Checker:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: Any, detail: str = "") -> bool:
        passed = bool(condition)
        self.results.append((name, passed, detail))
        return passed

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.results)

    def report(self) -> None:
        width = max(len(name) for name, _, _ in self.results)
        for name, passed, detail in self.results:
            print(f"[{'PASS' if passed else 'FAIL'}] {name.ljust(width)}  {detail}")
        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"\n{passed}/{len(self.results)} checks passed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_chunks(count: int) -> list[generation.ContextChunk]:
    return [
        generation.ContextChunk(
            chunk_id=f"{rank:02d}" + "a" * 62,
            scheme_name=f"Scheme {rank}",
            department="Social Welfare",
            state="All India",
            source_file=f"scheme_{rank}.pdf",
            page_start=rank,
            page_end=rank + 1,
            text=f"raw text {rank}",
            context_text=f"neighbour context {rank}",
            faiss_similarity=0.90 - rank * 0.01,
            reranker_score=8.0 - rank,
            final_rank=rank,
        )
        for rank in range(1, count + 1)
    ]


def make_result(
    question: str, candidate_k: int = 30, final_k: int = 5
) -> generation.AnswerResult:
    chunks = build_chunks(final_k)
    citations = [
        generation.Citation(
            chunk_id=chunks[0].chunk_id,
            scheme_name=chunks[0].scheme_name,
            source_file=chunks[0].source_file,
            page_start=chunks[0].page_start,
            page_end=chunks[0].page_end,
            label=generation.format_citation_label(
                chunks[0].source_file, chunks[0].page_start, chunks[0].page_end
            ),
        )
    ]
    return generation.AnswerResult(
        question=question,
        answer="Old age pension schemes provide periodic financial assistance.",
        citations=citations,
        insufficient_context=False,
        chunks=chunks,
        candidate_k=candidate_k,
        final_k=final_k,
        model=generation.DEFAULT_MODEL,
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


@contextlib.contextmanager
def patched_answer(replacement: Any) -> Iterator[list[dict[str, Any]]]:
    """Replace ``generation.answer`` for the duration of the block.

    ``api.py`` calls the function through the module object, so patching the
    module attribute proves the endpoint routes through the real name.
    """

    original = generation.answer
    calls: list[dict[str, Any]] = []
    generation.answer = replacement(original, calls)  # type: ignore[assignment]
    try:
        yield calls
    finally:
        generation.answer = original  # type: ignore[assignment]


def recording_stub(original: Any, calls: list[dict[str, Any]]) -> Any:
    def stub(question: str, candidate_k: int = 30, final_k: int = 5) -> Any:
        calls.append(
            {"question": question, "candidate_k": candidate_k, "final_k": final_k}
        )
        return make_result(question, candidate_k, final_k)

    return stub


def raising_stub(error: BaseException) -> Any:
    def factory(original: Any, calls: list[dict[str, Any]]) -> Any:
        def stub(*args: Any, **kwargs: Any) -> Any:
            calls.append({"args": args, "kwargs": kwargs})
            raise error

        return stub

    return factory


def json_body(response: Any) -> dict[str, Any]:
    return response.json()


def assert_error_shape(checker: Checker, name: str, body: dict[str, Any]) -> None:
    error = body.get("error")
    checker.check(
        f"{name} uses {{error: {{code, message}}}} shape",
        isinstance(error, dict)
        and isinstance(error.get("code"), str)
        and isinstance(error.get("message"), str),
        str(error)[:90],
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the HTTP API layer.")
    parser.add_argument("--manifest", type=Path, default=Path("data/vector_index_config.json"))
    parser.add_argument("--index", type=Path, default=Path("data/vector_index.faiss"))
    parser.add_argument("--database", type=Path, default=Path("data/vector_index_metadata.sqlite"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument(
        "--embedding-metadata", type=Path, default=Path("data/embeddings_metadata.jsonl")
    )
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument(
        "--server-dir", type=Path, default=Path("server"), help="Node gateway directory."
    )
    args = parser.parse_args(argv)

    checker = Checker()
    client = TestClient(api.app)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    api_logger = logging.getLogger("scheme_api")
    api_logger.propagate = False
    api_logger.addHandler(logging.NullHandler())

    # --- routing ---------------------------------------------------------
    routes = {
        (route.path, tuple(sorted(route.methods)))
        for route in api.app.routes
        if getattr(route, "methods", None)
    }
    checker.check(
        "POST /answer is registered",
        ("/answer", ("POST",)) in routes,
        "/answer POST",
    )
    checker.check(
        "GET /health is registered",
        ("/health", ("GET",)) in routes,
        "/health GET",
    )
    openapi_paths = api.app.openapi().get("paths", {})
    checker.check(
        "OpenAPI documents both endpoints",
        "post" in openapi_paths.get("/answer", {})
        and "get" in openapi_paths.get("/health", {}),
        ", ".join(sorted(openapi_paths)),
    )
    checker.check(
        "answer handler is sync (runs in FastAPI threadpool)",
        not inspect.iscoroutinefunction(api.answer_endpoint),
        "def answer_endpoint",
    )

    # --- happy path ------------------------------------------------------
    with patched_answer(recording_stub) as calls:
        response = client.post(
            "/answer", json={"question": f"   {EXAMPLE_QUESTION}   "}
        )
    body = json_body(response)

    checker.check("valid question returns 200", response.status_code == 200, str(response.status_code))
    checker.check(
        "generation.answer() is called exactly once",
        len(calls) == 1,
        f"{len(calls)} call(s)",
    )
    checker.check(
        "question is forwarded to generation.answer()",
        calls and calls[0]["question"] == EXAMPLE_QUESTION,
        str(calls[0]["question"])[:60] if calls else "no call",
    )
    checker.check(
        "question is stripped before the pipeline call",
        calls and calls[0]["question"] == calls[0]["question"].strip(),
        "no leading/trailing space",
    )
    checker.check(
        "defaults candidate_k=30 / final_k=5 are forwarded",
        calls and calls[0]["candidate_k"] == 30 and calls[0]["final_k"] == 5,
        f"candidate_k={calls[0]['candidate_k'] if calls else '?'} "
        f"final_k={calls[0]['final_k'] if calls else '?'}",
    )
    checker.check(
        "answer text is preserved",
        body.get("answer") == make_result(EXAMPLE_QUESTION).answer,
        body.get("answer", "")[:50],
    )

    citations = body.get("citations") or []
    checker.check(
        "citations contain chunk id, file, and pages",
        len(citations) == 1
        and citations[0].get("chunk_id") == build_chunks(5)[0].chunk_id
        and citations[0].get("source_file") == "scheme_1.pdf"
        and citations[0].get("page_start") == 1
        and citations[0].get("page_end") == 2,
        str(citations[0])[:80] if citations else "none",
    )
    checker.check(
        "citation label is rendered from metadata",
        citations and citations[0].get("label") == "[Source: scheme_1.pdf, pp. 1-2]",
        citations[0].get("label", "") if citations else "none",
    )
    checker.check(
        "citation_labels mirror the citations",
        body.get("citation_labels") == [c["label"] for c in citations],
        str(body.get("citation_labels"))[:60],
    )

    expected_ids = [chunk.chunk_id for chunk in build_chunks(5)]
    checker.check(
        "retrieved_chunk_ids preserve rank order",
        body.get("retrieved_chunk_ids") == expected_ids,
        f"{len(body.get('retrieved_chunk_ids') or [])} ids",
    )

    chunks = body.get("retrieved_chunks") or []
    first = chunks[0] if chunks else {}
    checker.check(
        "retrieved chunks expose source/page metadata",
        len(chunks) == 5
        and first.get("source_file") == "scheme_1.pdf"
        and first.get("page_start") == 1
        and first.get("page_end") == 2,
        f"{len(chunks)} chunks",
    )
    checker.check(
        "reranker scores are returned",
        abs(first.get("reranker_score", 0.0) - 7.0) < 1e-9,
        str(first.get("reranker_score")),
    )
    checker.check(
        "FAISS similarities are returned",
        abs(first.get("faiss_similarity", 0.0) - 0.89) < 1e-9,
        str(first.get("faiss_similarity")),
    )
    checker.check(
        "final ranks are returned",
        [chunk.get("final_rank") for chunk in chunks] == [1, 2, 3, 4, 5],
        str([chunk.get("final_rank") for chunk in chunks]),
    )
    checker.check(
        "chunk text is not echoed by the API",
        all("text" not in chunk and "context_text" not in chunk for chunk in chunks),
        "metadata only",
    )
    checker.check(
        "insufficient_context is preserved",
        body.get("insufficient_context") is False,
        str(body.get("insufficient_context")),
    )
    checker.check(
        "model and k values are echoed",
        body.get("model") == generation.DEFAULT_MODEL
        and body.get("candidate_k") == 30
        and body.get("final_k") == 5,
        f"{body.get('model')} / {body.get('candidate_k')} / {body.get('final_k')}",
    )

    # --- explicit k overrides -------------------------------------------
    with patched_answer(recording_stub) as calls:
        override = client.post(
            "/answer", json={"question": EXAMPLE_QUESTION, "candidate_k": 20, "final_k": 3}
        )
    override_body = json_body(override)
    checker.check(
        "candidate_k/final_k overrides are honoured",
        override.status_code == 200
        and calls
        and calls[0]["candidate_k"] == 20
        and calls[0]["final_k"] == 3
        and override_body.get("final_k") == 3
        and len(override_body.get("retrieved_chunks") or []) == 3,
        f"{override.status_code}, {len(override_body.get('retrieved_chunks') or [])} chunks",
    )

    # --- validation ------------------------------------------------------
    invalid_payloads = {
        "empty question": {"question": ""},
        "whitespace-only question": {"question": "   "},
        "missing question": {},
        "non-string question": {"question": ["a", "b"]},
        "question over 1000 chars": {"question": "x" * 1001},
        "unexpected field": {"question": EXAMPLE_QUESTION, "top_k": 9},
        "candidate_k below 1": {"question": EXAMPLE_QUESTION, "candidate_k": 0},
        "final_k greater than candidate_k": {
            "question": EXAMPLE_QUESTION,
            "candidate_k": 3,
            "final_k": 4,
        },
    }
    with patched_answer(recording_stub) as calls:
        for name, payload in invalid_payloads.items():
            response = client.post("/answer", json=payload)
            body = json_body(response)
            checker.check(
                f"rejects {name} with 422",
                response.status_code == 422,
                str(response.status_code),
            )
            assert_error_shape(checker, name, body)
        checker.check(
            "rejected requests never reach generation.answer()",
            len(calls) == 0,
            f"{len(calls)} calls",
        )

    response = client.post("/answer", json={"question": "   "})
    details = (json_body(response).get("error") or {}).get("details") or []
    checker.check(
        "validation errors name the offending field",
        any(detail.get("field") == "question" for detail in details),
        str(details)[:90],
    )

    with patched_answer(recording_stub) as calls:
        boundary = client.post("/answer", json={"question": "y" * 1000})
    checker.check(
        "1000-character question is accepted",
        boundary.status_code == 200 and len(calls) == 1,
        str(boundary.status_code),
    )

    # --- pipeline error mapping -----------------------------------------
    error_cases = (
        (
            "missing LLM key",
            generation.ConfigurationError("LLM_API_KEY is not set"),
            503,
            "llm_not_configured",
        ),
        (
            "fabricated citations",
            generation.CitationError("citation id not in context"),
            502,
            "invalid_citations",
        ),
        (
            "LLM transport failure",
            generation.LLMRequestError("connection reset"),
            502,
            "llm_unavailable",
        ),
        (
            "malformed LLM output",
            generation.LLMResponseError("invalid JSON"),
            502,
            "llm_invalid_response",
        ),
        (
            "generic generation failure",
            generation.GenerationError("boom"),
            500,
            "generation_failed",
        ),
        (
            "retrieval failure",
            retrieval.RetrievalError("index missing"),
            503,
            "retrieval_unavailable",
        ),
        (
            "unexpected exception",
            RuntimeError("unexpected boom"),
            500,
            "internal_error",
        ),
    )

    for label, error, expected_status, expected_code in error_cases:
        with patched_answer(raising_stub(error)):
            response = client.post("/answer", json={"question": EXAMPLE_QUESTION})
        body = json_body(response)
        checker.check(
            f"{label} maps to {expected_status} {expected_code}",
            response.status_code == expected_status
            and (body.get("error") or {}).get("code") == expected_code,
            f"{response.status_code} {((body.get('error') or {}).get('code'))}",
        )
        assert_error_shape(checker, label, body)

    leaked = "sk-live-9f3ab7c2d1e4"
    original_key = os.environ.get("LLM_API_KEY")
    os.environ["LLM_API_KEY"] = leaked
    try:
        with patched_answer(
            raising_stub(generation.ConfigurationError(f"bad key {leaked}"))
        ):
            response = client.post("/answer", json={"question": EXAMPLE_QUESTION})
        raw = response.text
        checker.check(
            "API key is never echoed in error responses",
            leaked not in raw and "sk-" not in raw,
            "sanitized message",
        )
        checker.check(
            "no traceback is returned to clients",
            "Traceback" not in raw and "generation.py" not in raw,
            "no internals",
        )

        stream = io.StringIO()
        capture = logging.StreamHandler(stream)
        api_logger.addHandler(capture)
        try:
            with patched_answer(raising_stub(RuntimeError(f"trace leaks {leaked}"))):
                client.post("/answer", json={"question": EXAMPLE_QUESTION})
            with patched_answer(
                raising_stub(generation.ConfigurationError(f"bad {leaked}"))
            ):
                client.post("/answer", json={"question": EXAMPLE_QUESTION})
        finally:
            api_logger.removeHandler(capture)
        emitted = stream.getvalue()
        checker.check(
            "API key never appears in logs",
            leaked not in emitted and "sk-" not in emitted and "***" in emitted,
            "redacted log lines",
        )

        # --- health ------------------------------------------------------
        health = client.get("/health")
        health_body = json_body(health)
        checker.check(
            "GET /health returns 200 ok",
            health.status_code == 200 and health_body.get("status") == "ok",
            f"{health.status_code} {health_body.get('status')}",
        )
        checker.check(
            "health reports all artifacts present",
            all((health_body.get("artifacts") or {}).values())
            and len(health_body.get("artifacts") or {}) == 3,
            str(health_body.get("artifacts")),
        )
        checker.check(
            "health reports the model from LLM_MODEL",
            (health_body.get("llm") or {}).get("model")
            == (os.environ.get("LLM_MODEL") or generation.DEFAULT_MODEL).strip(),
            str((health_body.get("llm") or {}).get("model")),
        )
        checker.check(
            "health reports LLM configuration as a boolean",
            (health_body.get("llm") or {}).get("configured") is True,
            str((health_body.get("llm") or {}).get("configured")),
        )
        checker.check(
            "health never includes the key value",
            leaked not in health.text and "sk-" not in health.text,
            "boolean only",
        )
    finally:
        if original_key is None:
            os.environ.pop("LLM_API_KEY", None)
        else:
            os.environ["LLM_API_KEY"] = original_key

    health_body = json_body(client.get("/health"))
    expected_configured = bool((os.environ.get("LLM_API_KEY") or "").strip())
    checker.check(
        "health reflects whether LLM_API_KEY is set",
        (health_body.get("llm") or {}).get("configured") is expected_configured,
        f"configured={health_body.get('llm', {}).get('configured')}",
    )

    missing_index = api.DEFAULT_INDEX
    api.DEFAULT_INDEX = Path("data/__missing_index__.faiss")
    try:
        degraded = client.get("/health")
    finally:
        api.DEFAULT_INDEX = missing_index
    checker.check(
        "health degrades when an artifact is missing",
        degraded.status_code == 503
        and json_body(degraded).get("status") == "degraded"
        and json_body(degraded)["artifacts"].get("vector_index") is False,
        f"{degraded.status_code} {json_body(degraded).get('status')}",
    )

    # --- no duplicated pipeline logic ------------------------------------
    api_source = Path("api.py").read_text(encoding="utf-8")
    offenders = [token for token in FORBIDDEN_API_TOKENS if token in api_source]
    checker.check(
        "api.py contains no retrieval/rerank/LLM transport logic",
        not offenders,
        ", ".join(offenders) or "clean adapter",
    )
    checker.check(
        "api.py delegates to generation.answer()",
        "generation.answer(" in api_source,
        "generation.answer(…)",
    )

    server_dir = args.server_dir
    node_source = (server_dir / "src" / "server.js").read_text(encoding="utf-8")
    node_offenders = [
        label
        for pattern, label in NODE_FORBIDDEN_PATTERNS
        if re.search(pattern, node_source)
    ]
    checker.check(
        "Node gateway never touches FAISS, embeddings, or LLM keys",
        not node_offenders,
        ", ".join(node_offenders) or "HTTP forwarding only",
    )
    checker.check(
        "Node forwards to PYTHON_API_URL",
        "PYTHON_API_URL" in node_source and "/answer" in node_source,
        "PYTHON_API_URL + /answer",
    )
    checker.check(
        "Node enables CORS",
        "cors(" in node_source,
        "cors middleware",
    )
    checker.check(
        "Node question limit matches the Python API",
        f"const MAX_QUESTION_LENGTH = {api.MAX_QUESTION_LENGTH};" in node_source,
        str(api.MAX_QUESTION_LENGTH),
    )

    package = json.loads((server_dir / "package.json").read_text(encoding="utf-8"))
    checker.check(
        "Node dependencies stay minimal (express + cors)",
        set(package.get("dependencies", {})) == {"express", "cors"},
        ", ".join(sorted(package.get("dependencies", {}))),
    )

    # --- frozen code -----------------------------------------------------
    for relative, expected_hash in FROZEN_CODE_SHA256.items():
        path = Path(relative)
        checker.check(
            f"{relative} unchanged",
            path.is_file() and sha256_file(path) == expected_hash,
            "sha256 baseline",
        )

    # --- artifacts -------------------------------------------------------
    artifact_hashes = (
        ("vector_index.faiss", args.index, manifest["index_sha256"]),
        ("embeddings.npy", args.embeddings, manifest["embeddings_sha256"]),
        ("chunks.jsonl", args.chunks, manifest["chunks_sha256"]),
        (
            "embeddings_metadata.jsonl",
            args.embedding_metadata,
            manifest["embedding_metadata_sha256"],
        ),
    )
    for label, path, expected_hash in artifact_hashes:
        checker.check(
            f"{label} byte-identical",
            sha256_file(path) == expected_hash,
            "sha256 matches manifest",
        )
    checker.check(
        "vector_index_metadata.sqlite byte-identical",
        sha256_file(args.database)
        == FROZEN_CODE_SHA256["data/vector_index_metadata.sqlite"],
        "sha256 baseline",
    )

    checker.report()
    return 0 if checker.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
