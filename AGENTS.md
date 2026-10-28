# Agent instructions

## Repository state

- This workspace began as data-only: `data/raw/` and `gov_myscheme/text_data/` each contain the same 2,876 PDF filenames; `gov_myscheme/text_data/` also contains an empty `test` file. Do not bulk rename, delete, deduplicate, or regenerate these trees without an explicit request.
- The Python pipeline, its HTTP layers, and the React UI live at the repo root, in `server/`, and in `frontend/` (see the stage sections below). There is still no CI workflow, no deployment configuration, and no reproducible Python dependency specification.

## Completed data pipeline

- `data/manifest_all.csv` has been created and validated for 2,876 PDFs representing 2,066 unique SHA-256 contents.
- `data/scheme_metadata.csv` has been created for all 2,066 unique PDFs.
- `data/extracted_text/` contains full text for all 2,066 unique PDFs, with page boundaries preserved; `data/extracted_text_manifest.csv` records the extraction results.
- `data/chunks.jsonl` contains 23,928 deterministic semantic chunks. Chunk coverage, IDs, ordering, metadata, page ranges, size bounds, and source-PDF hash checks all passed.
- Do not modify any PDFs or existing manifests, including `manifest_all.csv`, `scheme_metadata.csv`, or `extracted_text_manifest.csv`.
- `data/embeddings.npy` holds 23,928 x 384 float32 L2-normalized vectors from `intfloat/multilingual-e5-small`, with row-aligned `data/embeddings_metadata.jsonl` and `data/embeddings_config.json`. These are frozen; do not regenerate them.
- `data/vector_index.faiss`, `data/vector_index_metadata.sqlite`, and `data/vector_index_config.json` form the built vector index. `build_vector_index.py` creates it and `verify_vector_index.py` re-checks it (32 checks).

## e5 retrieval contract (must not change)

- Documents were embedded as `passage: Scheme: {scheme_name}` + newline + `{text}`.
- Queries MUST be embedded as `query: {user_question}` and L2-normalized before inner-product search. This asymmetry is recorded in `data/vector_index_config.json` under `retrieval_contract`.

## Retrieval stage

- `retrieval.py` is the only retrieval entrypoint: `E5QueryEncoder` (pinned model + revision from the manifest, `query: ` prefix, L2-normalized), `VectorIndex` (read-only FAISS + SQLite), and `Retriever.search(question, top_k)` returning `RetrievedChunk` rows with similarity.
- FAISS id `==` `chunks.row_id` `==` zero-based row of `embeddings.npy` `==` `embeddings_metadata.jsonl` line minus one.
- CLI: `\.venv\Scripts\python.exe retrieval.py "<question>" --top-k 5 [--json]`.

## Reranking stage

- `rerank.py` adds context assembly and cross-encoder reranking on top of `retrieval.py`: `candidate_k=30` from FAISS, reranked to `final_k=5`.
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` pinned at `1427fd652930e4ba29e8149678df786c240d8825`, run locally on CPU with 4 threads. Raw relevance logits are stored as `reranker_score`.
- No embedding-similarity fallback: if the cross-encoder cannot load, `RerankerError` is raised.
- Public entrypoint: `rerank.retrieve(question, candidate_k=30, final_k=5)`; `RerankPipeline.last_candidates` keeps the 30 intermediates.
- Neighbour expansion (radius 1, `--neighbor-radius 0` disables) only widens `context_text`; it never changes `final_k`. Neighbours must share the same `sha256`, stay within `row_id +/- radius`, are emitted in ascending row order, and are never repeated across blocks.

## Generation stage

- `generation.py` turns the 5 reranked chunks into a grounded answer: `answer(question, candidate_k=30, final_k=5)` returns `AnswerResult` with `answer`, resolved `citations`, `insufficient_context`, and the 5 `chunks` used.
- LLM access is provider- and model-agnostic through any OpenAI-compatible `/chat/completions` endpoint, called with the Python standard library (no new dependency). There are NO defaults: `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` are ALL required. `OpenAICompatibleClient.from_env()` raises `ConfigurationError` naming every missing variable; no provider, endpoint, or model is assumed or substituted, and the key is never logged. `.env.example` documents the three variables (it is not auto-loaded).
- The model returns `{"answer", "citation_ids"}`; any id outside the 5 retrieved chunk ids raises `CitationError`. Citation labels (`[Source: file.pdf, p. 2]`) are built from SQLite metadata only, never from model output.
- Retrieved chunk text is wrapped in `<context>` and declared untrusted in the system prompt.

## HTTP API stage

- `api.py` (FastAPI) is a thin adapter: every request calls `generation.answer(question, candidate_k=30, final_k=5)`. It contains no retrieval, reranking, embedding, or prompt logic and never imports FAISS, the embedding model, or SQLite.
- `POST /answer` accepts `{"question": "..."}` (required, 1–1000 characters after trimming; optional `candidate_k`/`final_k`, `final_k <= candidate_k`) and returns `answer`, `citations`, `citation_labels`, `retrieved_chunk_ids`, `retrieved_chunks` (source/page metadata plus `faiss_similarity`, `reranker_score`, `final_rank`), `insufficient_context`, `model`, `candidate_k`, and `final_k`. Chunk text is deliberately not returned.
- `GET /health` returns `{status, service, llm: {model, configured, base_url_configured}, artifacts}` without loading any model. `llm.configured` is a boolean; the key value is never returned or logged (log lines are redacted).
- Errors use `{"error": {"code", "message"}}`; 422 responses add `details`. Mapping: `ConfigurationError` → 503 `llm_not_configured`, `CitationError` → 502 `invalid_citations`, `LLMRequestError` → 502 `llm_unavailable`, `LLMResponseError` → 502 `llm_invalid_response`, `GenerationError` → 500 `generation_failed`, `RetrievalError` → 503 `retrieval_unavailable`, anything else → 500 `internal_error`.
- `server/` is a Node/Express gateway exposing `POST /api/answer` and `GET /api/health`. It validates input, forwards only the question to `PYTHON_API_URL`, enables CORS, and normalises errors: validation → 400, Python 4xx passed through, Python 5xx → 502 with the upstream `error.code` preserved, unreachable → 503 `python_api_unavailable`, timeout → 504 `python_api_timeout`. It must never import FAISS/embedding/SQLite code or hold LLM credentials.
- Gateway env: `PYTHON_API_URL` (default `http://localhost:8000`), `PYTHON_API_TIMEOUT_MS` (180000), `PYTHON_HEALTH_TIMEOUT_MS` (5000), `CORS_ORIGIN` (`*`), `PORT` (3000), `HOST` (0.0.0.0). Python runner env: `API_HOST` / `API_PORT`; the LLM vars are unchanged.
- Run the Python API with `\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000`; run the gateway from `server/` with `npm install` then `npm start`.
- The `fastapi`, `uvicorn`, and `httpx` versions installed in `.venv` are environment state, not a pinned dependency specification.

## Frontend stage

- `frontend/` is a Vite + React 19 single-page app: `index.html` → `src/main.jsx` → `src/App.jsx`. Components are `QuestionForm` (textarea, microphone, Ask/Clear, Enter and Ctrl/Cmd+Enter submit, Shift+Enter newline), `AnswerPanel` (formatted answer sections, Read Answer/Stop, `insufficient_context` notice, grounded-source indicator), `CitationCard` (scheme name, source PDF, page or page range, citation label), and `ErrorBanner` (message, error code, retry/dismiss). Styling is plain responsive CSS in `src/styles.css`.
- `src/api/client.js` is the ONLY module that calls the backend: `POST ${VITE_API_BASE_URL}/api/answer` with `{"question": "..."}`. It returns the gateway payload unchanged, turns network failures and `{"error": {code, message}}` bodies into `ApiError`, re-throws aborts, and rejects malformed payloads. Components never hardcode the backend URL.
- Backend URL comes from `VITE_API_BASE_URL` (default `http://localhost:3000`). Only `VITE_*` values reach the browser bundle, so no secret may ever be placed there; the frontend never sees `LLM_API_KEY`, FAISS, embeddings, or Python.
- Behaviour: Ask is disabled while loading; the previous request is aborted when a new one starts or Clear is pressed; empty/whitespace questions are rejected locally without a network call; API/network errors render a readable message and do not clear the question.
- `src/voice/speech.js` is the ONLY module using browser speech APIs, with no external voice service, voice model, or new dependency. Input uses `SpeechRecognition`/`webkitSpeechRecognition`; only final transcripts are appended to the textarea (never auto-submitted), interim speech is display-only, and unsupported browsers, denied permission, and recognition errors show readable messages. Output uses `window.speechSynthesis`; "Read Answer" speaks only `result.answer` (never citations or metadata), always cancels existing speech first, and never speaks automatically. "Stop" cancels playback.
- Run: `cd frontend`, `npm install`, then `npm run dev` (http://localhost:5173) with the Node gateway on port 3000. Copy `.env.example` to `.env.local` to point at another gateway. `npm run build` produces `frontend/dist/`.
- There is no authentication and no deployment configuration. Voice input requires a browser that supports the Web Speech API and microphone permission; text input remains fully usable regardless.

## Not implemented

- The Streamlit UI, authentication, and deployment setup do not exist. Do not add them without an explicit request.

## Verification

- No automated test suite is defined for the data pipeline. For Python changes, use the existing environment when available: `\.venv\Scripts\python.exe -m py_compile <changed-file.py>`; do not treat the environment as a reproducible dependency specification.
- To re-check the vector index after any change: `\.venv\Scripts\python.exe verify_vector_index.py` (exits non-zero on failure). Rebuilding requires `--overwrite` and must keep the index byte-identical.
- To re-check retrieval: `\.venv\Scripts\python.exe verify_retrieval.py` (30 checks; exits non-zero on failure).
- To re-check context assembly + reranking: `\.venv\Scripts\python.exe verify_rerank.py` (48 checks; exits non-zero on failure).
- To re-check grounded generation: `\.venv\Scripts\python.exe verify_generation.py` (44 checks, mocked LLM; the live smoke test runs only when `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` are all set).
- To re-check the HTTP layer: `\.venv\Scripts\python.exe verify_api.py` (89 checks, mocked `generation.answer`; also asserts the frozen pipeline files and RAG artifacts are byte-identical). Node gateway tests: `npm test` in `server/` (14 checks, stubbed Python service).
- Frontend tests: `npm test` in `frontend/` (55 Vitest + Testing Library checks: rendering, submit, loading, formatted answers, citations, errors, empty-input guard, keyboard, clear, source-hygiene guards, and voice input/output with mocked Web Speech APIs). Also run `npm run build` to prove the production bundle compiles.

## Git boundary

- This directory has no nested `.git`; `git rev-parse --show-toplevel` resolves to `C:\Users\navya\Desktop`, which contains unrelated user files. Keep Git commands path-scoped to `scheme-voice-assistant/` and never stage, clean, reset, or otherwise modify the parent workspace broadly.
