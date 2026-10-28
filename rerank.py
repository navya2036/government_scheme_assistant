"""Context assembly and cross-encoder reranking over the retrieval stage.

Pipeline implemented here::

    question
      -> e5 query encoder (unchanged, from retrieval.py)
      -> FAISS IndexFlatIP           (candidate_k = 30)
      -> context assembly            (bounded, deterministic neighbour expansion)
      -> multilingual cross-encoder  (rerank the 30 candidates)
      -> final_k = 5 chunks

The reranker is a *second-stage relevance model*.  It never touches the
embedding model, the embeddings, the FAISS index, or the metadata database, and
there is no fallback to embedding similarity: if the cross-encoder cannot be
loaded, :class:`RerankerError` is raised.

Adjacency expansion is optional, bounded and documented in
:meth:`ContextAssembler.assemble`.  It never changes ``final_k``: it only widens
the text attached to each of the final chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from retrieval import (
    DEFAULT_DATABASE,
    DEFAULT_INDEX,
    DEFAULT_MANIFEST,
    RetrievalError,
    Retriever,
    RetrievedChunk,
)

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"

DEFAULT_CANDIDATE_K = 30
DEFAULT_FINAL_K = 5
DEFAULT_NEIGHBOR_RADIUS = 1
DEFAULT_RERANK_BATCH_SIZE = 30
DEFAULT_TORCH_THREADS = 4
CONTEXT_SEPARATOR = "\n\n"


class RerankerError(RetrievalError):
    """Raised when the reranker cannot be loaded or used.  Never downgraded."""


@dataclass(frozen=True)
class Candidate:
    """One of the ``candidate_k`` chunks returned by FAISS, before reranking."""

    chunk_id: str
    scheme_name: str
    department: str
    state: str
    beneficiaries: str
    source_file: str
    page_start: int
    page_end: int
    text: str
    faiss_similarity: float
    reranker_score: float
    faiss_rank: int
    row_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RerankedChunk:
    """A final chunk, ordered by reranker relevance, with assembled context."""

    chunk_id: str
    scheme_name: str
    department: str
    state: str
    beneficiaries: str
    source_file: str
    page_start: int
    page_end: int
    text: str
    faiss_similarity: float
    reranker_score: float
    final_rank: int
    row_id: int
    context_text: str
    context_row_ids: tuple[int, ...]
    context_expanded: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["context_row_ids"] = list(self.context_row_ids)
        return data


@dataclass
class RetrievalResult:
    """Final answer-ready payload plus the intermediate candidates."""

    question: str
    candidate_k: int
    final_k: int
    neighbor_radius: int
    candidates: list[Candidate] = field(default_factory=list)
    results: list[RerankedChunk] = field(default_factory=list)

    def to_dict(self, include_candidates: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "candidate_k": self.candidate_k,
            "final_k": self.final_k,
            "neighbor_radius": self.neighbor_radius,
            "results": [item.to_dict() for item in self.results],
        }
        if include_candidates:
            payload["candidates"] = [item.to_dict() for item in self.candidates]
        return payload


class CrossEncoderReranker:
    """Deterministic CPU cross-encoder scoring ``(question, chunk_text)`` pairs."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        revision: Optional[str] = RERANKER_REVISION,
        device: str = "cpu",
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        torch_threads: int = DEFAULT_TORCH_THREADS,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise RerankerError("reranker batch_size must be at least 1")
        if torch_threads < 1:
            raise RerankerError("reranker torch_threads must be at least 1")
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.torch_threads = torch_threads
        self.seed = seed
        self._model: Any = None
        self.loaded_revision: Optional[str] = None

    def _load(self) -> Any:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RerankerError(
                "the reranker requires sentence-transformers and PyTorch; "
                "install them with: pip install sentence-transformers"
            ) from error

        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
        except Exception:  # pragma: no cover - cosmetic only
            pass

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(self.torch_threads)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older PyTorch
            torch.use_deterministic_algorithms(True)

        try:
            model = CrossEncoder(
                self.model_name, revision=self.revision, device=self.device
            )
            model.model.eval()
        except Exception as error:
            # Deliberately explicit: no embedding-similarity fallback.
            raise RerankerError(
                f"unable to load reranker model {self.model_name!r} "
                f"at revision {self.revision!r}: {error}"
            ) from error

        num_labels = getattr(model.model.config, "num_labels", 1)
        if num_labels != 1:
            raise RerankerError(
                f"reranker {self.model_name!r} reports num_labels={num_labels}; "
                "a single-logit relevance head is required"
            )
        # sentence-transformers records the resolved base-model revision on the
        # model card; transformers' ``config._commit_hash`` is not always set.
        card = getattr(model, "model_card_data", None)
        self.loaded_revision = getattr(card, "base_model_revision", None) or getattr(
            model.model.config, "_commit_hash", None
        )
        if (
            self.revision
            and self.loaded_revision
            and self.loaded_revision != self.revision
        ):
            raise RerankerError(
                f"reranker revision mismatch: requested {self.revision!r} but "
                f"loaded {self.loaded_revision!r}"
            )
        return model

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load()
        return self._model

    def score(self, question: str, texts: Sequence[str]) -> np.ndarray:
        """Return one relevance logit per ``(question, text)`` pair."""

        if not isinstance(question, str) or not question.strip():
            raise RerankerError("question must be a non-empty string")
        texts = list(texts)
        if not texts:
            raise RerankerError("at least one candidate text is required")
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise RerankerError(f"candidate text at position {index} is empty")

        pairs = [(question, text) for text in texts]
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                activation_fn=None,  # raw relevance logits; ranking is monotonic
            )
        except Exception as error:
            raise RerankerError(f"reranker inference failed: {error}") from error

        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if values.shape[0] != len(pairs):
            raise RerankerError(
                f"reranker returned {values.shape[0]} scores for {len(pairs)} pairs"
            )
        if not np.isfinite(values).all():
            raise RerankerError("reranker returned a non-finite score")
        return values


class ContextAssembler:
    """Attach bounded, deterministic neighbourhood context to final chunks.

    Rules (all pure functions of ``row_id`` and the frozen metadata):

    1. A primary chunk's context window is the contiguous row range
       ``[row_id - radius, row_id + radius]``.
    2. A neighbour is kept only if it shares the primary chunk's ``sha256``
       (same source PDF), so context never bleeds across documents.
    3. Rows are emitted in ascending ``row_id`` order, i.e. document order.
    4. Neighbours already claimed by a better-ranked context block are skipped,
       so no text is repeated.  Primaries are always included in their own block.
    5. The window is capped at ``2 * radius + 1`` rows; ``final_k`` never changes.
    """

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def _fetch(self, row_id: int) -> Optional[dict[str, Any]]:
        row = self.retriever.index.connection.execute(
            "SELECT row_id, chunk_id, sha256, text FROM chunks WHERE row_id = ?",
            (int(row_id),),
        ).fetchone()
        if row is None:
            return None
        return {
            "row_id": int(row["row_id"]),
            "chunk_id": row["chunk_id"],
            "sha256": row["sha256"],
            "text": row["text"],
        }

    def _window(self, row_id: int, sha256: str, radius: int) -> list[dict[str, Any]]:
        if radius <= 0:
            primary = self._fetch(row_id)
            return [primary] if primary else []
        low = max(0, row_id - radius)
        high = row_id + radius
        rows = self.retriever.index.connection.execute(
            "SELECT row_id, chunk_id, sha256, text FROM chunks "
            "WHERE row_id BETWEEN ? AND ? ORDER BY row_id",
            (low, high),
        ).fetchall()
        window = [
            {
                "row_id": int(row["row_id"]),
                "chunk_id": row["chunk_id"],
                "sha256": row["sha256"],
                "text": row["text"],
            }
            for row in rows
            if row["sha256"] == sha256
        ]
        if not any(item["row_id"] == row_id for item in window):
            primary = self._fetch(row_id)
            if primary:
                window.append(primary)
                window.sort(key=lambda item: item["row_id"])
        return window

    def assemble(
        self, chunks: Sequence[RerankedChunk], radius: int = DEFAULT_NEIGHBOR_RADIUS
    ) -> list[RerankedChunk]:
        if radius < 0:
            raise RetrievalError("neighbor_radius must not be negative")

        claimed: set[int] = set()
        assembled: list[RerankedChunk] = []
        for chunk in chunks:
            primary = self._fetch(chunk.row_id)
            if primary is None:
                raise RetrievalError(f"no metadata row for FAISS id {chunk.row_id}")
            window = self._window(chunk.row_id, primary["sha256"], radius)

            kept: list[dict[str, Any]] = []
            for item in window:
                if item["row_id"] == chunk.row_id:
                    kept.append(item)
                elif item["row_id"] not in claimed:
                    kept.append(item)
            kept.sort(key=lambda item: item["row_id"])
            claimed.update(item["row_id"] for item in kept)
            claimed.add(chunk.row_id)

            assembled.append(
                RerankedChunk(
                    chunk_id=chunk.chunk_id,
                    scheme_name=chunk.scheme_name,
                    department=chunk.department,
                    state=chunk.state,
                    beneficiaries=chunk.beneficiaries,
                    source_file=chunk.source_file,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    faiss_similarity=chunk.faiss_similarity,
                    reranker_score=chunk.reranker_score,
                    final_rank=chunk.final_rank,
                    row_id=chunk.row_id,
                    context_text=CONTEXT_SEPARATOR.join(item["text"] for item in kept),
                    context_row_ids=tuple(item["row_id"] for item in kept),
                    context_expanded=len(kept) > 1,
                )
            )
        return assembled


class RerankPipeline:
    """``candidate_k`` -> rerank -> ``final_k`` with assembled context."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        final_k: int = DEFAULT_FINAL_K,
        neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
        index_path: Path = DEFAULT_INDEX,
        database_path: Path = DEFAULT_DATABASE,
        manifest_path: Path = DEFAULT_MANIFEST,
        device: str = "cpu",
        torch_threads: int = DEFAULT_TORCH_THREADS,
    ) -> None:
        self.retriever = retriever or Retriever(
            index_path=index_path,
            database_path=database_path,
            manifest_path=manifest_path,
            device=device,
            top_k=candidate_k,
        )
        self.reranker = reranker or CrossEncoderReranker(
            device=device, torch_threads=torch_threads
        )
        self.candidate_k = self._validate_candidate_k(candidate_k)
        self.final_k = self._validate_final_k(final_k, self.candidate_k)
        if neighbor_radius < 0:
            raise RetrievalError("neighbor_radius must not be negative")
        self.neighbor_radius = neighbor_radius
        self.assembler = ContextAssembler(self.retriever)
        self.last_candidates: list[Candidate] = []

    def _validate_candidate_k(self, candidate_k: int) -> int:
        if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
            raise RetrievalError("candidate_k must be an integer")
        count = self.retriever.index.count
        if candidate_k < 1 or candidate_k > count:
            raise RetrievalError(
                f"candidate_k must be between 1 and {count}; got {candidate_k}"
            )
        return candidate_k

    def _validate_final_k(self, final_k: int, candidate_k: int) -> int:
        if isinstance(final_k, bool) or not isinstance(final_k, int):
            raise RetrievalError("final_k must be an integer")
        if final_k < 1:
            raise RetrievalError("final_k must be at least 1")
        if final_k > candidate_k:
            raise RetrievalError(
                f"final_k ({final_k}) must not exceed candidate_k ({candidate_k})"
            )
        return final_k

    def retrieve(
        self,
        question: str,
        candidate_k: Optional[int] = None,
        final_k: Optional[int] = None,
    ) -> RetrievalResult:
        k_candidates = (
            self.candidate_k
            if candidate_k is None
            else self._validate_candidate_k(candidate_k)
        )
        k_final = (
            self.final_k if final_k is None else self._validate_final_k(final_k, k_candidates)
        )

        hits: list[RetrievedChunk] = self.retriever.search(question, top_k=k_candidates)
        if len(hits) != k_candidates:
            raise RetrievalError(
                f"retrieved {len(hits)} candidates; expected {k_candidates}"
            )

        reranker_scores = self.reranker.score(
            question, [hit.text for hit in hits]
        )

        candidates = [
            Candidate(
                chunk_id=hit.chunk_id,
                scheme_name=hit.scheme_name,
                department=hit.department,
                state=hit.state,
                beneficiaries=hit.beneficiaries,
                source_file=hit.source_file,
                page_start=hit.page_start,
                page_end=hit.page_end,
                text=hit.text,
                faiss_similarity=hit.similarity,
                reranker_score=float(score),
                faiss_rank=hit.rank,
                row_id=hit.row_id,
            )
            for hit, score in zip(hits, reranker_scores)
        ]
        self.last_candidates = list(candidates)

        ranked = sorted(
            candidates,
            key=lambda item: (-item.reranker_score, -item.faiss_similarity, item.row_id),
        )
        selected = [
            RerankedChunk(
                chunk_id=item.chunk_id,
                scheme_name=item.scheme_name,
                department=item.department,
                state=item.state,
                beneficiaries=item.beneficiaries,
                source_file=item.source_file,
                page_start=item.page_start,
                page_end=item.page_end,
                text=item.text,
                faiss_similarity=item.faiss_similarity,
                reranker_score=item.reranker_score,
                final_rank=rank,
                row_id=item.row_id,
                context_text=item.text,
                context_row_ids=(item.row_id,),
                context_expanded=False,
            )
            for rank, item in enumerate(ranked[:k_final], start=1)
        ]
        assembled = self.assembler.assemble(selected, radius=self.neighbor_radius)

        return RetrievalResult(
            question=question,
            candidate_k=k_candidates,
            final_k=k_final,
            neighbor_radius=self.neighbor_radius,
            candidates=candidates,
            results=assembled,
        )

    def close(self) -> None:
        self.retriever.close()

    def __enter__(self) -> "RerankPipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


_DEFAULT_PIPELINE: Optional[RerankPipeline] = None


def get_pipeline(
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
    neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
) -> RerankPipeline:
    """Process-wide pipeline so the e5 and cross-encoder models load once."""

    global _DEFAULT_PIPELINE
    if (
        _DEFAULT_PIPELINE is None
        or _DEFAULT_PIPELINE.candidate_k != candidate_k
        or _DEFAULT_PIPELINE.final_k != final_k
        or _DEFAULT_PIPELINE.neighbor_radius != neighbor_radius
    ):
        if _DEFAULT_PIPELINE is not None:
            _DEFAULT_PIPELINE.close()
        _DEFAULT_PIPELINE = RerankPipeline(
            candidate_k=candidate_k,
            final_k=final_k,
            neighbor_radius=neighbor_radius,
        )
    return _DEFAULT_PIPELINE


def retrieve(
    question: str,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
    neighbor_radius: int = DEFAULT_NEIGHBOR_RADIUS,
) -> RetrievalResult:
    """Convenience entrypoint for the next (generation) stage."""

    return get_pipeline(
        candidate_k=candidate_k, final_k=final_k, neighbor_radius=neighbor_radius
    ).retrieve(question)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve candidates with e5 + FAISS, rerank with a multilingual "
            "cross-encoder, and assemble context."
        )
    )
    parser.add_argument("question", help="user question")
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=DEFAULT_NEIGHBOR_RADIUS,
        help="adjacent chunks per side to attach as context (0 disables)",
    )
    parser.add_argument("--torch-threads", type=int, default=DEFAULT_TORCH_THREADS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="list all candidate_k candidates before reranking",
    )
    parser.add_argument("--max-text-chars", type=int, default=320)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with RerankPipeline(
            candidate_k=args.candidate_k,
            final_k=args.final_k,
            neighbor_radius=args.neighbor_radius,
            index_path=args.index,
            database_path=args.database,
            manifest_path=args.manifest,
            torch_threads=args.torch_threads,
        ) as pipeline:
            result = pipeline.retrieve(args.question)
    except Exception as error:
        print(f"Context assembly/reranking failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"Question: {args.question}")
    print(
        f"Pipeline: {result.candidate_k} FAISS candidates -> reranker -> "
        f"{len(result.results)} final chunks "
        f"(neighbour radius {result.neighbor_radius})"
    )
    print(f"Reranker: {RERANKER_MODEL} @ {RERANKER_REVISION[:12]}\n")

    if args.show_candidates:
        print("Candidates before reranking (FAISS order):")
        for item in result.candidates:
            print(
                f"  faiss#{item.faiss_rank:<2} sim={item.faiss_similarity:.4f} "
                f"rerank={item.reranker_score:+.4f}  row={item.row_id:<6} "
                f"{item.scheme_name[:60]}"
            )
        print()

    for item in result.results:
        text = item.context_text
        if args.max_text_chars and len(text) > args.max_text_chars:
            text = text[: args.max_text_chars].rstrip() + "..."
        expanded = f" (+{len(item.context_row_ids) - 1} neighbours)" if item.context_expanded else ""
        print(
            f"[{item.final_rank}] rerank={item.reranker_score:+.4f}  "
            f"faiss_sim={item.faiss_similarity:.4f}{expanded}"
        )
        print(f"    scheme: {item.scheme_name}")
        print(f"    source: {item.source_file} (pages {item.page_start}-{item.page_end})")
        print(f"    chunk:  {item.chunk_id}")
        print(f"    text:   {text}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
