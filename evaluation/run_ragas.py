#!/usr/bin/env python3
"""Run RAGAS evaluation for Medical-Project outputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("ragas_eval")


@dataclass
class EvalCase:
    question: str
    contexts: List[str]
    answer: Optional[str] = None
    ground_truth: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GenerationTimeout(Exception):
    """Raised when model generation exceeds the configured timeout."""


def infer_task_type_from_text(question: str) -> str:
    """Infer task type for text-only eval generation."""
    q = question.lower()
    if any(k in q for k in ["ldl", "hdl", "triglyceride", "cholesterol", "lipid"]):
        return "lipid_profile"
    if any(k in q for k in ["mammogram", "birads", "breast", "calcification"]):
        return "breast_imaging"
    if any(k in q for k in ["biopsy", "histology", "pathology", "carcinoma", "grade"]):
        return "biopsy_report"
    return "ct_coronary"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_local_env(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a local .env file into process env."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {idx} in {path}: {exc}") from exc
    return rows


def normalize_contexts(raw_contexts: Any) -> List[str]:
    if raw_contexts is None:
        return []
    if isinstance(raw_contexts, str):
        return [raw_contexts]
    if isinstance(raw_contexts, list):
        return [str(c) for c in raw_contexts if str(c).strip()]
    return [str(raw_contexts)]


def validate_rows(rows: Iterable[Dict[str, Any]]) -> List[EvalCase]:
    cases: List[EvalCase] = []
    for i, row in enumerate(rows, start=1):
        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError(f"Row {i} missing required field: question")

        contexts = normalize_contexts(row.get("contexts"))
        answer = row.get("answer")
        ground_truth = row.get("ground_truth")
        metadata = row.get("metadata") or {}

        if answer is not None:
            answer = str(answer).strip()
        if ground_truth is not None:
            ground_truth = str(ground_truth).strip()

        cases.append(
            EvalCase(
                question=question,
                contexts=contexts,
                answer=answer,
                ground_truth=ground_truth,
                metadata=metadata,
            )
        )
    return cases


def chunk_text(text: str, chunk_size: int = 900) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size
    return chunks


def keyword_score(query: str, candidate: str) -> int:
    tokens = {t for t in re.findall(r"[a-zA-Z]{3,}", query.lower())}
    if not tokens:
        return 0
    candidate_lower = candidate.lower()
    return sum(1 for t in tokens if t in candidate_lower)


def retrieve_contexts(question: str, kb_dir: Path, top_k: int = 3) -> List[str]:
    if not kb_dir.exists():
        return []

    candidates: List[str] = []
    for file_path in sorted(kb_dir.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunk_text(content):
            candidates.append(chunk)

    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda c: keyword_score(question, c), reverse=True)
    top = [c for c in ranked[:top_k] if c.strip()]
    return top


def state_to_answer(state: Any) -> str:
    assessment = getattr(state, "structured_assessment", None) or {}
    ordered_keys = ["clinical_summary", "primary_diagnosis", "treatment_plan", "follow_up"]
    parts: List[str] = []
    for key in ordered_keys:
        value = str(assessment.get(key, "")).strip()
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts).strip()


def _run_with_timeout(timeout_sec: int, fn, *args, **kwargs):
    """Run a callable with SIGALRM timeout on Unix platforms."""
    if timeout_sec <= 0:
        return fn(*args, **kwargs)

    if os.name != "posix":
        return fn(*args, **kwargs)

    def _handler(signum, frame):
        raise GenerationTimeout(f"Timed out after {timeout_sec}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def generate_answers(
    cases: List[EvalCase],
    kb_dir: Path,
    generation_timeout_sec: int = 0,
    eval_max_new_tokens: int = 128,
) -> None:
    from app.graph import MedicalGraph
    from app.input_preprocessor import preprocess_user_input

    logger.info("Generating answers for %d cases using MedicalGraph", len(cases))
    graph = MedicalGraph(preload_model=True, diagnosis_max_new_tokens=eval_max_new_tokens)

    try:
        for idx, case in enumerate(cases, start=1):
            if case.answer:
                logger.info("Case %d: answer already present, skipping generation", idx)
                continue

            if not case.contexts:
                case.contexts = retrieve_contexts(case.question, kb_dir)

            logger.info("Case %d: running model inference", idx)
            input_data = preprocess_user_input(text=case.question, metadata=case.metadata)
            inferred_task = infer_task_type_from_text(case.question)
            started = time.perf_counter()
            try:
                state = _run_with_timeout(
                    generation_timeout_sec,
                    graph.run,
                    task_type=inferred_task,
                    input_data=input_data,
                    cleanup_after=False,
                )
                case.answer = state_to_answer(state)
                if not case.answer:
                    case.answer = "No answer generated"
                logger.info(
                    "Case %d: inference finished in %.2fs",
                    idx,
                    time.perf_counter() - started,
                )
            except GenerationTimeout as exc:
                logger.error("Case %d: %s", idx, exc)
                case.answer = (
                    f"Generation timeout after {generation_timeout_sec}s. "
                    "No model answer generated."
                )
    finally:
        graph._unload_model()


def run_ragas(cases: List[EvalCase]) -> Dict[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS dependencies are missing. Install with: pip install -r requirements.txt"
        ) from exc

    rows: List[Dict[str, Any]] = []
    include_gt = any(c.ground_truth for c in cases)

    for case in cases:
        row: Dict[str, Any] = {
            "question": case.question,
            "answer": case.answer or "",
            "contexts": case.contexts,
        }
        if include_gt:
            row["ground_truth"] = case.ground_truth or ""
        rows.append(row)

    dataset = Dataset.from_list(rows)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    if include_gt:
        metrics.append(answer_correctness)

    logger.info("Running RAGAS with metrics: %s", [m.name for m in metrics])
    result = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False)

    frame = result.to_pandas()
    aggregate = frame.mean(numeric_only=True).to_dict()
    per_case = frame.to_dict(orient="records")

    return {
        "aggregate": aggregate,
        "per_case": per_case,
        "num_cases": len(cases),
        "metrics": list(aggregate.keys()),
    }


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "evaluation" / "results" / f"ragas_report_{stamp}.json"


def save_report(path: Path, payload: Dict[str, Any]) -> None:
    def _json_default(obj: Any):
        # Handle numpy and pandas scalar/array values emitted by metric outputs.
        try:
            import numpy as np  # type: ignore

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
        except Exception:
            pass

        # Datetime-like and fallback string conversion.
        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except Exception:
                pass
        return str(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def save_cases_jsonl(path: Path, cases: List[EvalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            row: Dict[str, Any] = {
                "question": case.question,
                "contexts": case.contexts,
                "answer": case.answer or "",
            }
            if case.ground_truth:
                row["ground_truth"] = case.ground_truth
            if case.metadata:
                row["metadata"] = case.metadata
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on medical QA cases")
    parser.add_argument("--dataset", default="evaluation/eval_cases.jsonl", help="Path to eval dataset JSONL")
    parser.add_argument("--output", default="", help="Output JSON path for report")
    parser.add_argument("--kb-dir", default="data/knowledge_base", help="Knowledge base dir for auto-context retrieval")
    parser.add_argument("--top-k", type=int, default=3, help="Top K context chunks to retrieve when missing")
    parser.add_argument("--generate-answers", action="store_true", help="Generate answers with MedicalGraph before scoring")
    parser.add_argument("--generate-only", action="store_true", help="Generate answers and exit without RAGAS scoring")
    parser.add_argument("--score-only", action="store_true", help="Run RAGAS only; do not generate answers")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and exit")
    parser.add_argument("--max-cases", type=int, default=0, help="Only evaluate first N cases (0 = all)")
    parser.add_argument(
        "--generated-output",
        default="",
        help="Optional JSONL path to save generated answers before scoring",
    )
    parser.add_argument(
        "--generation-timeout-sec",
        type=int,
        default=600,
        help="Per-case timeout for answer generation in seconds (0 disables timeout)",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens used while generating eval answers",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    load_local_env(ROOT / ".env")
    args = parse_args()

    dataset_path = (ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return 1

    rows = read_jsonl(dataset_path)
    cases = validate_rows(rows)
    if args.max_cases > 0:
        cases = cases[:args.max_cases]

    if not cases:
        logger.error("No evaluation cases found in %s", dataset_path)
        return 1

    kb_dir = (ROOT / args.kb_dir).resolve() if not Path(args.kb_dir).is_absolute() else Path(args.kb_dir)
    for case in cases:
        if not case.contexts:
            case.contexts = retrieve_contexts(case.question, kb_dir, top_k=args.top_k)

    missing_answers = sum(1 for c in cases if not c.answer)
    logger.info("Loaded %d cases (missing answers: %d)", len(cases), missing_answers)

    if args.dry_run:
        logger.info("Dry run successful. Dataset is valid.")
        return 0

    if args.generate_only and args.score_only:
        logger.error("Choose only one of --generate-only or --score-only.")
        return 1

    do_generate = args.generate_answers or args.generate_only
    do_score = not args.generate_only
    if args.score_only:
        do_generate = False
        do_score = True

    if do_generate:
        generate_answers(
            cases,
            kb_dir,
            generation_timeout_sec=args.generation_timeout_sec,
            eval_max_new_tokens=args.eval_max_new_tokens,
        )
        generated_default = default_output_path()
        generated_out = (
            Path(args.generated_output).resolve()
            if args.generated_output
            else generated_default.with_name(
                generated_default.name.replace("ragas_report_", "generated_answers_").replace(".json", ".jsonl")
            )
        )
        save_cases_jsonl(generated_out, cases)
        logger.info("Generated answers saved to: %s", generated_out)
        if args.generate_only:
            logger.info("Generate-only mode complete.")
            return 0

    if not do_generate and missing_answers:
        logger.error(
            "%d case(s) are missing answers. Use --generate-answers/--generate-only or add 'answer' in dataset.",
            missing_answers,
        )
        return 1

    if do_score and not os.getenv("OPENAI_API_KEY"):
        logger.error("Missing OPENAI_API_KEY. RAGAS evaluator needs this in current setup.")
        logger.error("Export it and rerun: export OPENAI_API_KEY='your_key'")
        return 1

    if not do_score:
        logger.info("No scoring requested. Exiting.")
        return 0

    try:
        report = run_ragas(cases)
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        logger.error(
            "If this is an evaluator LLM issue, set your provider key (for example OPENAI_API_KEY) and retry."
        )
        return 1

    output_path = Path(args.output).resolve() if args.output else default_output_path()
    report_payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "output": str(output_path),
        "aggregate": report["aggregate"],
        "num_cases": report["num_cases"],
        "metrics": report["metrics"],
        "per_case": report["per_case"],
    }

    save_report(output_path, report_payload)

    logger.info("Evaluation complete")
    logger.info("Aggregate scores: %s", report_payload["aggregate"])
    logger.info("Report saved to: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
