"""Command-line interface for validation, policy review, and release gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .arithmetic import calculate, verify_equation
from .citation import CitationVerifier
from .evidence_store import EvidenceStore
from .hardware import inspect_cuda
from .inference import BiblicalMoralAgent, LocalChatBackend
from .pipeline import InferenceReviewPipeline
from .policy import CommandmentPolicyEngine
from .preflight import ProjectPreflight
from .registry import load_commandment_rules, load_json
from .release import ReleaseGateEvaluator, ReleaseMetrics
from .schemas import MoralAnswer
from .training import TrainingBlockedError, inspect_training_request, run_training


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _pipeline(root: Path, corpus_path: Path) -> InferenceReviewPipeline:
    corpus_payload = load_json(corpus_path)
    corpora = corpus_payload.get("sources", corpus_payload)
    return InferenceReviewPipeline(
        commandment_policy=CommandmentPolicyEngine(
            load_commandment_rules(root / "configs/commandments.json")
        ),
        citation_verifier=CitationVerifier(corpora),
        organizational_source_ids=corpus_payload.get("organizational_source_ids", []),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biblical-moral-ai")
    parser.add_argument("--root", default=".", help="project root")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate repository structure and registries")
    commands.add_parser("preflight", help="check whether full training is allowed")
    cuda = commands.add_parser("cuda-check", help="inspect local CUDA readiness")
    cuda.add_argument("--minimum-vram-gib", type=float, default=24.0)
    arithmetic = commands.add_parser("calculate", help="run safe decimal arithmetic")
    arithmetic.add_argument("expression")
    arithmetic.add_argument("--unit", default="")
    arithmetic.add_argument("--equation", action="store_true")
    review = commands.add_parser("review-answer", help="verify a structured answer JSON")
    review.add_argument("answer")
    review.add_argument("--corpus", required=True)
    release = commands.add_parser("check-release", help="evaluate release metrics")
    release.add_argument("metrics")
    ingest = commands.add_parser(
        "ingest-corpus", help="ingest one approved, digest-matched corpus artifact"
    )
    ingest.add_argument("artifact")
    ingest.add_argument("--database", default="data/index/biblical_evidence.sqlite3")
    ingest.add_argument("--registry", default="configs/data/source_registry.json")
    ingest.add_argument("--canon", default="configs/canon.json")
    search = commands.add_parser("search-evidence", help="search approved biblical evidence")
    search.add_argument("query")
    search.add_argument("--database", default="data/index/biblical_evidence.sqlite3")
    search.add_argument("--limit", type=int, default=12)
    answer = commands.add_parser(
        "answer", help="run retrieval-first inference against a local endpoint"
    )
    answer.add_argument("request")
    answer.add_argument("--database", default="data/index/biblical_evidence.sqlite3")
    answer.add_argument("--config", default="configs/inference/local_vllm.json")
    training = commands.add_parser("train", help="inspect or execute a configured training stage")
    training.add_argument("config")
    training.add_argument("--execute", action="store_true", help="execute after all gates pass")
    training.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()

    if args.command == "validate":
        report = ProjectPreflight(root).validate_structure()
        _json(report.to_dict())
        return 0 if report.ready else 1
    if args.command == "preflight":
        report = ProjectPreflight(root).training_readiness()
        _json(report.to_dict())
        return 0 if report.ready else 2
    if args.command == "cuda-check":
        report = inspect_cuda(minimum_vram_gib=args.minimum_vram_gib)
        _json(report.to_dict())
        return 0 if report.ready else 2
    if args.command == "calculate":
        if args.equation:
            _json({"equation": args.expression, "valid": verify_equation(args.expression)})
        else:
            result = calculate(args.expression, args.unit)
            _json(
                {"expression": result.expression, "value": str(result.value), "unit": result.unit}
            )
        return 0
    if args.command == "review-answer":
        answer = MoralAnswer.from_dict(load_json(args.answer))
        report = _pipeline(root, Path(args.corpus)).review(answer)
        _json(report.to_dict())
        return 0 if report.passed else 2
    if args.command == "check-release":
        metrics = ReleaseMetrics.from_dict(load_json(args.metrics))
        result = ReleaseGateEvaluator().evaluate(metrics)
        _json(result.to_dict())
        return 0 if result.approved else 2
    if args.command == "ingest-corpus":
        database = root / args.database
        database.parent.mkdir(parents=True, exist_ok=True)
        with EvidenceStore(database) as store:
            store.import_artifact(root / args.artifact, root / args.registry, root / args.canon)
        _json({"status": "ingested", "artifact": args.artifact, "database": str(database)})
        return 0
    if args.command == "search-evidence":
        with EvidenceStore(root / args.database) as store:
            passages = store.search(args.query, limit=args.limit)
        _json(
            [
                {
                    "source_id": item.source_id,
                    "reference": item.reference,
                    "language": item.language,
                    "text": item.text,
                    "context": item.context,
                }
                for item in passages
            ]
        )
        return 0
    if args.command == "answer":
        config = load_json(root / args.config)
        backend = LocalChatBackend(
            endpoint=config["endpoint"],
            model=config["model"],
            allow_remote=bool(config.get("allow_remote", False)),
        )
        with EvidenceStore(root / args.database) as store:
            result = BiblicalMoralAgent(
                root=root,
                store=store,
                backend=backend,
                retrieval_limit=int(config.get("retrieval_limit", 12)),
                graph_expansion_limit=int(config.get("graph_expansion_limit", 8)),
                max_corrections=int(config.get("max_corrections", 1)),
            ).answer(args.request)
        payload = {
            "decision": result.decision.value,
            "attempts": result.attempts,
            "retrieved": [f"{item.source_id} {item.reference}" for item in result.retrieved],
            "issues": [
                {"code": item.code, "message": item.message} for item in result.report.issues
            ],
            "answer": result.delivery_text,
        }
        _json(payload)
        return 0 if result.decision.value == "release" else 2
    if args.command == "train":
        if not args.execute:
            _json(inspect_training_request(args.config, root=root))
            return 0
        try:
            manifest = run_training(args.config, root=root, smoke_test=args.smoke_test)
        except TrainingBlockedError as exc:
            _json({"status": "blocked", "reason": str(exc)})
            return 2
        _json({"status": "completed", "manifest": str(manifest)})
        return 0
    return 1
