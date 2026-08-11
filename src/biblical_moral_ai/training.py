"""Lazy-loaded QLoRA SFT and DPO launchers with reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from .hardware import inspect_cuda
from .pilot import PilotWorkflow
from .preflight import ProjectPreflight
from .registry import load_json


class TrainingBlockedError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inspect_training_request(config_path: str | Path, *, root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    path = (
        (root_path / config_path).resolve()
        if not Path(config_path).is_absolute()
        else Path(config_path)
    )
    config = load_json(path)
    stage = str(config.get("stage", "all"))
    preflight_mode = config.get("preflight_mode")
    research_mode = preflight_mode == "research_unreviewed"
    readiness = (
        PilotWorkflow(root_path).research_readiness()
        if research_mode
        else (
            PilotWorkflow(root_path).readiness()
            if preflight_mode == "pilot"
            else ProjectPreflight(root_path).training_readiness(stage=stage)
        )
    )
    research_inputs_ready, research_inputs_detail = (
        _inspect_unreviewed_research_inputs(root_path, config)
        if research_mode
        else (True, "not an unreviewed research configuration")
    )
    cuda = inspect_cuda(
        minimum_vram_gib=float(config.get("gates", {}).get("minimum_gpu_vram_gib", 24))
    )
    return {
        "config_path": str(path),
        "config_sha256": _sha256(path),
        "stage": config.get("stage"),
        "project_ready": readiness.ready and research_inputs_ready,
        "project_checks": list(readiness.to_dict()["checks"])
        + (
            [
                {
                    "name": "research_unreviewed_inputs",
                    "passed": research_inputs_ready,
                    "detail": research_inputs_detail,
                    "blocking": True,
                }
            ]
            if research_mode
            else []
        ),
        "research_only": research_mode,
        "human_review_bypassed": research_mode,
        "release_eligible": False if research_mode else None,
        "cuda": cuda.to_dict(),
    }


def run_training(
    config_path: str | Path,
    *,
    root: str | Path = ".",
    smoke_test: bool = False,
    allow_unreviewed_research: bool = False,
) -> Path:
    root_path = Path(root).resolve()
    config_file = (
        (root_path / config_path).resolve()
        if not Path(config_path).is_absolute()
        else Path(config_path)
    )
    config = load_json(config_file)
    preflight_mode = config.get("preflight_mode")
    pilot_mode = preflight_mode == "pilot"
    research_mode = preflight_mode == "research_unreviewed"
    if (pilot_mode or research_mode) and not smoke_test:
        raise TrainingBlockedError("pilot configurations may only run with --smoke-test")
    if research_mode and not allow_unreviewed_research:
        raise TrainingBlockedError(
            "unreviewed research requires --allow-unreviewed-research"
        )
    readiness = (
        PilotWorkflow(root_path).research_readiness()
        if research_mode
        else (
            PilotWorkflow(root_path).readiness()
            if pilot_mode
            else ProjectPreflight(root_path).training_readiness(
                stage=str(config.get("stage", "all"))
            )
        )
    )
    if not readiness.ready:
        failed = [check.detail for check in readiness.checks if check.blocking and not check.passed]
        raise TrainingBlockedError("project preflight failed: " + " | ".join(failed))
    if research_mode:
        research_inputs_ready, research_inputs_detail = _inspect_unreviewed_research_inputs(
            root_path, config
        )
        if not research_inputs_ready:
            raise TrainingBlockedError(research_inputs_detail)
    cuda = inspect_cuda(
        minimum_vram_gib=float(config.get("gates", {}).get("minimum_gpu_vram_gib", 24))
    )
    if not cuda.ready:
        raise TrainingBlockedError(f"CUDA preflight failed: {cuda.error or cuda.to_dict()}")

    dependencies = _load_training_dependencies()
    output_dir = root_path / config["training"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "status": "started",
        "stage": config["stage"],
        "smoke_test": smoke_test,
        "research_only": research_mode,
        "human_review_bypassed": research_mode,
        "release_eligible": False if research_mode else None,
        "warning": (
            "UNREVIEWED RESEARCH OUTPUT: not accepted data and not eligible for release"
            if research_mode
            else None
        ),
        "git_commit": _git_commit(root_path),
        "config_path": str(config_file.relative_to(root_path)),
        "config_sha256": _sha256(config_file),
        "cuda": cuda.to_dict(),
        "packages": {
            package: metadata.version(package)
            for package in (
                "torch",
                "transformers",
                "datasets",
                "peft",
                "trl",
                "accelerate",
                "bitsandbytes",
            )
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    try:
        if config["stage"] == "sft":
            _run_sft(config, root_path, dependencies, smoke_test=smoke_test)
        elif config["stage"] == "dpo":
            _run_dpo(config, root_path, dependencies, smoke_test=smoke_test)
        else:
            raise TrainingBlockedError(f"unsupported training stage: {config.get('stage')}")
    except Exception:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise

    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _inspect_unreviewed_research_inputs(
    root: Path, config: dict[str, Any]
) -> tuple[bool, str]:
    gates = config.get("gates", {})
    if gates.get("release_prohibited") is not True:
        return False, "unreviewed research config must set gates.release_prohibited=true"
    if gates.get("smoke_test_only") is not True:
        return False, "unreviewed research config must set gates.smoke_test_only=true"
    dataset = config.get("dataset", {})
    manifest_path = root / "data/pilot/research_unreviewed_manifest.json"
    if not manifest_path.is_file():
        return False, "materialize unreviewed research data first: missing research manifest"
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError) as exc:
        return False, f"invalid unreviewed research manifest: {exc}"
    if (
        manifest.get("status") != "unreviewed_research_only"
        or manifest.get("release_eligible") is not False
        or manifest.get("accepted_counts_modified") is not False
    ):
        return False, "unreviewed research manifest safety labels are invalid"
    for split in ("sft", "preferences", "evals"):
        source_path = root / "data/pilot/candidates" / f"{split}.jsonl"
        source_meta = manifest.get("source_candidates", {}).get(split, {})
        if (
            not source_path.is_file()
            or source_meta.get("path")
            != source_path.relative_to(root).as_posix()
            or source_meta.get("sha256") != _sha256(source_path)
        ):
            return False, f"research manifest is stale for {split} candidates"
    for field, split, minimum_field in (
        ("train_path", "sft", "minimum_candidate_train_records"),
        ("eval_path", "evals", "minimum_candidate_eval_records"),
    ):
        relative = Path(str(dataset.get(field, "")))
        if not relative.as_posix().startswith("data/pilot/research_unreviewed_"):
            return False, f"{field} must use a research_unreviewed pilot artifact"
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return False, f"{field} escapes the project root"
        if not path.is_file():
            return False, f"materialize unreviewed research data first: missing {relative.as_posix()}"
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"invalid unreviewed research data in {relative.as_posix()}: {exc}"
        if not rows:
            return False, f"unreviewed research data is empty: {relative.as_posix()}"
        minimum = int(dataset.get(minimum_field, 0))
        if minimum < 1 or len(rows) < minimum:
            return False, (
                f"unreviewed research data count is below {minimum_field}: "
                f"{len(rows)}/{minimum}"
            )
        if any(
            row.get("research_status") != "unreviewed_candidate"
            or row.get("release_eligible") is not False
            or row.get("human_review_bypassed") is not True
            for row in rows
        ):
            return False, f"unreviewed research labels are missing: {relative.as_posix()}"
        output_meta = manifest.get("outputs", {}).get(split, {})
        if (
            output_meta.get("path") != relative.as_posix()
            or output_meta.get("sha256") != _sha256(path)
            or output_meta.get("count") != len(rows)
        ):
            return False, f"research manifest output binding is invalid: {relative.as_posix()}"
    return True, "unreviewed research inputs are labeled and release-ineligible"


def _load_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    except ImportError as exc:
        raise TrainingBlockedError(
            "training dependencies are missing; install the 'training' extra with a CUDA-compatible PyTorch build"
        ) from exc
    return {
        "torch": torch,
        "load_dataset": load_dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DPOConfig": DPOConfig,
        "DPOTrainer": DPOTrainer,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def _quantization(config: dict[str, Any], dependencies: dict[str, Any]) -> Any:
    torch = dependencies["torch"]
    BitsAndBytesConfig = dependencies["BitsAndBytesConfig"]
    values = dict(config["quantization"])
    dtype = values.pop("bnb_4bit_compute_dtype")
    values["bnb_4bit_compute_dtype"] = getattr(torch, dtype)
    return BitsAndBytesConfig(**values)


def _run_sft(config: dict[str, Any], root: Path, dep: dict[str, Any], *, smoke_test: bool) -> None:
    model_config = config["model"]
    dataset_config = config["dataset"]
    tokenizer = dep["AutoTokenizer"].from_pretrained(
        model_config["model_id"], revision=model_config["revision"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = dep["AutoModelForCausalLM"].from_pretrained(
        model_config["model_id"],
        revision=model_config["revision"],
        quantization_config=_quantization(config, dep),
        device_map="auto",
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    model.config.use_cache = False
    model = dep["prepare_model_for_kbit_training"](model, use_gradient_checkpointing=True)
    lora = dep["LoraConfig"](**config["lora"])
    train = dep["load_dataset"](
        "json", data_files=str(root / dataset_config["train_path"]), split="train"
    )
    evaluation = dep["load_dataset"](
        "json", data_files=str(root / dataset_config["eval_path"]), split="train"
    )
    values = dict(config["training"])
    values["output_dir"] = str(root / values["output_dir"])
    if smoke_test:
        values.update(max_steps=2, save_strategy="no", eval_strategy="no")
        train = train.select(range(min(8, len(train))))
    args = dep["SFTConfig"](**values)
    trainer = dep["SFTTrainer"](
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=None if smoke_test else evaluation,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(values["output_dir"])


def _run_dpo(config: dict[str, Any], root: Path, dep: dict[str, Any], *, smoke_test: bool) -> None:
    model_config = config["model"]
    dataset_config = config["dataset"]
    tokenizer = dep["AutoTokenizer"].from_pretrained(
        model_config["base_model_id"], revision=model_config["base_revision"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = dep["AutoModelForCausalLM"].from_pretrained(
        model_config["base_model_id"],
        revision=model_config["base_revision"],
        quantization_config=_quantization(config, dep),
        device_map="auto",
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    base.config.use_cache = False
    base = dep["prepare_model_for_kbit_training"](base, use_gradient_checkpointing=True)
    model = dep["PeftModel"].from_pretrained(
        base, root / model_config["sft_adapter_path"], is_trainable=True
    )
    train = dep["load_dataset"](
        "json", data_files=str(root / dataset_config["train_path"]), split="train"
    )
    evaluation = dep["load_dataset"](
        "json", data_files=str(root / dataset_config["eval_path"]), split="train"
    )
    values = dict(config["training"])
    values["output_dir"] = str(root / values["output_dir"])
    if smoke_test:
        values.update(max_steps=2, save_strategy="no", eval_strategy="no")
        train = train.select(range(min(8, len(train))))
    args = dep["DPOConfig"](**values)
    trainer = dep["DPOTrainer"](
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train,
        eval_dataset=None if smoke_test else evaluation,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(values["output_dir"])
