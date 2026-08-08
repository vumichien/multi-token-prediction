from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .environment import collect_environment_metadata
from .metrics import (
    extract_speculative_metrics,
    measure_call,
    quality_gate_errors,
    summarize_paired_speedups,
    summarize_records,
)
from .model_loader import (
    LoadedRuntime,
    RuntimeLoadError,
    default_assistant_model,
    load_runtime,
    unload_runtime,
)
from .prompts import load_prompts
from .results import build_raw_payload, write_results


@dataclass(slots=True)
class BenchmarkConfig:
    target_model: str
    assistant_model: str | None
    prompts_path: Path
    output_dir: Path
    max_new_tokens: int
    warmup_runs: int
    timed_runs: int
    policy: str

    def metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload["prompts_path"] = str(self.prompts_path)
        payload["output_dir"] = str(self.output_dir)
        payload["assistant_model"] = self.assistant_model or default_assistant_model(self.target_model)
        return payload


def run_benchmark(config: BenchmarkConfig, *, dry_run: bool) -> dict[str, str]:
    prompts = load_prompts(config.prompts_path)
    assistant_model = config.assistant_model or default_assistant_model(config.target_model)
    environment = collect_environment_metadata(config.target_model, assistant_model, config.prompts_path)
    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    runtime = None
    try:
        if dry_run:
            payload = build_raw_payload(config.metadata(), environment, prompts, records, summaries, dry_run=True)
            return write_results(config.output_dir, payload, summaries)
        runtime = load_runtime(config.target_model, assistant_model, config.policy)
        environment["loader"] = runtime.metadata
        records = _run_trials(runtime, prompts, config)
        summaries = _summary_rows(records)
        record_errors = sorted(
            {str(record["error"]) for record in records if record.get("error")}
            | set(quality_gate_errors(records))
        )
        payload = build_raw_payload(
            config.metadata(),
            environment,
            prompts,
            records,
            summaries,
            dry_run=False,
            errors=record_errors,
        )
        return write_results(config.output_dir, payload, summaries)
    except RuntimeLoadError as exc:
        environment["loader"] = {
            "selected_policy": None,
            "attempts": exc.attempts,
            "transformers_support": exc.support or None,
            "target_device": None,
            "assistant_device": None,
        }
        payload = build_raw_payload(
            config.metadata(),
            environment,
            prompts,
            records,
            summaries,
            dry_run=False,
            errors=[str(exc)],
        )
        write_results(config.output_dir, payload, summaries)
        raise
    except Exception as exc:
        payload = build_raw_payload(
            config.metadata(),
            environment,
            prompts,
            records,
            summaries,
            dry_run=dry_run,
            errors=[str(exc)],
        )
        write_results(config.output_dir, payload, summaries)
        raise
    finally:
        unload_runtime(runtime)


def _run_trials(
    runtime: LoadedRuntime,
    prompts: list[dict[str, str]],
    config: BenchmarkConfig,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for prompt in prompts:
        records.extend(_run_prompt(runtime, prompt, "warmup", config.warmup_runs, config, alternating=False))
        records.extend(_run_prompt(runtime, prompt, "timed", config.timed_runs, config, alternating=True))
    return records


def _run_prompt(
    runtime: LoadedRuntime,
    prompt: dict[str, str],
    stage: str,
    runs: int,
    config: BenchmarkConfig,
    *,
    alternating: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for pair_index in range(runs):
        order = ("baseline", "assistant")
        if alternating and pair_index % 2:
            order = ("assistant", "baseline")
        pair = [_run_mode(runtime, prompt, stage, pair_index, mode, config) for mode in order]
        tokens = {record["mode"]: record.get("generated_token_ids") for record in pair if not record.get("error")}
        match = tokens.get("baseline") == tokens.get("assistant") if len(tokens) == 2 else None
        note = None if match is not None else "Comparison unavailable because one arm failed."
        for record in pair:
            record["output_match"] = match
            record["comparison_note"] = note
            record.pop("generated_token_ids", None)
        records.extend(pair)
    return records


def _run_mode(
    runtime: LoadedRuntime,
    prompt: dict[str, str],
    stage: str,
    pair_index: int,
    mode: str,
    config: BenchmarkConfig,
) -> dict[str, object]:
    prompt_text = prompt["text"]
    if hasattr(runtime.tokenizer, "apply_chat_template"):
        prompt_text = runtime.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    encoded = runtime.tokenizer(prompt_text, return_tensors="pt")
    device = next(runtime.model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generate_kwargs = {
        **encoded,
        "max_new_tokens": config.max_new_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "pad_token_id": runtime.tokenizer.pad_token_id,
    }
    if mode == "assistant":
        generate_kwargs["assistant_model"] = runtime.assistant_model
    output, measured, error = measure_call(lambda: runtime.model.generate(**generate_kwargs), runtime.torch)
    record: dict[str, object] = {
        "stage": stage,
        "pair_index": pair_index,
        "mode": mode,
        "prompt_id": prompt["id"],
        "prompt_category": prompt["category"],
        "max_new_tokens": config.max_new_tokens,
        "do_sample": False,
        **measured,
    }
    if error is not None:
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    sequences = output.sequences if hasattr(output, "sequences") else output
    generated_ids = sequences[0][encoded["input_ids"].shape[-1] :].tolist()
    generated_text = runtime.tokenizer.decode(generated_ids, skip_special_tokens=True)
    speculative = extract_speculative_metrics(output)
    generated_tokens = len(generated_ids)
    record.update(
        {
            "generated_tokens": generated_tokens,
            "tokens_per_second": round(generated_tokens / measured["latency_seconds"], 6)
            if measured["latency_seconds"]
            else None,
            "generated_text": generated_text,
            "generated_token_ids": generated_ids,
            **speculative,
        }
    )
    return record


def _summary_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    scoped = [
        {"scope": "prompt", **row}
        for row in summarize_records(records, ("mode", "prompt_id", "prompt_category"))
    ]
    overall = [{"scope": "overall", **row} for row in summarize_records(records, ("mode",))]
    return scoped + overall + summarize_paired_speedups(records)
