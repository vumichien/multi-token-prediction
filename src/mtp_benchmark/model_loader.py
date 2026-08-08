from __future__ import annotations

import gc
import inspect
from dataclasses import dataclass


class RuntimeLoadError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict[str, object]] | None = None,
        support: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts or []
        self.support = support or {}


@dataclass(slots=True)
class LoadedRuntime:
    torch: object
    tokenizer: object
    model: object
    assistant_model: object
    metadata: dict[str, object]


def default_assistant_model(target_model: str) -> str:
    return target_model if target_model.endswith("-assistant") else f"{target_model}-assistant"


def policy_attempts(policy: str) -> list[str]:
    mapping = {
        "auto": ["8bit", "4bit", "cpu_offload"],
        "8bit": ["8bit"],
        "4bit": ["4bit"],
        "cpu_offload": ["cpu_offload"],
        "none": ["none"],
    }
    if policy not in mapping:
        raise ValueError(f"Unsupported --policy '{policy}'. Expected one of {sorted(mapping)}.")
    return mapping[policy]


def execute_policy_attempts(
    attempts: list[str],
    load_attempt: object,
) -> tuple[object, list[dict[str, object]]]:
    history: list[dict[str, object]] = []
    for name in attempts:
        try:
            result = load_attempt(name)
        except Exception as exc:
            history.append({"policy": name, "status": "failed", "error": str(exc)})
            continue
        history.append({"policy": name, "status": "loaded"})
        return result, history
    summary = "; ".join(f"{item['policy']}: {item['error']}" for item in history)
    raise RuntimeLoadError(f"Unable to load any requested policy. {summary}", attempts=history)


def probe_transformers_support(transformers_module: object) -> dict[str, object]:
    generate_parameters = inspect.signature(
        transformers_module.generation.utils.GenerationMixin.generate
    ).parameters
    config_names = transformers_module.models.auto.configuration_auto.CONFIG_MAPPING_NAMES
    return {
        "assistant_model_arg": "assistant_model" in generate_parameters,
        "recognized_configs": [name for name in ("gemma4", "gemma4_assistant") if name in config_names],
    }


def load_runtime(
    target_model: str,
    assistant_model: str,
    policy: str,
) -> LoadedRuntime:
    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise RuntimeLoadError(
            f"Missing runtime dependency '{exc.name}'. Install the pinned packages in pyproject.toml."
        ) from exc
    support = probe_transformers_support(transformers)
    if not support["assistant_model_arg"]:
        raise RuntimeLoadError(
            "Installed Transformers does not expose generate(..., assistant_model=...).",
            support=support,
        )
    if len(support["recognized_configs"]) < 2:
        raise RuntimeLoadError(
            "Installed Transformers does not recognize both gemma4 and gemma4_assistant configs.",
            support=support,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(target_model)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        runtime, attempts = execute_policy_attempts(
            policy_attempts(policy),
            lambda name: _load_attempt(name, torch, transformers, target_model, assistant_model),
        )
    except RuntimeLoadError as exc:
        exc.support = support
        raise
    return LoadedRuntime(
        torch=torch,
        tokenizer=tokenizer,
        model=runtime["model"],
        assistant_model=runtime["assistant_model"],
        metadata={
            "selected_policy": runtime["policy"],
            "attempts": attempts,
            "transformers_support": support,
            "target_device": _device_summary(runtime["model"]),
            "assistant_device": _device_summary(runtime["assistant_model"]),
        },
    )


def unload_runtime(runtime: LoadedRuntime | None) -> None:
    if runtime is None:
        return
    torch = runtime.torch
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_attempt(
    name: str,
    torch: object,
    transformers: object,
    target_model: str,
    assistant_model: str,
) -> dict[str, object]:
    dtype = getattr(torch, "float16")
    target_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "dtype": dtype,
    }
    assistant_kwargs = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "dtype": dtype,
        "attn_implementation": "sdpa",
    }
    target_kwargs["attn_implementation"] = "sdpa"
    if name == "8bit":
        target_kwargs |= {
            "quantization_config": transformers.BitsAndBytesConfig(load_in_8bit=True),
            "device_map": "auto",
        }
        assistant_kwargs["device_map"] = "auto"
    elif name == "4bit":
        target_kwargs |= {
            "quantization_config": transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            ),
            "device_map": "auto",
        }
        assistant_kwargs["device_map"] = "auto"
    elif name == "cpu_offload":
        memory = _max_memory(torch)
        target_kwargs |= {"device_map": "auto", "max_memory": memory}
        assistant_kwargs |= {"device_map": "auto", "max_memory": memory}
    elif name == "none":
        target_kwargs["device_map"] = "auto"
        assistant_kwargs["device_map"] = "auto"
    model = transformers.AutoModelForCausalLM.from_pretrained(target_model, **target_kwargs)
    assistant = transformers.AutoModelForCausalLM.from_pretrained(assistant_model, **assistant_kwargs)
    return {"policy": name, "model": model, "assistant_model": assistant}


def _device_summary(model: object) -> dict[str, object]:
    device = None
    if hasattr(model, "parameters"):
        try:
            device = str(next(model.parameters()).device)
        except StopIteration:  # pragma: no cover - defensive only
            device = None
    return {
        "primary_device": device,
        "hf_device_map": getattr(model, "hf_device_map", None),
        "revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "dtype": str(getattr(model, "dtype", None)),
    }


def _max_memory(torch: object) -> dict[object, str]:
    if not torch.cuda.is_available():
        return {"cpu": "64GiB"}
    total_gib = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
    safe_gib = max(1, min(total_gib - 1, 9))
    return {0: f"{safe_gib}GiB", "cpu": "64GiB"}
