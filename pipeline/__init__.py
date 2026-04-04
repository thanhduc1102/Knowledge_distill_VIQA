"""
VLSP 2025 Knowledge Distillation Pipeline.
"""


def _load_config_robust(model_name_or_path: str, trust_remote_code: bool = True):
    """
    Load model config robustly. In old transformers, AutoConfig.from_pretrained
    raises ValueError for unknown model types even with trust_remote_code=True.
    Fallback: read config.json manually and import the custom config class from
    the model's local Python files (if the model was downloaded with remote code).
    """
    import json
    import importlib.util
    from pathlib import Path
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    except (KeyError, ValueError) as e:
        if "model type" not in str(e) and "model_type" not in str(e):
            raise
        print(f"  AutoConfig failed ({str(e)[:80]}...) — loading config manually.")

    model_path = Path(model_name_or_path)
    config_json_path = model_path / "config.json"
    if not config_json_path.exists():
        raise FileNotFoundError(f"config.json not found at {model_path}")

    with open(config_json_path) as f:
        config_dict = json.load(f)

    auto_map = config_dict.get("auto_map", {})

    # Try loading the custom config class from the model's local Python file
    if "AutoConfig" in auto_map:
        config_cls = _import_class_from_local(model_path, auto_map["AutoConfig"])
        if config_cls is not None:
            try:
                return config_cls.from_pretrained(model_name_or_path)
            except Exception as e2:
                print(f"  Custom config class failed: {e2}")

    # Final fallback: PretrainedConfig (no architecture info but keeps hparams)
    from transformers import PretrainedConfig
    print("  Using PretrainedConfig as last-resort config loader.")
    return PretrainedConfig.from_pretrained(model_name_or_path)


def _import_class_from_local(model_path, auto_map_entry: str):
    """
    Dynamically import a class from a model's local Python file.
    auto_map_entry format: "module_filename.ClassName" (e.g. "modeling_qwen3_5.Qwen3_5ForCausalLM")
    """
    import importlib.util
    import sys
    from pathlib import Path

    try:
        parts = auto_map_entry.rsplit(".", 1)
        if len(parts) != 2:
            return None
        module_name, class_name = parts
        module_file = Path(model_path) / f"{module_name}.py"
        if not module_file.exists():
            return None
        # Unique module name to avoid conflicts
        full_module_name = f"_remote_{module_name}"
        spec = importlib.util.spec_from_file_location(full_module_name, str(module_file))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, class_name, None)
    except Exception as e:
        print(f"  _import_class_from_local({auto_map_entry}) failed: {e}")
        return None


def _load_tokenizer_robust(model_name_or_path: str, trust_remote_code: bool = True):
    """
    Load a tokenizer robustly, handling cases where the installed transformers
    version doesn't natively support the model architecture (e.g., Qwen3.5
    requires transformers >= 4.52 for CONFIG_MAPPING recognition).

    Strategy:
    1. Try AutoTokenizer.from_pretrained normally
    2. On model_type KeyError/ValueError, try PreTrainedTokenizerFast directly
       (works for all Qwen models — tokenizer is self-contained in tokenizer.json)
    3. Fall back to AutoTokenizer with forced trust_remote_code and no config lookup
    """
    import json
    from pathlib import Path
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
    except (KeyError, ValueError) as e:
        error_msg = str(e)
        if "model type" not in error_msg and "model_type" not in error_msg:
            raise  # Not a model-type recognition issue

        print(f"  AutoTokenizer failed ({error_msg[:80]}...)")
        print("  Falling back to PreTrainedTokenizerFast (reads tokenizer.json directly)...")

    # Method 1: PreTrainedTokenizerFast — reads tokenizer.json without needing model type
    # This works for Qwen3.5 and most modern tokenizers.
    try:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        print("  Tokenizer loaded via PreTrainedTokenizerFast.")
        return tok
    except Exception as e2:
        print(f"  PreTrainedTokenizerFast also failed: {e2}")

    # Method 2: Read tokenizer_config.json and try the declared tokenizer_class directly
    model_path = Path(model_name_or_path)
    tok_config_path = model_path / "tokenizer_config.json"
    if tok_config_path.exists():
        try:
            with open(tok_config_path) as f:
                tok_cfg = json.load(f)
            tok_class_name = tok_cfg.get("tokenizer_class", "")
            # Look up known classes
            known_classes = {
                "PreTrainedTokenizerFast": "transformers.PreTrainedTokenizerFast",
                "Qwen2Tokenizer": "transformers.Qwen2Tokenizer",
                "LlamaTokenizer": "transformers.LlamaTokenizer",
                "LlamaTokenizerFast": "transformers.LlamaTokenizerFast",
            }
            if tok_class_name in known_classes:
                import importlib
                mod_name, cls_name = known_classes[tok_class_name].rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    tok = cls.from_pretrained(model_name_or_path)
                    print(f"  Tokenizer loaded via {tok_class_name}.")
                    return tok
        except Exception as e3:
            print(f"  tokenizer_config.json method failed: {e3}")

    # Method 3: Last resort — try again with explicit trust_remote_code, skip config
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=True
    )
    return tok


def _load_model_robust(model_name_or_path: str, model_kwargs: dict):
    """
    Load a causal LM model robustly, handling cases where the installed
    transformers version doesn't natively support the model architecture
    (e.g., Qwen3.5 requires transformers >= 4.52).

    Strategy:
    1. Try AutoModelForCausalLM.from_pretrained (works if transformers is new enough)
    2. If KeyError on model_type, load config with trust_remote_code=True to get
       the model's own registered classes, then use those directly.
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    import json
    from pathlib import Path

    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except (KeyError, ValueError) as e:
        error_msg = str(e)
        if "model type" not in error_msg and "model_type" not in error_msg:
            raise  # Not a model-type recognition issue, re-raise

        print(f"  AutoModelForCausalLM failed: {error_msg[:120]}...")
        print("  Falling back to model's own config/model classes via trust_remote_code...")

        # Load config — AutoConfig.from_pretrained may also fail in old transformers
        # if model_type is unknown even with trust_remote_code=True.
        # Fallback: load config.json manually and import the custom class from local files.
        config = _load_config_robust(model_name_or_path)

        # Method 1: config has auto_map → AutoModelForCausalLM registered via trust_remote_code
        if hasattr(config, 'auto_map') and 'AutoModelForCausalLM' in config.auto_map:
            kwargs_with_config = {**model_kwargs, "config": config, "trust_remote_code": True}
            return AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **kwargs_with_config
            )

        # Method 2: Read config.json auto_map and import model class manually
        model_path = Path(model_name_or_path)
        if model_path.is_dir():
            config_json = model_path / "config.json"
            if config_json.exists():
                with open(config_json) as f:
                    cfg_dict = json.load(f)
                auto_map = cfg_dict.get("auto_map", {})
                if "AutoModelForCausalLM" in auto_map:
                    # Try trust_remote_code path first
                    try:
                        kwargs_with_config = {**model_kwargs, "config": config, "trust_remote_code": True}
                        return AutoModelForCausalLM.from_pretrained(
                            model_name_or_path, **kwargs_with_config
                        )
                    except Exception:
                        pass
                    # Directly import the model class from local Python file
                    model_cls = _import_class_from_local(
                        model_path, auto_map["AutoModelForCausalLM"]
                    )
                    if model_cls is not None:
                        return model_cls.from_pretrained(model_name_or_path, **model_kwargs)

        # Method 3: Last resort — force trust_remote_code with loaded config
        kwargs_force = {**model_kwargs, "config": config, "trust_remote_code": True}
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **kwargs_force
        )