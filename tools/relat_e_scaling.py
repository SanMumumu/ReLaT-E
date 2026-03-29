from __future__ import annotations


MODEL_SCALE_PRESETS = {
    "tiny": {"hidden_size": 384, "depth": 6, "num_heads": 6},
    "small": {"hidden_size": 512, "depth": 8, "num_heads": 8},
    "base": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "large": {"hidden_size": 1024, "depth": 16, "num_heads": 16},
}


def apply_model_scale(cfg, scale=None):
    resolved_scale = scale or getattr(cfg.generator, "scale", "base")
    if resolved_scale == "custom":
        cfg.generator.scale = "custom"
        return "custom"
    if resolved_scale not in MODEL_SCALE_PRESETS:
        raise ValueError(f"Unknown model scale: {resolved_scale}")
    for key, value in MODEL_SCALE_PRESETS[resolved_scale].items():
        setattr(cfg.generator.mot, key, value)
    cfg.generator.scale = resolved_scale
    return resolved_scale


def infer_model_scale_from_state_dict(state_dict):
    hidden_size = int(state_dict["pos_embed"].shape[-1])
    if any(key.startswith("rgb_stream.blocks.") for key in state_dict):
        depth = len(
            {
                key.split(".")[2]
                for key in state_dict
                if key.startswith("rgb_stream.blocks.") and len(key.split(".")) > 2 and key.split(".")[2].isdigit()
            }
        )
    else:
        depth = len({key.split(".")[1] for key in state_dict if key.startswith("blocks.") and key.split(".")[1].isdigit()})
    for scale, preset in MODEL_SCALE_PRESETS.items():
        if preset["hidden_size"] == hidden_size and preset["depth"] == depth:
            return scale
    return "custom"
