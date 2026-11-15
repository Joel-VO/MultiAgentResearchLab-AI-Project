"""Name normalization for concept, method, dataset, and metric nodes."""
import re
import unicodedata


def normalize_name(name: str) -> str:
    """Return a conservative lowercase key for a research term."""
    if not isinstance(name, str):
        return ""
    value = unicodedata.normalize("NFKC", name).strip().casefold()
    value = value.replace("&", " and ")
    value = re.sub(r"[_/ -]+", " ", value)
    value = re.sub(r"[^a-z0-9 .]", "", value)
    value = " ".join(value.split())
    if value in {"imagenet 1k", "imagenet 21k"}:
        value = "imagenet"
    return value


def concept_node_id(kind: str, name: str) -> str:
    """Return a typed graph-node ID such as ``dataset:truthfulqa``."""
    if kind not in {"concept", "method", "dataset", "metric"}:
        raise ValueError(f"Unsupported concept type: {kind}")
    normalized = normalize_name(name)
    if not normalized:
        raise ValueError("Concept name must not be empty")
    return f"{kind}:{normalized.replace(' ', '_')}"
