"""Shrink JSON-schema tool definitions for prompts. Use in training, eval AND serving."""
import copy

_DROP = {"title", "additionalProperties", "$defs"}


def _walk(node, defs):
    if isinstance(node, list):
        return [_walk(x, defs) for x in node]
    if not isinstance(node, dict):
        return node

    # Inline $ref -> the referenced definition (keeps sibling keys like description)
    if "$ref" in node:
        target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
        target.update({k: v for k, v in node.items() if k != "$ref"})
        return _walk(target, defs)

    # Optional[X] -> X (optionality is already expressed by not being in `required`)
    if "anyOf" in node:
        non_null = [b for b in node["anyOf"] if b.get("type") != "null"]
        if len(non_null) == 1:
            merged = {k: v for k, v in node.items() if k != "anyOf"}
            merged.update(non_null[0])
            return _walk(merged, defs)

    out = {}
    for k, v in node.items():
        if k in _DROP:
            continue
        if k == "default" and v is None:
            continue
        if k == "properties":  # keys here are arg NAMES, never drop them
            out[k] = {name: _walk(sub, defs) for name, sub in v.items()}
        else:
            out[k] = _walk(v, defs)
    return out


def compact_tools(tools):
    """Take an OpenAI-shaped tool list; return a compacted deep copy."""
    result = []
    for t in tools:
        t = copy.deepcopy(t)
        params = t["function"]["parameters"]
        defs = params.get("$defs", {})
        t["function"]["parameters"] = _walk(params, defs)
        result.append(t)
    return result