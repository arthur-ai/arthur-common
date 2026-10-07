from functools import lru_cache

from litellm import cost_per_token

from arthur_common.models.llm_model_providers import ModelProvider

# Route prefixes are the provider names plus a "/" (e.g. "bedrock/"); region prefixes
# are not providers, so they stay a literal list
ROUTE_PREFIXES = tuple(f"{provider.value}/" for provider in ModelProvider)
REGION_PREFIXES = ("us.", "eu.", "apac.", "us-gov.")


def model_name_candidates(model_name: str) -> list[str]:
    """Return pricing candidates for a model name, most specific first.

    Yields the name as-is, then variants with a leading route prefix
    (e.g. ``bedrock/``, ``vertex_ai/``) and/or a leading region prefix
    (e.g. ``us.`` in ``us.anthropic.claude-...``) stripped. These prefixes
    describe where a model is served, not which model it is, so the stripped
    forms resolve to the same rate.
    """
    candidates = [model_name]
    for prefix in ROUTE_PREFIXES:
        if model_name.lower().startswith(prefix):
            candidates.append(model_name[len(prefix) :])
            break
    base = candidates[-1]
    for prefix in REGION_PREFIXES:
        if base.lower().startswith(prefix):
            candidates.append(base[len(prefix) :])
            break

    return [c for i, c in enumerate(candidates) if c and c not in candidates[:i]]


@lru_cache(maxsize=None)
def per_token_rates(model_name: str | None) -> tuple[float, float] | None:
    """USD (input, output) price of a single token for the model, or None if the model
    is missing or litellm can't price it.

    Base per-token rates are used rather than pricing token totals directly, since
    totals summed across many inferences would wrongly hit long-context price tiers.
    """
    # NULL model names come out of DuckDB as NaN
    if not isinstance(model_name, str) or not model_name:
        return None
    for candidate in model_name_candidates(model_name):
        try:
            input_rate, output_rate = cost_per_token(
                model=candidate,
                prompt_tokens=1,
                completion_tokens=1,
            )
            return float(input_rate), float(output_rate)
        except Exception:
            continue
    return None
