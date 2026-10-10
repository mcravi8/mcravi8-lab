"""Pure request and response helpers for JEV's documented System 1 APIs.

Both backends require the JEV decision LoRA to be served. The chat backend is
model-card System 1 readout, including its decision-head bias and calibration,
rather than a probability guessed from generated text.

Sources, checked 2026-10-09:
https://huggingface.co/autotrust/JEV-27B-VL#system-1-post-v1decide
https://huggingface.co/autotrust/JEV-27B-VL/blob/main/adapter_vllm/decision_head.json
https://huggingface.co/autotrust/JEV-27B-VL/blob/main/calibration.json
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

API_PATHS = {
    "decide": "/v1/decide",
    "chat-logprobs": "/v1/chat/completions",
}
DIRECTION_OPTIONS = ("Up", "Flat", "Down")
DIRECTION_QUESTION = (
    "This is a 20-day daily stock chart. Over the next 5 trading days, "
    "will the price go Up, stay Flat, or go Down?"
)
TICKER_OPTIONS = ("TSLA", "NVDA")
TICKER_QUESTION = "Which ticker does this unlabeled stock chart show?"

# Native choice slots start at decision_head.json slot 8. Only A/B/C are
# needed here; their token IDs are independent of the option text's tokens.
CHOICE_TOKEN_IDS = (32, 33, 34)
CHOICE_BIASES = (
    -0.00017852929886430502,
    -0.00010998119250871241,
    -0.00011865551641676575,
)
CHOICE_TEMPERATURE = 1.0161280671450397
PROBABILITY_SUM_TOLERANCE = 1e-5

# This is the raw image decision template shown in the model card. It must
# be accepted by a server started with --trust-request-chat-template.
RAW_CHAT_TEMPLATE = (
    "{%- for m in messages -%}{%- for c in m['content'] -%}"
    "{%- if c['type'] == 'text' -%}{{ c['text'] }}"
    "{%- else -%}<|vision_start|><|image_pad|><|vision_end|>"
    "{%- endif -%}{%- endfor -%}{%- endfor -%}"
)


def _options(options: Sequence[str]) -> list[str]:
    if isinstance(options, (str, bytes)):
        raise ValueError("options must be a sequence of distinct strings")
    requested = list(options)
    if not 2 <= len(requested) <= len(CHOICE_TOKEN_IDS):
        raise ValueError("this scorer supports two or three choice options")
    if any(not isinstance(option, str) or not option or "\n" in option or "\r" in option
           for option in requested):
        raise ValueError("options must be nonempty strings without newlines")
    if len(set(requested)) != len(requested):
        raise ValueError("options must be distinct")
    return requested


def _api(api: str) -> None:
    if api not in API_PATHS:
        raise ValueError(f"unsupported JEV API: {api!r}")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validate_probabilities(values: Sequence[Any]) -> list[float]:
    probabilities = [_number(value, "probability") for value in values]
    if any(not 0 <= value <= 1 for value in probabilities):
        raise ValueError("probabilities must be between zero and one")
    total = math.fsum(probabilities)
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"probabilities must sum to one, got {total}")
    return [value / total for value in probabilities]


def build_payload(
    image_url: str,
    options: Sequence[str],
    question: str,
    api: str = "decide",
    model: str = "jev-decision",
) -> dict[str, Any]:
    """Build one image choice request, with no HTTP or filesystem work."""
    _api(api)
    requested = _options(options)
    if not isinstance(image_url, str) or not image_url:
        raise ValueError("image_url must be a nonempty string")
    if not isinstance(question, str) or not question:
        raise ValueError("question must be a nonempty string")
    if api == "decide":
        return {
            "kind": "choice",
            "state": ["Chart: ", {"image": image_url}],
            "question": question,
            "options": requested,
            "thinking": "off",
        }

    labels = "ABC"
    option_lines = "\n".join(f"{labels[index]}) {option}"
                             for index, option in enumerate(requested))
    content = [
        {"type": "text", "text": "[kind] choice\n[state] "},
        {"type": "text", "text": "Chart: "},
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": (
            f"\n[question] {question}\n[options]\n{option_lines}\n[decision]:"
        )},
    ]
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "chat_template": RAW_CHAT_TEMPLATE,
        "add_generation_prompt": False,
        "add_special_tokens": False,
        "max_tokens": 1,
        "temperature": 1.0,
        # Override generation_config.json truncation of processed logprobs.
        "top_k": 0,
        "top_p": 1.0,
        "logprobs": True,
        "top_logprobs": len(requested),
        "allowed_token_ids": list(CHOICE_TOKEN_IDS[:len(requested)]),
        "return_tokens_as_token_ids": True,
    }


def parse_probabilities(
    response: dict[str, Any],
    options: Sequence[str],
    api: str = "decide",
) -> list[float]:
    """Return validated probabilities in the caller's requested option order."""
    _api(api)
    requested = _options(options)
    if not isinstance(response, dict):
        raise ValueError("JEV response must be a JSON object")
    if api == "decide":
        returned = response.get("options")
        values = response.get("probabilities")
        if not isinstance(returned, list) or not isinstance(values, list):
            raise ValueError("JEV response needs options and probabilities arrays")
        if (len(returned) != len(requested) or len(values) != len(requested)
                or any(not isinstance(option, str) for option in returned)
                or len(set(returned)) != len(returned)
                or set(returned) != set(requested)):
            raise ValueError("JEV response options do not match the requested options")
        by_option = dict(zip(returned, values))
        return _validate_probabilities([by_option[option] for option in requested])

    try:
        top_logprobs = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("JEV chat response is missing option logprobs") from error
    if not isinstance(top_logprobs, list):
        raise ValueError("JEV chat top_logprobs must be an array")
    expected_ids = CHOICE_TOKEN_IDS[:len(requested)]
    logprobs: dict[int, float] = {}
    for entry in top_logprobs:
        if not isinstance(entry, dict):
            raise ValueError("JEV chat logprob entry must be an object")
        token = entry.get("token")
        if not isinstance(token, str) or not token.startswith("token_id:"):
            raise ValueError("JEV chat must return tokens as token_id:<id>")
        try:
            token_id = int(token.removeprefix("token_id:"))
        except ValueError as error:
            raise ValueError("JEV chat returned an invalid token ID") from error
        if token_id in expected_ids:
            if token_id in logprobs:
                raise ValueError(f"JEV chat duplicated option token {token_id}")
            logprobs[token_id] = _number(entry.get("logprob"), "option logprob")
    if any(token_id not in logprobs for token_id in expected_ids):
        raise ValueError("JEV chat response is missing a requested option logprob")
    logits = [
        (logprobs[token_id] + CHOICE_BIASES[index]) / CHOICE_TEMPERATURE
        for index, token_id in enumerate(expected_ids)
    ]
    largest = max(logits)
    weights = [math.exp(logit - largest) for logit in logits]
    total = math.fsum(weights)
    return _validate_probabilities([weight / total for weight in weights])
