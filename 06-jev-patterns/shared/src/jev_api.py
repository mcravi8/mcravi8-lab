"""Generalized Experiment 03 native JEV System 1 choice contract.

Model-card endpoint: https://huggingface.co/autotrust/JEV-27B-VL
No generated-text probability parsing or three-token chat approximation is used.
"""
from __future__ import annotations
import math
import re
from collections.abc import Sequence

MODEL = "autotrust/JEV-27B-VL"
PROTOCOL = "jev27-bare-v1"
CHOICE_TEMPERATURE = 1.0161280671450397
PARITY_ATOL = .01
PARITY_FLAG_THRESHOLD = .03
PARITY_DRIFT_STATISTIC = "mean_abs_diff"
PARITY_REQUIRE_ARGMAX = True
CHOICE_STRATEGY = "single"
NATIVE_CHOICES = 16


def parity_policy():
    """Matteo-approved v6 gate; mean includes every chart-option comparison."""
    return {'version':6,'n_charts':20,'atol':PARITY_ATOL,
            'drift_statistic':PARITY_DRIFT_STATISTIC,
            'drift_population':'all chart-option probabilities, including flagged charts',
            'flag_chart_max_abs_diff_above':PARITY_FLAG_THRESHOLD,
            'p90_method':'linear interpolation at (n - 1) * 0.9',
            'require_argmax_agreement':PARITY_REQUIRE_ARGMAX,
            'argmax_tie_break':'first option in declared order'}


def options_list(options):
    if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
        raise ValueError("Options must be a sequence of distinct strings")
    result = list(options)
    if not 2 <= len(result) <= 256:
        raise ValueError("JEV choice requires 2 to 256 options")
    if any(not isinstance(x, str) or not x.strip() or '\n' in x or '\r' in x for x in result):
        raise ValueError("Options must be nonempty single-line strings")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate options")
    return result


def probabilities(values, n):
    if not isinstance(values, (list, tuple)) or len(values) != n:
        raise ValueError("Expected one probability per option")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Invalid probability")
    total = math.fsum(values)
    if abs(total - 1) > 1e-5:
        raise ValueError("Probabilities do not sum to one")
    return [float(x / total) for x in values]


def build_payload(image_url, options, question, context_prefix="", reference_images=()):
    if not isinstance(question, str) or not question.strip() or not isinstance(context_prefix, str):
        raise ValueError("A question and optional text context are required")
    if not isinstance(image_url, str) or not image_url.startswith("data:image/png;base64,"):
        raise ValueError("Expected a base64 PNG data URL")
    text = (context_prefix.strip() + '\n\n' if context_prefix.strip() else '') + question.strip()
    if not isinstance(reference_images, (list, tuple)) or len(reference_images) > 7:
        raise ValueError("At most seven reference images plus one target are supported")
    state = []
    for reference in reference_images:
        if not isinstance(reference, dict) or set(reference) != {"image", "caption"}:
            raise ValueError("References need image and caption fields")
        if not isinstance(reference['image'], str) or not reference['image'].startswith('data:image/png;base64,') or not isinstance(reference['caption'], str):
            raise ValueError("References need a base64 PNG and text caption")
        state.extend([reference['caption'], {"image": reference['image']}])
    state.extend(["Target chart to classify: " if state else "Chart: ", {"image": image_url}])
    return {"kind": "choice", "state": state,
            "question": text, "options": options_list(options), "thinking": "off",
            "strategy": CHOICE_STRATEGY}


def server_identity(info, declared_revision="", expected_model=MODEL):
    if not isinstance(expected_model, str) or not expected_model.strip():
        raise ValueError("Pin a nonempty served model name")
    if not isinstance(info, dict) or info.get("model") != expected_model or info.get("protocol") != PROTOCOL:
        raise ValueError("Expected the documented JEV-27B-VL System 1 server")
    temperatures = info.get("temperatures")
    if not isinstance(temperatures, dict):
        raise ValueError("Missing calibration temperatures")
    temp = temperatures.get("choice")
    if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not math.isclose(temp, CHOICE_TEMPERATURE, abs_tol=1e-9, rel_tol=0):
        raise ValueError("Server calibration differs from the pinned JEV readout")
    advertised = info.get("weights_sha256") or info.get("model_revision")
    if advertised and declared_revision and advertised != declared_revision:
        raise ValueError("Server and caller model revisions differ")
    revision = advertised or declared_revision
    if not isinstance(revision, str) or re.fullmatch(r'(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})', revision) is None:
        raise ValueError("Pin a 40-character snapshot commit or 64-character weight hash with --model-revision")
    return {"model": expected_model, "canonical_model": MODEL, "protocol": PROTOCOL,
            "choice_strategy": CHOICE_STRATEGY, "model_revision": revision,
            "revision_source": "server" if advertised else "caller-declared",
            "choice_temperature": temp, "calibration_sha256": info.get("calibration_sha256"),
            "decision_head_sha256": info.get("decision_head_sha256")}


def parse_probabilities(response, options, identity):
    requested = options_list(options)
    if not isinstance(response, dict) or response.get("model") != identity["model"] or response.get("protocol") != identity["protocol"]:
        raise ValueError("Response model/protocol changed")
    # serve_decide.py emits the strategy name beyond its 16 trained choice slots.
    expected_adaptation = "native" if len(requested) <= NATIVE_CHOICES else CHOICE_STRATEGY
    if response.get("kind") != "choice" or response.get("effective_kind") != "choice" or response.get("adaptation") != expected_adaptation:
        raise ValueError(f"Expected a System 1 choice decision with adaptation={expected_adaptation}")
    thinking = response.get("thinking", {})
    if not isinstance(thinking, dict) or thinking.get("used", False) or thinking.get("mode", "off") != "off" or response.get("system", 1) != 1:
        raise ValueError("System 2 output is not allowed")
    revision = response.get("weights_sha256") or response.get("model_revision")
    if revision is not None and revision != identity["model_revision"]:
        raise ValueError("Model weights changed during scoring")
    returned = options_list(response.get("options"))
    if set(returned) != set(requested) or len(returned) != len(requested):
        raise ValueError("Returned options differ")
    values = probabilities(response.get("probabilities"), len(requested))
    mapping = dict(zip(returned, values))
    return [mapping[x] for x in requested]
