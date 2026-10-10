"""Local synthetic JEV service for scorer tests; never returns model predictions.

Run: python tests/fake_jev_server.py --port 8765
Only loopback addresses are accepted. Request bodies stay in memory for tests;
the server never logs image payloads, prompts, labels, or other request contents.
"""

from __future__ import annotations

import argparse
import base64
import io
import ipaddress
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import random
import threading
from typing import Any

from PIL import Image


def decode_png(data_url: str) -> Image.Image:
    """Validate the request contains an actual PNG and return its decoded pixels."""
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("Expected a PNG base64 data URL")
    data = base64.b64decode(data_url[len(prefix):], validate=True)
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Image is not a PNG")
    with Image.open(io.BytesIO(data)) as original:
        original.verify()
    with Image.open(io.BytesIO(data)) as original:
        original.load()
        return original.copy()


class FakeJevServer(ThreadingHTTPServer):
    """Thread-safe random fixture with inspectable requests and retry injection."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], seed: int, fail_first: int):
        super().__init__(address, FakeJevHandler)
        self.requests: list[dict[str, Any]] = []
        self.fail_first = fail_first
        self.lock = threading.Lock()
        self.rng = random.Random(seed)

    def probabilities(self, count: int) -> list[float]:
        with self.lock:
            values = [self.rng.uniform(0.01, 1.0) for _ in range(count)]
        total = sum(values)
        return [value / total for value in values]


class FakeJevHandler(BaseHTTPRequestHandler):
    server: FakeJevServer

    def log_message(self, format: str, *args: Any) -> None:
        # BaseHTTPRequestHandler's default logs are deliberately suppressed.
        return

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "fake_server": True})
        else:
            self.send_json(404, {"error": "Unknown fixture endpoint", "fake_server": True})

    def do_POST(self) -> None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16 * 1024 * 1024:
                raise ValueError("Request body has invalid size")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Request must be a JSON object")
            with self.server.lock:
                self.server.requests.append({"path": self.path, "body": body})
                fail = self.server.fail_first > 0
                if fail:
                    self.server.fail_first -= 1
            if fail:
                self.send_json(503, {"error": "Synthetic transient failure", "fake_server": True})
                return
            if self.path == "/v1/decide":
                self.decision(body)
            elif self.path == "/v1/chat/completions":
                self.chat(body)
            else:
                self.send_json(404, {"error": "Unknown fixture endpoint", "fake_server": True})
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.send_json(400, {"error": str(error), "fake_server": True})

    def decision(self, body: dict[str, Any]) -> None:
        if body.get("kind") != "choice" or body.get("thinking") != "off":
            raise ValueError("Decision fixture requires kind=choice and thinking=off")
        state = body.get("state")
        if (not isinstance(state, list) or len(state) != 2 or state[0] != "Chart: "
                or not isinstance(state[1], dict) or set(state[1]) != {"image"}):
            raise ValueError("Expected documented image decision state")
        decode_png(state[1]["image"])
        options = body.get("options")
        if options not in (["Up", "Flat", "Down"], ["TSLA", "NVDA"]):
            raise ValueError("Unexpected decision options")
        if not isinstance(body.get("question"), str) or not body["question"]:
            raise ValueError("Expected a question")
        probabilities = self.server.probabilities(len(options))
        self.send_json(200, {"options": options, "probabilities": probabilities,
                             "fake_server": True})

    def chat(self, body: dict[str, Any]) -> None:
        if body.get("max_tokens") != 1 or body.get("logprobs") is not True:
            raise ValueError("Chat fixture requires one-token logprobs")
        allowed = body.get("allowed_token_ids")
        if allowed not in ([32, 33, 34], [32, 33]):
            raise ValueError("Chat fixture requires the calibrated A/B/C token ids")
        if body.get("top_logprobs", 0) < len(allowed):
            raise ValueError("Insufficient top_logprobs")
        if body.get("return_tokens_as_token_ids") is not True:
            raise ValueError("Chat fixture requires stable token id logprobs")
        if (body.get("add_generation_prompt") is not False
                or body.get("add_special_tokens") is not False
                or body.get("top_k") != 0 or body.get("top_p") != 1.0
                or not isinstance(body.get("chat_template"), str)):
            raise ValueError("Chat fixture requires the raw untruncated decision template")
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise ValueError("Expected messages")
        images = [item["image_url"]["url"] for message in messages
                  for item in (message.get("content") if isinstance(message.get("content"), list) else [])
                  if item.get("type") == "image_url"]
        if len(images) != 1:
            raise ValueError("Expected one PNG image")
        decode_png(images[0])
        text = "".join(item.get("text", "") for message in messages
                       for item in message.get("content", [])
                       if item.get("type") == "text")
        if not text.startswith("[kind] choice\n[state] Chart: ") or not text.endswith("\n[decision]:"):
            raise ValueError("Expected the model-card raw decision prompt")
        probabilities = self.server.probabilities(len(allowed))
        top = [{"token": f"token_id:{token_id}", "logprob": math.log(probability),
                "bytes": [65 + position]} for position, (token_id, probability)
               in enumerate(zip(allowed, probabilities))]
        best = max(range(len(allowed)), key=probabilities.__getitem__)
        self.send_json(200, {
            "id": "synthetic-jev", "object": "chat.completion",
            "model": "autotrust/JEV-27B-VL", "fake_server": True,
            "choices": [{"index": 0, "message": {"role": "assistant",
                         "content": chr(65 + best)}, "finish_reason": "length",
                         "logprobs": {"content": [{"token": top[best]["token"],
                             "logprob": top[best]["logprob"], "bytes": [65 + best],
                             "top_logprobs": top}]}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
        })


def make_server(host: str = "127.0.0.1", port: int = 0, seed: int = 17,
                fail_first: int = 0) -> FakeJevServer:
    """Create, but do not start, a local fixture server on a free port by default."""
    if host == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("Fake server only accepts loopback IP addresses") from error
    if not address.is_loopback or address.version != 4:
        raise ValueError("Fake server only accepts IPv4 loopback addresses")
    if fail_first < 0:
        raise ValueError("fail_first must be nonnegative")
    return FakeJevServer((host, port), seed, fail_first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fail-first", type=int, default=0)
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.seed, args.fail_first)
    print(f"Synthetic random probabilities only: http://{server.server_address[0]}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
