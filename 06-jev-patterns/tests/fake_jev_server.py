"""Deterministic loopback JEV fixture with optional drift/failure injection; no model."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.src.jev_api import MODEL, PROTOCOL, CHOICE_TEMPERATURE


class FakeServer(ThreadingHTTPServer):
    def __init__(self, address=('127.0.0.1',0), drift=False, fail_first=0, reverse=False, model=MODEL):
        super().__init__(address, Handler)
        self.counts, self.requests = Counter(), []
        self.lock = threading.Lock()
        self.drift, self.fail_first, self.reverse = drift, fail_first, reverse
        self.revision = 'a'*40
        self.model = model


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def answer(self, status, value):
        raw = json.dumps(value).encode()
        self.send_response(status); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        if self.path != '/v1/decide/info':
            return self.answer(404,{})
        self.answer(200,{'model':self.server.model,'protocol':PROTOCOL,'temperatures':{'choice':CHOICE_TEMPERATURE},
                        'model_revision':self.server.revision,'synthetic':True})

    def do_POST(self):
        if self.path != '/v1/decide':
            return self.answer(404,{})
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with self.server.lock:
            self.server.requests.append(body)
            if self.server.fail_first:
                self.server.fail_first -= 1
                return self.answer(503,{'error':'fixture retry'})
            key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
            self.server.counts[key]+=1
            count=self.server.counts[key]
        rng=random.Random(key)
        options=body['options']
        values=[rng.random()+.1 for _ in options]; total=sum(values); values=[x/total for x in values]
        if self.server.drift and count>=2:
            values=[1.]+[0.]*(len(options)-1)
        if self.server.reverse:
            options,values=options[::-1],values[::-1]
        self.answer(200,{'model':self.server.model,'protocol':PROTOCOL,'model_revision':self.server.revision,
                        'kind':'choice','effective_kind':'choice','adaptation':'native' if len(options)<=16 else 'single',
                        'options':options,'probabilities':values})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765)
    p.add_argument('--drift',action='store_true');args=p.parse_args()
    server=FakeServer(('127.0.0.1',args.port),drift=args.drift)
    print(f'Synthetic server on http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
