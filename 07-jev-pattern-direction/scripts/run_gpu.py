#!/usr/bin/env python3
"""Plan by default; explicit execution serves the pinned native JEV on an existing pod."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from common import EXP, EXP06, REPO, digest, verify_bundle, write_json


def runtime_helpers():
    spec = importlib.util.spec_from_file_location('exp06_runtime', EXP06/'scripts/run_gpu.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(args):
    protocol, manifest = verify_bundle()
    if manifest['n_decide_calls'] != protocol['call_budget']:
        raise ValueError('Prepared calls differ from frozen budget')
    if not args.execute:
        print(json.dumps({'rows': manifest['n'], 'dates': manifest['n_dates'],
            'decide_calls': manifest['n_decide_calls'], 'gpu_started': False,
            'note': 'Supply --execute on an already authorized, prepared GPU host to run'}, indent=2))
        return
    if not args.server_python or not args.model_dir:
        raise ValueError('Execution requires --server-python and --model-dir')
    helper = runtime_helpers()
    pins = json.loads((EXP06/'config/gpu_runtime.json').read_text())
    contract = json.loads((EXP06/'config/model_contract.json').read_text())
    config = {'runtime': protocol['runtime'], 'parity_gate': protocol['parity_gate']}
    helper.validate_execution(config, pins)
    folder = EXP/'results/session_v1'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'session.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        helper.verify_model(args.model_dir, contract)
        versions = helper.runtime_lock(args.server_python, args.scorer_python, folder, pins)
        port = protocol['runtime']['port']
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                raise ValueError('Port occupied; will not replace an existing server')
        environment = helper.server_environment(pins, args.model_dir, port, protocol['runtime']['max_model_len'])
        command = helper.server_command(args.server_python, args.model_dir, config, contract)
        helper.validate_server_flags(args.server_python, args.model_dir, command, environment)
        fingerprint = hashlib.sha256(json.dumps({'runtime': versions, 'server_command': command,
            'protocol_sha256': digest(EXP/'protocol.json')}, sort_keys=True).encode()).hexdigest()
        score_command = [str(args.scorer_python), str(EXP/'src/score.py'), '--base-url', f'http://127.0.0.1:{port}',
                         '--execution-fingerprint', fingerprint]
        result = {'status': 'starting', 'execution_fingerprint': fingerprint, 'server_command': command,
            'protocol_sha256': digest(EXP/'protocol.json'), 'preparation_sha256': digest(EXP/'results/preparation.json'),
            'runtime_lock_sha256': digest(folder/'runtime_lock.json'), 'weights_verified': True,
            'provisions_pod': False, 'prediction_calls_planned': protocol['call_budget']}
        started = time.monotonic()
        process = None
        with (folder/'server.log').open('a') as log:
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    env=environment, start_new_session=True)
                helper.wait_ready(process, port, contract)
                # The scorer repeats 20 charts per arm and aborts on a failed gate.
                subprocess.run(score_command, check=True, cwd=REPO, timeout=protocol['runtime']['scoring_timeout_seconds'])
                result['status'] = 'complete'
            except BaseException as error:
                result.update(status='failed', error=f'{type(error).__name__}: {error}')
                raise
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                result['elapsed_seconds'] = round(time.monotonic()-started, 3)
                result['billing_note'] = 'Stops its own server only; the existing pod must be stopped separately.'
                write_json(folder/'session_summary.json', result)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--server-python', type=Path)
    p.add_argument('--scorer-python', type=Path, default=Path(sys.executable))
    p.add_argument('--model-dir', type=Path)
    return p.parse_args(argv)


if __name__ == '__main__':
    execute(arguments())
