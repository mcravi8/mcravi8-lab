#!/usr/bin/env python3
"""One entry point for a future, explicitly authorized existing-GPU session.

Default is a local plan only. --execute starts the Experiment 01 native server,
checks both option schemas for repeatability, then scores frozen jobs. This file
has not been run against a GPU. It never provisions or manages a Runpod pod.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

EXP=Path(__file__).resolve().parents[1];REPO=EXP.parent
sys.path.insert(0,str(EXP))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def checked_path(relative):
    root=REPO.resolve()
    path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Frozen artifact path escapes repository')
    return path


def validate_assets(config):
    for relative,expected in config['assets'].items():
        path=checked_path(relative)
        if not path.is_file() or sha(path)!=expected:raise ValueError(f'Frozen artifact missing or changed: {relative}')


def selected_jobs(config,include_references=False):
    jobs=[j for j in config['jobs'] if include_references or not j['optional_reference']]
    if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate inference job IDs')
    if any(j['n_rows']<20 for j in jobs):raise ValueError('Every job needs 20 distinct chart rows for its repeatability gate')
    return jobs


def score_command(job,python,config,contract,*,parity_only=False,global_gate=False):
    runtime=config['runtime'];root=checked_path(job['root'])
    out=f"results/inference/frozen_v1/global_parity_{job['kind']}.csv" if global_gate else job['out']
    command=[str(python),str(EXP/'shared/src/score.py'),'--root',str(root),'--index',job['index'],
             '--out',out,'--options-file',str(checked_path(job['options_file'])),
             '--question',job['question'],'--control',job['control'],
             '--base-url',f"http://127.0.0.1:{runtime['port']}",
             '--served-model-name',contract['model'],'--model-revision',contract['revision'],
             '--concurrency',str(runtime['concurrency']),'--batch-size',str(runtime['batch_size']),
             '--timeout',str(runtime['timeout']),'--retries',str(runtime['retries'])]
    if job['context_file']:command+=['--context-file',str(checked_path(job['context_file']))]
    if job['reference_image']:command+=['--reference-image',str(checked_path(job['reference_image']))]
    if parity_only:command.append('--parity-only')
    return command


def global_gate_jobs(config):
    names={'candles_zero_shot_p1','synthetic_variable_zero_shot_p1'}
    jobs=[j for j in config['jobs'] if j['id'] in names]
    if len(jobs)!=2:raise ValueError('Both 8-option candle and 7-option pattern global gates must exist')
    return jobs


def execute_jobs(config,contract,python,include_references=False,runner=subprocess.run):
    """Testable ordering: all global gates pass before any prediction job starts."""
    jobs=selected_jobs(config,include_references)
    for job in global_gate_jobs(config):
        runner(score_command(job,python,config,contract,parity_only=True,global_gate=True),check=True)
    for number,job in enumerate(jobs,1):
        print(f"Scoring {number}/{len(jobs)} {job['id']} ({job['n_rows']} rows)",flush=True)
        runner(score_command(job,python,config,contract),check=True)


def verify_model(model_dir,contract):
    """Verify all published safetensor hashes, plus pinned native serving assets."""
    model_dir=Path(model_dir)
    for name,info in contract['weight_files'].items():
        path=model_dir/name
        if not path.is_file() or path.stat().st_size!=info['size'] or sha(path)!=info['sha256']:
            raise ValueError(f'Model weight differs from frozen snapshot: {name}')
    for name,expected in contract['files'].items():
        path=model_dir/name
        if not path.is_file() or sha(path)!=expected:raise ValueError(f'Cached model asset differs: {name}')
    # Non-weight model/config/tokenizer assets are also pinned by Git blob IDs.
    for name,info in contract.get('configuration_files',{}).items():
        path=model_dir/name
        if not path.is_file():raise ValueError(f'Cached model configuration missing: {name}')
        data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=info['git_blob_sha1']:raise ValueError(f'Model configuration differs: {name}')


def runtime_lock(server_python,scorer_python,folder):
    code="import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},sort_keys=True))"
    server=json.loads(subprocess.check_output([str(server_python),'-c',code],text=True))
    scorer=json.loads(subprocess.check_output([str(scorer_python),'-c',code],text=True))
    if not any(k.lower()=='vllm' for k in server):raise ValueError('--server-python must point to an installed compatible vLLM environment')
    versions={'server':server,'scorer':scorer}
    path=folder/'runtime_lock.json'
    if path.exists() and json.loads(path.read_text())!=versions:raise ValueError('Installed runtime changed since this frozen run started')
    path.write_text(json.dumps(versions,indent=2)+'\n')
    return versions


def wait_ready(process,port,contract,timeout=900):
    from shared.src.jev_api import server_identity
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('Native JEV server exited; inspect server.log')
        try:
            with urlopen(f'http://127.0.0.1:{port}/v1/decide/info',timeout=5) as response:info=json.load(response)
            server_identity(info,contract['revision'],contract['model']);return
        except (URLError,TimeoutError,ConnectionError,OSError):time.sleep(2)
    raise TimeoutError('Native server readiness timed out')


def run_session(config,contract,args):
    validate_assets(config);selected_jobs(config,args.include_references)
    folder=EXP/'shared/results/inference_run';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'session.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_model(args.model_dir,contract)
        server_python=Path(args.server_python).absolute()
        if server_python.name!='python3':
            raise ValueError('--server-python must name the prepared environment bin/python3 used by native serve.sh')
        runtime_lock(server_python,args.scorer_python,folder)
        port=config['runtime']['port']
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0:raise ValueError('Port 8000 is occupied; this script will not replace an existing server')
        environment=os.environ.copy();environment.update(MODEL_DIR=str(Path(args.model_dir).resolve()),PORT=str(port),
            MAX_MODEL_LEN=str(config['runtime']['max_model_len']),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        environment['PATH']=str(Path(args.server_python).absolute().parent)+os.pathsep+environment['PATH']
        started=time.time();process=None;result={'status':'starting','model_revision':contract['revision'],
            'config_sha256':sha(args.config),'weights_verified':True,'include_references':args.include_references,
            'runtime_lock_sha256':sha(folder/'runtime_lock.json'),'provisions_pod':False}
        with (folder/'server.log').open('a') as log:
            try:
                process=subprocess.Popen(['bash',str(REPO/contract['serving_entrypoint'])],stdout=log,stderr=subprocess.STDOUT,
                                         env=environment,start_new_session=True)
                wait_ready(process,port,contract)
                execute_jobs(config,contract,args.scorer_python,args.include_references)
                result['status']='complete'
            except BaseException as error:
                result.update(status='failed',error=type(error).__name__+': '+str(error));raise
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                result['elapsed_seconds']=round(time.time()-started,3)
                result['pod_billing_note']='Only this script\'s server process is stopped; an existing cloud pod must be stopped separately.'
                (folder/'session_summary.json').write_text(json.dumps(result,indent=2)+'\n')


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=EXP/'config/inference_arms.json')
    p.add_argument('--execute',action='store_true',help='Start native JEV on an already authorized GPU host and score')
    p.add_argument('--include-references',action='store_true')
    p.add_argument('--model-dir',type=Path,default=Path('/workspace/models/JEV-27B-VL'))
    p.add_argument('--server-python',type=Path)
    p.add_argument('--scorer-python',type=Path,default=EXP/'.venv/bin/python')
    args=p.parse_args(argv)
    if args.execute and args.server_python is None:p.error('--execute requires an explicit --server-python from the prepared vLLM environment')
    return args


def main(argv=None):
    args=arguments(argv);config=json.loads(args.config.read_text());contract=json.loads(checked_path(config['model_contract']).read_text())
    if not args.execute:
        validate_assets(config)
        print(json.dumps({'execution':'plan_only','budget':config['call_budget']['with_references' if args.include_references else 'default'],
                          'jobs':[j['id'] for j in selected_jobs(config,args.include_references)]},indent=2));return
    run_session(config,contract,args)


if __name__=='__main__':main()
