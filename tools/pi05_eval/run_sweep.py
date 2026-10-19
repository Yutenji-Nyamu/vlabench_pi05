"""Finite official task sweep. One server and two simulation processes per card.

The outer resource owner remains responsible for the prior-work/RLT handoff.
This runner touches only its own process sessions and writes unique outputs.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from summarize import summarize


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def identity(pid):
    try:
        p = Path(f'/proc/{pid}')
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'uid': p.stat().st_uid, 'start': int(stat[19]),
                'session': int(stat[3]), 'state': stat[0]}
    except (OSError, ValueError):
        return None


def members(record):
    # A new-session child owns this session; unrelated jobs cannot join it.
    rows = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        r = identity(int(p.name))
        if r and r['uid'] == os.getuid() and r['session'] == record['pid'] and r['start'] >= record['start'] and r['state'] != 'Z':
            try:
                env = (p/'environ').read_bytes().split(b'\0')
            except OSError:
                continue
            # A long sweep may outlive PID reuse. Session numbers alone must
            # never select a later, unrelated process owned by the same user.
            if f'VLABENCH_RUN_TOKEN={record["token"]}'.encode() in env:
                rows.append(r)
    return rows


def stop(record):
    for sig, seconds in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        for row in members(record):
            current = identity(row['pid'])
            if current and current['start'] == row['start']:
                try:
                    os.kill(row['pid'], sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + seconds
        while members(record) and time.monotonic() < deadline:
            time.sleep(.5)
        if not members(record):
            return
    raise RuntimeError(f'Owned session did not clear: {record["pid"]}')


def gpu_scope(token, allowed):
    root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], timeout=15))
    rows = []
    for index, gpu in enumerate(root.findall('gpu')):
        for entry in gpu.findall('processes/process_info'):
            pid = int(entry.findtext('pid'))
            try:
                env = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            except OSError:
                continue
            if f'VLABENCH_RUN_TOKEN={token}'.encode() in env:
                rows.append({'gpu': index, 'pid': pid, 'type': entry.findtext('type')})
    assert all(r['gpu'] in allowed for r in rows), f'GPU scope violation: {rows}'
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    a = parser.parse_args()
    cfg = json.loads(a.config.read_text())
    project = Path(cfg['project'])
    source = Path(cfg['source'])
    output = Path(cfg['output']) / a.mode
    output.mkdir(parents=True, exist_ok=False)
    token = output.name + '-' + str(os.getpid())
    scripts = source / 'tools/pi05_eval'
    cancel = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: cancel.set())
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    processes = []
    guard = threading.Lock()
    servers = {}

    def launch(command, env, logpath):
        with logpath.open('x') as log:
            child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        row = identity(child.pid)
        assert row and row['uid'] == os.getuid()
        row['token'] = env['VLABENCH_RUN_TOKEN']
        with guard:
            processes.append((child, row))
            save(output / 'processes.json', [r for _, r in processes])
        return child, row

    def environment(gpu, policy=False):
        result = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                      CUDA_VISIBLE_DEVICES=cfg['gpu_uuids'][str(gpu)], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      HF_HOME=str(project/'cache/huggingface'), OPENPI_DATA_HOME=str(project/'cache/openpi'),
                      XDG_CACHE_HOME=str(project/'cache/xdg'), VLABENCH_RUN_TOKEN=token,
                      MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', EGL_DEVICE_ID='0', MUJOCO_EGL_DEVICE_ID='0',
                      __GL_APPLICATION_PROFILE='1', LD_PRELOAD=cfg['scope_libraries'][str(gpu)],
                      XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.65',
                      TOKENIZERS_PARALLELISM='false', MPLBACKEND='Agg', JAX_PLATFORMS='cuda' if policy else 'cpu')
        result.pop('PYTHONPATH', None)
        return result

    jobs = []
    tracks = sorted((source/'VLABench/configs/evaluation/tracks').glob('*.json'))
    for track in tracks:
        for task, scenes in json.loads(track.read_text()).items():
            jobs.append({'track': track.stem, 'task': task, 'count': len(scenes)})
    assert len(jobs) == 50 and sum(j['count'] for j in jobs) == 2460
    if a.mode == 'smoke':
        chosen = ['select_fruit', 'add_condiment', 'select_drink', 'select_poker']
        jobs = [dict(track='track_1_in_distribution', task=t, count=1) for t in chosen]
    lanes = [(gpu, worker) for gpu in cfg['gpus'] for worker in range(cfg.get('workers_per_gpu', 2))]
    assert len(lanes) == 4
    save(output/'plan.json', {'mode': a.mode, 'jobs': jobs, 'lanes': lanes, 'expected_episodes': sum(j['count'] for j in jobs)})
    save(output/'identity.json', identity(os.getpid()))

    def run_lane(number, gpu, worker):
        results = []
        for job in jobs[number::len(lanes)]:
            if cancel.is_set():
                break
            if servers[gpu].poll() is not None:
                cancel.set()
                raise RuntimeError(f'Policy server on GPU {gpu} exited')
            jobout = output/'jobs'/job['track']/job['task']
            jobout.mkdir(parents=True)
            command = [str(project/'envs/sim/bin/python'), '-u', str(scripts/'eval_job.py'),
                       '--openpi-root', str(project/'openpi'), '--track', job['track'], '--task', job['task'],
                       '--count', str(job['count']), '--port', str(cfg['ports'][str(gpu)]), '--output', str(jobout)]
            child, row = launch(command, environment(gpu), jobout/'driver.log')
            started = time.time()
            failure = None
            while child.poll() is None:
                if cancel.wait(5):
                    failure = 'cancelled'
                    break
                progress = jobout/'progress.json'
                updated = progress.stat().st_mtime if progress.exists() else started
                if time.time()-updated > cfg.get('no_progress_timeout_seconds', 900):
                    failure = 'no_progress_timeout'
                    break
                if servers[gpu].poll() is not None:
                    failure = 'policy_server_exited'
                    cancel.set()
                    break
            if child.poll() is None:
                stop(row)
                child.wait(timeout=10)
            if child.returncode != 0 and failure is None:
                failure = 'worker_exit_' + str(child.returncode)
            records = [json.loads(p.read_text()) for p in sorted(jobout.glob('episode-*.json'))]
            errors = list(jobout.glob('error-*.json'))
            result = {**job, 'gpu': gpu, 'worker': worker, 'exit_code': child.returncode,
                      'elapsed_seconds': time.time()-started, 'completed': len(records), 'errors': len(errors),
                      'successes': sum(bool(r['success']) for r in records),
                      'not_completed': job['count']-len(records), 'reason': failure}
            save(jobout/'job-result.json', result)
            results.append(result)
            save(output/f'lane-{number}.json', {'time': time.time(), 'results': results})
        return results

    success = False
    try:
        for gpu in cfg['gpus']:
            command = [str(project/'openpi/.venv/bin/python'), '-u', str(scripts/'serve.py'),
                       '--openpi-root', str(project/'openpi'), '--checkpoint', str(project/'checkpoints/pi05-primitive-10task'),
                       '--port', str(cfg['ports'][str(gpu)])]
            child, _ = launch(command, environment(gpu, policy=True), output/f'policy-g{gpu}.log')
            servers[gpu] = child
        deadline = time.monotonic()+1200
        while True:
            if cancel.is_set():
                raise RuntimeError('Cancelled during policy startup')
            ready = []
            for gpu, child in servers.items():
                assert child.poll() is None, f'Policy GPU {gpu} exited; inspect its log'
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{cfg["ports"][str(gpu)]}/healthz', timeout=2) as r:
                        ready.append(r.status == 200)
                except OSError:
                    ready.append(False)
            if all(ready):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Policy load timeout')
            time.sleep(5)
        save(output/'policy-ready.json', {'time': time.time(), 'gpus': cfg['gpus']})
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            futures = [pool.submit(run_lane, i, *lane) for i, lane in enumerate(lanes)]
            while not all(f.done() for f in futures):
                try:
                    save(output/'gpu-scope.json', {'time': time.time(), 'processes': gpu_scope(token, cfg['gpus'])})
                except Exception:
                    cancel.set()
                    raise
                for f in futures:
                    if f.done() and f.exception():
                        cancel.set()
                        raise f.exception()
                time.sleep(15)
            rows = [row for f in futures for row in f.result()]
        assert not cancel.is_set()
        save(output/'complete.json', {'time': time.time(), 'jobs': rows,
             'expected': sum(j['count'] for j in jobs), 'completed': sum(j['completed'] for j in rows),
             'successes': sum(j['successes'] for j in rows), 'errors': sum(j['errors'] for j in rows),
             'not_completed': sum(j['not_completed'] for j in rows)})
        summarize(output)
        success = True
    finally:
        cancel.set()
        for child, row in processes:
            stop(row)
            if child.poll() is None:
                child.wait(timeout=10)
        save(output/'released.json', {'time': time.time(), 'completed': success,
             'remaining_contexts': gpu_scope(token, cfg['gpus'])})


if __name__ == '__main__':
    main()
