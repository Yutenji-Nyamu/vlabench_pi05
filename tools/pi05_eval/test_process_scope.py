"""CPU-only check: a mismatched run tag cannot select a process for cleanup."""
import os
import subprocess
import sys
import time
import uuid
from run_sweep import identity, members, stop

token = str(uuid.uuid4())
code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); time.sleep(60)'
child = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES='', VLABENCH_RUN_TOKEN=token))
record = identity(child.pid)
record['token'] = token
try:
    deadline = time.monotonic()+5
    while len(members(record)) < 2 and time.monotonic() < deadline:
        time.sleep(.05)
    assert len(members(record)) == 2
    assert members(dict(record, token='different-run')) == []
    assert child.poll() is None
    stop(record)
    child.wait(timeout=5)
    assert members(record) == []
    print('PASS: mismatched run excluded; exact owned parent and child cleared')
finally:
    stop(record)
    if child.poll() is None:
        child.wait(timeout=5)
