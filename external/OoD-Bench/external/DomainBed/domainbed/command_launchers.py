# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch
import sys

def local_launcher(commands):
    for cmd in commands:
        subprocess.call(cmd, shell=True)

# def local_launcher(commands):
#     """Launch commands serially on the local machine."""
#     for cmd in commands:
#         sys.stdout.flush()
#         sys.stderr.flush()
#         print(f'cmd: {cmd}')
#         proc = subprocess.Popen(cmd, shell=True, 
#                                 stdin=subprocess.PIPE,
#                                 stdout=subprocess.PIPE,
#                                 stderr=subprocess.PIPE,
#                                 )
#         print(f'-- proc id {proc.pid}')
        
#         O = b''
#         E = b''
#         last = time.time()
#         while proc.poll() is None:
#             print(f'proc poll {proc.poll()}')
#             tmp_time = time.time() - last
#             print(f'{time.time()} - {last} = {tmp_time}')
#             if tmp_time > 1:
#                 print(f'proc {proc.pid} still running...')
#                 last = time.time()
#             tmp = proc.stdout.read(1)
#             if tmp:
#                 O += tmp
#             tmp = proc.stderr.read(1)
#             if tmp:
#                 E += tmp
#             time.sleep(5)
#         outs, errs = proc.communicate(timeout=5)
#         proc.stdout = O + outs + proc.stdout.read()
#         proc.stderr = E + errs + proc.stderr.read()
#         proc.stdout.flush()
#         proc.stderr.flush()
#         proc.stdout.close()
#         proc.stderr.close()
#         print('flush finished')


#         returncode = proc.wait()
#         try:
#             print('-- communicate')
#             outs, errs = proc.communicate(timeout=5)
#             print('-- communicate after timeout')
#         except subprocess.TimeoutExpired:
#             print(f'\tException: TimeoutExpired.')
#             print(f'\n\tReturn code: {returncode}')
#             proc.kill()
#             outs, errs = proc.communicate()
#             print('outs---\n', outs)
#             print('errs---\n', errs)
#         time.sleep(1)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    sys.stdin.flush()
    sys.stdout.flush()
    sys.stdout.write('')
    sys.stderr.flush()
    sys.stderr.write('')

    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None]*n_gpus

    print(f'-- {n_gpus}, --- {procs_by_gpu}')

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True, 
                                                stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE,
                                            )
                procs_by_gpu[gpu_idx] = new_proc
                break
        new_proc.stdout.flush()
        new_proc.stderr.flush()
        new_proc.stdout.close()
        new_proc.stderr.close()
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
