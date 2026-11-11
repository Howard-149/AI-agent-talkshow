"""Process exit for the GPU entry points (run.py, judge.py): the exit code is the outcome, and nothing lingers.

A vLLM engine runs in child processes and background threads. If the main code raises after the engine started (or
before it is shut down), the interpreter can wait on those forever and the job holds its GPUs until the time limit.
``main_exit`` runs a main function, prints any traceback, stops child processes and leaves with os._exit:
0 on success, the SystemExit code, or 1 on any other exception.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
from typing import Callable


def stop_children(timeout: float = 5.0) -> None:
    """Terminate (then kill) every child process this process started (e.g. a vLLM engine core)."""
    kids = multiprocessing.active_children()
    for p in kids:
        p.terminate()
    for p in kids:
        p.join(timeout)
        if p.is_alive():
            p.kill()


def exit_code(fn: Callable[[], object]) -> int:
    """Run fn and return the process exit code it stands for (prints the traceback of an unexpected exception)."""
    try:
        fn()
        return 0
    except SystemExit as e:
        if e.code is None:
            return 0
        if isinstance(e.code, int):
            return e.code
        print(e.code, file=sys.stderr)
        return 1
    except BaseException:  # noqa: BLE001  (KeyboardInterrupt too: the job must end)
        traceback.print_exc()
        return 1


def main_exit(fn: Callable[[], object]) -> None:
    code = exit_code(fn)
    try:
        stop_children()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
