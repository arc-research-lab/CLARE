from typing import List, Optional
import math
import inspect
import logging
import sys
import random

def print_iters(workload, fields: List[str]=['layer','idx','is_preemptive','strategy']):
    """Print a list of AccIter object in pandas dataframe manner"""
    # Print header
    header = " | ".join(f"{f}".ljust(12) for f in fields)
    print(header)
    print("-" * len(header))
    # Print each row
    for iter in workload.iters:
        row = " | ".join(str(getattr(iter, f, "")).ljust(12) for f in fields)
        print(row)

def lcm_pair(a, b):
    return abs(a * b) // math.gcd(a, b)

def lcm(numbers):
    from functools import reduce
    return reduce(lcm_pair, numbers)

def debug_print(*args, sep=' ', end='\n'):
    # Get calling frame info
    frame = inspect.currentframe()
    outer = inspect.getouterframes(frame, 2)[1]
    caller = outer.function
    lineno = outer.lineno  # ← this gives the source line number
    # Combine arguments into text
    message = sep.join(str(arg) for arg in args)
    print(f"[line {lineno}, {caller}] {message}", end=end)

def init_logger(
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        enable: bool = True,
        fmt: str = "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S"
    ) -> logging.Logger:
        """set enable to False to disable the logger"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()  # avoid duplicated handlers if called multiple times
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        if not enable:
            # debug_print('disable logger')
            logger.handlers.clear()     # remove any handlers
            logger.disabled = True      # completely disable the logger
            logger.propagate = False
        if log_file is None:# Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        else:# File handler
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

def uunifast(n, U_total):
    """Return a List of utilizations, the sum of util is U_total, #elements is n"""
    utilizations = []
    sum_u = U_total
    for i in range(1, n):
        next_sum_u = sum_u * (random.random() ** (1 / (n - i)))
        utilizations.append(sum_u - next_sum_u)
        sum_u = next_sum_u
    utilizations.append(sum_u)
    utilizations.sort(reverse=True)
    return utilizations