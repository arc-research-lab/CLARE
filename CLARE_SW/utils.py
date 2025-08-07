from typing import List
import math

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