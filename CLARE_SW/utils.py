from typing import List

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