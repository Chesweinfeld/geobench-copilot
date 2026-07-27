"""Force the 'fork' multiprocessing start method for GeoBench-v2 runs.

Python 3.13 on macOS defaults to 'spawn', which pickles DataLoader worker
arguments — and the GeoBench-v2 loader returns a local closure
(_V2Dataset.get_dataset.<locals>.chained) that can't be pickled, so v2 runs
crash under spawn. Setting num_workers=0 avoids the crash but makes loading the
large .tortilla files pathologically slow (single-process, I/O-bound, effectively
wedged). 'fork' inherits memory instead of pickling, so workers start fine AND
data loading stays parallel/fast.

This file is imported automatically at interpreter startup whenever its directory
is on PYTHONPATH (scripts/probe_datasets.py adds it only for v2 datasets, so the
m-* runs keep the platform default). Worker processes only do CPU data loading —
MPS work stays in the main process — so fork is safe here.
"""

import multiprocessing

try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    # already set (e.g. re-import) — leave it
    pass
