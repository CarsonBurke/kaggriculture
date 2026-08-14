"""Thread limits for the test session.

The suite exercises deliberately tiny models -- a few hundred kibibytes of
weights over batches of a handful of rows. At that size every parallel pool
in the stack is pure overhead: torch fans each op out to one thread per
physical core, allocates one inter-op thread per logical core on top, and the
native environment's rayon pool adds another full set. The result is a suite
that saturates every core on the machine to do arithmetic a single core would
finish sooner, and that collapses outright when it has to share the machine
with a training job.

Pinning every pool to one thread per process is therefore both cheaper and
faster here. Production entry points choose their own thread counts
explicitly -- ``inference.py`` and ``external_eval_worker.py`` both call
``torch.set_num_threads`` -- so none of this reaches a real run.
"""

from __future__ import annotations

import os

# rayon sizes its global pool once, the first time the native extension asks
# for it, and ignores the variable afterwards. This has to land before any
# test imports kagg_env, which is why it sits at conftest import time rather
# than in a fixture.
for _variable in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import torch  # noqa: E402  (the pools above must be sized before torch loads)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
