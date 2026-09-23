"""Kaggriculture competition research package."""

import os

# Expandable segments let the CUDA caching allocator grow segments in place
# instead of fragmenting fixed-size blocks, which the update's varying run
# shapes otherwise strand. The allocator reads its config when CUDA
# initializes, not when torch is imported, so this takes effect for every
# entry point that imports the package before touching the GPU. An explicit
# setting wins; PYTORCH_CUDA_ALLOC_CONF is torch's deprecated spelling of the
# same knob.
if "PYTORCH_ALLOC_CONF" not in os.environ and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

__all__ = ["__version__"]

__version__ = "0.1.0"
