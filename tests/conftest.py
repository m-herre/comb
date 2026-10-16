import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # tests/tiny.py
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
