"""GPU checks of the released weights (fp16 autocast, as Comb runs on CUDA by default).

    python bench/gpu_check.py

1. Edit exactness: fit on chunk 0, add chunks 1-3 one by one, forget chunk 2, and merge two shards, each against
   building the same rows in one go on the same frozen state: max |d logit| (G1 criterion: < 1e-3).
2. A pickle round trip predicts the same, and a CUDA pickle loads on a machine without a GPU; ``to`` moves only its
   own model (shards share the network) and a CUDA -> CPU -> CUDA round trip keeps fp16.
"""
import copy
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import torch
from sklearn.datasets import make_classification

from comb import CombClassifier


def check(name, value, limit):
    print(f"{'PASS' if value <= limit else 'FAIL'} {name}: {value:.3e} (limit {limit:g})", flush=True)
    return value <= limit


def main():
    ok = True
    X, y = make_classification(24000, 30, n_informative=12, n_classes=5, random_state=0)
    Xt = X[20000:]
    chunks = [(X[i:i + 5000], y[i:i + 5000]) for i in range(0, 20000, 5000)]
    def fit0():
        return CombClassifier(device="cuda").fit(*chunks[0])

    oneshot = fit0().add(np.vstack([c[0] for c in chunks[1:]]), np.concatenate([c[1] for c in chunks[1:]]))
    ref = oneshot._member_logits(Xt)
    streamed = fit0()
    for c in chunks[1:]:
        streamed.add(*c)
    ok &= check("stream (3 adds) vs one-shot: max |d logit|", float(np.abs(streamed._member_logits(Xt) - ref).max()),
                1e-3)
    m = fit0()
    s1 = m.empty_like().add(*chunks[1])
    s2 = m.empty_like().add(np.vstack([chunks[2][0], chunks[3][0]]), np.concatenate([chunks[2][1], chunks[3][1]]))
    m.merge(s1).merge(s2)
    ok &= check("merge of 2 shards vs one-shot: max |d logit|", float(np.abs(m._member_logits(Xt) - ref).max()), 1e-3)
    f = fit0().add(*chunks[1]).add(*chunks[3])
    g = fit0().add(*chunks[1]).add(*chunks[2]).add(*chunks[3]).forget(*chunks[2])
    ok &= check("forget vs never added: max |d logit|",
                float(np.abs(f._member_logits(Xt) - g._member_logits(Xt)).max()), 1e-3)
    p = pickle.loads(pickle.dumps(streamed))
    ok &= check("pickle round trip: max |d proba|", float(np.abs(p.predict_proba(Xt) - streamed.predict_proba(Xt)).max()),
                0.0)
    base = streamed.predict_proba(Xt)
    with tempfile.TemporaryDirectory() as d:  # a pickle of a CUDA model loads on a machine without a GPU
        pickle.dump(streamed, open(f"{d}/m.pkl", "wb"))
        np.save(f"{d}/X.npy", Xt[:500])
        code = (f"import pickle, numpy as np; m = pickle.load(open('{d}/m.pkl', 'rb')); assert m.device_.type == 'cpu'; "
                f"np.save('{d}/p.npy', m.predict_proba(np.load('{d}/X.npy')))")
        subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        on_cpu = np.load(f"{d}/p.npy")
    ok &= check("CUDA pickle loaded without a GPU vs to('cpu'): max |d proba|",
                float(np.abs(on_cpu - copy.deepcopy(streamed).to("cpu").predict_proba(Xt[:500])).max()), 1e-6)
    shard = streamed.empty_like().add(*chunks[0]).to("cpu")  # shares the network: must not move streamed's
    ok &= check("shard.to('cpu') leaves the original on CUDA: max |d proba|",
                float(np.abs(streamed.predict_proba(Xt) - base).max()), 0.0)
    ok &= check("shard on CPU predicts", float(not np.isfinite(shard.predict_proba(Xt[:100])).all()), 0.0)
    ok &= check("to('cpu').to('cuda') keeps fp16: max |d proba|",
                float(np.abs(streamed.to("cpu").to("cuda").predict_proba(Xt) - base).max()), 0.0)
    print("ALL PASS" if ok else "SOME FAILED", torch.cuda.get_device_name())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
