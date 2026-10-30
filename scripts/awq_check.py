"""awq_check.py : validate the AWQ unpack against the unquantized checkpoint.

The AWQ file alone cannot tell us whether the nibble order and zero-point
convention were read correctly -- a wrong order is a permutation of adjacent
output columns, which still looks statistically plausible.  The bf16 release
of the same model is the ground truth: the dequantized tile must match it up
to quantization error.

    python scripts/awq_check.py downloads/qwen3-0.6b-awq downloads/qwen3-0.6b-bf16
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_ref as qr

# AutoGPTQ / AutoAWQ pack 8 int4 values per int32 in this order
ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def unpack(packed, out_dim, order=None):
    u = np.ascontiguousarray(packed).view(np.uint32)
    nib = np.empty((packed.shape[0], out_dim), dtype=np.int16)
    for j in range(8):
        col = j if order is None else order[j]
        nib[:, col::8] = (u >> (4 * j)) & 0xF
    return nib


def dequant(ts, base, order=None, zoff=0):
    out = ts[base + ".scales"].shape[1]
    q = unpack(ts[base + ".qweight"], out, order)
    z = unpack(ts[base + ".qzeros"], out, order) + zoff
    s = ts[base + ".scales"].astype(np.float32)
    g = q.shape[0] // z.shape[0]
    zf = np.repeat(z, g, axis=0).astype(np.float32)
    sf = np.repeat(s, g, axis=0)
    return (q.astype(np.float32) - zf) * sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("awq")
    ap.add_argument("orig")
    args = ap.parse_args()

    ta = qr.load_safetensors(os.path.join(args.awq, "model.safetensors"))
    to = qr.load_safetensors(os.path.join(args.orig, "model.safetensors"))

    bases = ["model.layers.0.self_attn.q_proj",
             "model.layers.5.mlp.down_proj",
             "model.layers.27.mlp.gate_proj"]
    for base in bases:
        # torch stores [out, in]; dequant_awq returns [in, out]
        ref = to[base + ".weight"].astype(np.float32).T.copy()
        print(base, "orig", ref.shape)
        best = None
        for name, order, zoff in [("straight", None, 0),
                                  ("straight z-1", None, -1),
                                  ("order   ", ORDER, 0),
                                  ("order z-1", ORDER, -1)]:
            w = dequant(ta, base, order, zoff)
            err = np.abs(w - ref).mean() / np.abs(ref).mean()
            cor = np.corrcoef(w.ravel(), ref.ravel())[0, 1]
            print("   %-12s rel-err %.5f  corr %.6f" % (name, err, cor))
            if best is None or cor > best[1]:
                best = (name, cor)
        print("   -> best:", best[0])


if __name__ == "__main__":
    main()
