"""qwen3_ref.py : real Qwen3-0.6B inference as the numeric reference for RTL.

Loads the AWQ checkpoint in downloads/qwen3-0.6b-awq, dequantizes it to fp32
once, and runs the actual Qwen3 network (RMSNorm + qk-norm, RoPE, grouped
query attention with a causal mask, SwiGLU MLP) in numpy.

This is a reference model, not a model of the hardware: it computes what the
network computes.  The RTL is compared against it with a tolerance that
absorbs the hardware's own quantization.

    python scripts/qwen3_ref.py --prompt "The capital of France is" --gen 8
    python scripts/qwen3_ref.py --dump-layer 0 --dump-out build/ref/layer0.npz

Only numpy and the tokenizer json are needed; the tokenizer is a byte-level
BPE, decoded here without transformers.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "downloads", "qwen3-0.6b-awq")


# ----------------------------------------------------------------------
# safetensors + AWQ
# ----------------------------------------------------------------------
def load_safetensors(path):
    """return {name: np.ndarray} with the raw dtypes of the file"""
    dt = {"I32": "<i4", "F16": "<f2", "BF16": None}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            beg, end = info["data_offsets"]
            f.seek(base + beg)
            raw = f.read(end - beg)
            if info["dtype"] == "BF16":
                u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
                out[name] = (u16 << 16).view(np.float32).reshape(info["shape"])
            else:
                out[name] = np.frombuffer(raw, dtype=dt[info["dtype"]]) \
                              .reshape(info["shape"]).copy()
        return out


# AutoAWQ / AutoGPTQ place the eight int4 fields of an int32 in this order:
# nibble j is output column 8*c + ORDER[j].  Verified against the bf16
# release in scripts/awq_check.py (corr 0.99 vs 0.25 for the naive order).
ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def unpack_int4(packed, out_dim):
    """AWQ gemm packing: int32 word w holds 8 out columns, interleaved."""
    in_dim, words = packed.shape
    nib = np.empty((in_dim, out_dim), dtype=np.int16)
    u = packed.view(np.uint32)
    for j in range(8):
        nib[:, ORDER[j]::8] = (u >> (4 * j)) & 0xF
    return nib


def dequant_awq(base, ts, group=128):
    """AWQ gemm -> fp32 [in, out]"""
    q = unpack_int4(ts[base + ".qweight"], ts[base + ".scales"].shape[1])
    z = unpack_int4(ts[base + ".qzeros"], ts[base + ".scales"].shape[1])
    s = ts[base + ".scales"].astype(np.float32)
    in_dim = q.shape[0]
    groups = in_dim // group
    zf = np.repeat(z, group, axis=0).astype(np.float32)
    sf = np.repeat(s, group, axis=0)
    return (q.astype(np.float32) - zf) * sf


# ----------------------------------------------------------------------
# tokenizer (byte-level BPE, no transformers)
# ----------------------------------------------------------------------
class Tok:
    def __init__(self, d):
        self.enc = json.load(open(os.path.join(d, "tokenizer.json"),
                                  encoding="utf-8"))
        self.vocab = self.enc["model"]["vocab"]
        self.ivocab = {i: s for s, i in self.vocab.items()}
        self.merges = [tuple(m) for m in self.enc["model"]["merges"]]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.special = {}
        for a in self.enc.get("added_tokens", []):
            self.special[a["content"]] = a["id"]

    @staticmethod
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("\xa1"), ord("\xac") + 1)) + \
             list(range(ord("\xae"), ord("\xff") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return dict(zip(bs, [chr(c) for c in cs]))

    def bpe(self, tok):
        word = list(tok)
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            cand = [(self.rank[p], p) for p in pairs if p in self.rank]
            if not cand:
                break
            _, (a, b) = min(cand)
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        return word

    # GPT-2 style pre-tokenization; \p{L} is spelled [^\W\d_], \p{N} as \d
    PAT = __import__("re").compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+",
        __import__("re").IGNORECASE)

    def encode(self, text):
        b2u = self.bytes_to_unicode()
        ids = []
        for piece in self.PAT.findall(text):
            key = "".join(b2u[b] for b in piece.encode("utf-8"))
            for p in self.bpe(key):
                ids.append(self.vocab[p])
        return ids

    def decode(self, ids):
        u2b = {v: k for k, v in self.bytes_to_unicode().items()}
        raw = bytes(u2b[c] for i in ids for c in self.ivocab[i])
        return raw.decode("utf-8", errors="replace")


# ----------------------------------------------------------------------
# network
# ----------------------------------------------------------------------
def rmsnorm(x, w, eps=1e-6):
    v = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(v + eps)) * w


def rope(x, cos, sin):
    """x [T, H, D]; rotate_half convention"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[..., :half], sin[..., :half]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


class Qwen3:
    def __init__(self, path):
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        ts = load_safetensors(os.path.join(path, "model.safetensors"))
        self.tok = Tok(path)
        c = self.cfg
        self.nh, self.nkv = c["num_attention_heads"], c["num_key_value_heads"]
        self.hd = c["head_dim"]
        self.eps = c["rms_norm_eps"]
        self.embed = ts["model.embed_tokens.weight"]

        self.L = []
        for i in range(c["num_hidden_layers"]):
            p = "model.layers.%d." % i
            self.L.append({
                "ln1": ts[p + "input_layernorm.weight"],
                "ln2": ts[p + "post_attention_layernorm.weight"],
                "qn": ts[p + "self_attn.q_norm.weight"],
                "kn": ts[p + "self_attn.k_norm.weight"],
                "wq": dequant_awq(p + "self_attn.q_proj", ts),
                "wk": dequant_awq(p + "self_attn.k_proj", ts),
                "wv": dequant_awq(p + "self_attn.v_proj", ts),
                "wo": dequant_awq(p + "self_attn.o_proj", ts),
                "wg": dequant_awq(p + "mlp.gate_proj", ts),
                "wu": dequant_awq(p + "mlp.up_proj", ts),
                "wd": dequant_awq(p + "mlp.down_proj", ts),
            })
        self.final = ts["model.norm.weight"]

        self.inv_freq = 1.0 / (c["rope_theta"] **
                               (np.arange(0, self.hd, 2, dtype=np.float64) / self.hd))

    def cos_sin(self, T):
        pos = np.arange(T, dtype=np.float64)
        f = np.outer(pos, self.inv_freq)          # [T, D/2]
        emb = np.concatenate([f, f], axis=-1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    def forward(self, ids, dump=None, start=0, x_start=None):
        """the whole network, or from `start` with `x_start` [T, H] as its
        input -- used to check a layer that is fed another layer's output
        without re-deriving the reference by hand."""
        c = self.cfg
        T = len(ids)
        x = self.embed[ids].astype(np.float32)              # [T, H]
        cos, sin = self.cos_sin(T)
        scale = self.hd ** -0.5
        mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), 1)
        dump_data = {}

        for li, L in enumerate(self.L):
            if li < start:
                continue
            if li == start and x_start is not None:
                x = np.asarray(x_start, dtype=np.float32)
            x_in = x
            h = rmsnorm(x, L["ln1"].astype(np.float32), self.eps)
            h1 = h
            q_out = h @ L["wq"]          # the GEMV this hardware computes
            k_out = h @ L["wk"]
            v_out = h @ L["wv"]
            q = q_out.reshape(T, self.nh, self.hd)
            k = k_out.reshape(T, self.nkv, self.hd)
            v = v_out.reshape(T, self.nkv, self.hd)
            q = rmsnorm(q, L["qn"].astype(np.float32), self.eps) * scale
            k = rmsnorm(k, L["kn"].astype(np.float32), self.eps)
            q = rope(q, cos[:, None, :], sin[:, None, :])
            k = rope(k, cos[:, None, :], sin[:, None, :])
            k = np.repeat(k, self.nh // self.nkv, axis=1)
            v = np.repeat(v, self.nh // self.nkv, axis=1)
            att = np.einsum("thd,shd->hts", q, k)
            att = att + mask
            att = att - att.max(axis=-1, keepdims=True)
            att = np.exp(att)
            att = att / att.sum(axis=-1, keepdims=True)
            o = np.einsum("hts,shd->thd", att, v).reshape(T, self.nh * self.hd)
            o_out = o @ L["wo"]
            x = x + o_out

            x_mid = x                                   # MLP block input
            h = rmsnorm(x, L["ln2"].astype(np.float32), self.eps)
            h2 = h
            g = h @ L["wg"]
            u = h @ L["wu"]
            a = (g / (1.0 + np.exp(-g))) * u
            down_out = a @ L["wd"]
            x = x + down_out

            if dump is not None and li == dump:
                # every projection in the block, with the activation row that
                # feeds it, so scripts/real_gemv.py can check them one by one
                dump_data = {"x_in": x_in, "h1": h1, "q": q_out,
                             "k": k_out, "v": v_out,
                             "o_in": o, "o": o_out,
                             "x_mid": x_mid,                     # MLP input
                             "h2": h2, "g": g, "u": u, "mlp_a": a,
                             "down": down_out, "x": x}
        hf = rmsnorm(x, self.final.astype(np.float32), self.eps)
        logits = hf @ self.embed.astype(np.float32).T
        return logits, dump_data

    def generate(self, text, n=8):
        ids = self.tok.encode(text)
        for _ in range(n):
            logits, _ = self.forward(ids)
            ids.append(int(np.argmax(logits[-1])))
        return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--dump-layer", type=int, default=-1)
    ap.add_argument("--dump-out", default="build/ref/layer.npz")
    args = ap.parse_args()

    m = Qwen3(args.model)
    ids = m.tok.encode(args.prompt)
    print("tokens:", ids)
    print("text  :", m.tok.decode(ids))

    dump = args.dump_layer if args.dump_layer >= 0 else None
    logits, d = m.forward(ids, dump=dump)
    top = np.argsort(-logits[-1])[:5]
    print("next-token top5:", [(int(t), round(float(logits[-1][t]), 3),
                                m.tok.decode([int(t)])) for t in top])

    if args.gen:
        out = m.generate(args.prompt, args.gen)
        print("greedy  :", m.tok.decode(out))

    if dump is not None:
        os.makedirs(os.path.dirname(args.dump_out) or ".", exist_ok=True)
        np.savez(args.dump_out, **d)
        print("dumped layer", dump, "->", args.dump_out)


if __name__ == "__main__":
    main()
