#!/usr/bin/env python
"""Nodes 3-4 shared model: functional Cech cover at the '=' token + three anti-trivial locks.

Cover: the final residual (at '=') is projected to 126 dims and split into three disjoint 42-dim chunks
Za,Zb,Zc -- same causal prefix => same algorithmic state, different basis (no causal-mask violation).
Locks: (1) independent random task heads Wa,Wb,Wc force each chunk to solve the task in a DIFFERENT
basis -> restriction maps must be genuine dictionaries (forbids r->I). (2) COSINE consistency (scale-
invariant) forbids norm-collapse. (3) OPERATOR cocycle ||Rca Rbc Rab - I||_F^2 (on weights, not data)
forbids Z->0 and anchors scale/rank.
  arms: baseline (task only) | consistency (+a*Lcons) | cocycle (+a*Lcons + b*Lcyc)
Restriction maps present in ALL arms (param-matched); used in the loss only where the arm calls for it.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from node1_grok_llc import P, D_MODEL, N_HEADS, D_HEAD, D_MLP

CHUNK, READ = 42, 126


class TopoTF(nn.Module):
    def __init__(self, mode="baseline"):
        super().__init__(); self.mode = mode
        self.embed = nn.Embedding(P + 1, D_MODEL)
        self.pos = nn.Parameter(torch.randn(3, D_MODEL) / math.sqrt(D_MODEL))
        self.Wq, self.Wk, self.Wv = (nn.Linear(D_MODEL, N_HEADS * D_HEAD, bias=False) for _ in range(3))
        self.Wo = nn.Linear(N_HEADS * D_HEAD, D_MODEL, bias=False)
        self.Win, self.Wout = nn.Linear(D_MODEL, D_MLP, bias=False), nn.Linear(D_MLP, D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, READ, bias=False)
        self.Wa, self.Wb, self.Wc = (nn.Linear(CHUNK, P, bias=False) for _ in range(3))   # independent heads
        self.Rab, self.Rbc, self.Rca = (nn.Linear(CHUNK, CHUNK, bias=False) for _ in range(3))  # restriction maps (all arms)

    def features(self, x):
        B = x.shape[0]
        h = self.embed(x) + self.pos[None]
        q, k, v = (W(h).view(B, 3, N_HEADS, D_HEAD).transpose(1, 2) for W in (self.Wq, self.Wk, self.Wv))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D_HEAD)
        m = torch.triu(torch.ones(3, 3, device=x.device), 1).bool()
        att = att.masked_fill(m, float("-inf")).softmax(-1)
        z = (att @ v).transpose(1, 2).reshape(B, 3, -1)
        h = h + self.Wo(z); h = h + self.Wout(F.relu(self.Win(h)))
        zr = self.proj(h[:, -1])
        return zr[:, :CHUNK], zr[:, CHUNK:2 * CHUNK], zr[:, 2 * CHUNK:]

    def forward(self, x):
        Za, Zb, Zc = self.features(x)
        return (self.Wa(Za), self.Wb(Zb), self.Wc(Zc)), (Za, Zb, Zc)


def task_loss(out, y):
    la, lb, lc = out
    return F.cross_entropy(la, y) + F.cross_entropy(lb, y) + F.cross_entropy(lc, y)


def topo_losses(model, chunks):
    Za, Zb, Zc = chunks
    cd = lambda pred, tgt: (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()   # cosine distance (scale-invariant)
    Lcons = (cd(model.Rab(Za), Zb) + cd(model.Rbc(Zb), Zc) + cd(model.Rca(Zc), Za)) / 3
    prod = model.Rca.weight @ model.Rbc.weight @ model.Rab.weight
    Lcyc = ((prod - torch.eye(CHUNK, device=prod.device)) ** 2).sum()            # operator Frobenius^2
    return Lcons, Lcyc


def test_acc(model, xte, yte):
    with torch.no_grad():
        (la, lb, lc), _ = model(xte)
        return sum((l.argmax(-1) == yte).float().mean().item() for l in (la, lb, lc)) / 3
