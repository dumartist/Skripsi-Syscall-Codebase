"""
Corrected training/eval pipeline + canonicalization probe (Path A, steps 2-4).

Fixes the four defects flagged in review, in a self-contained harness so the
preliminary gate can be answered before touching baselines_ablation.py:

  #1  True no-defense B0 — vanilla CNN on CLEAN data only (no adversarial
      augmentation). The old "B3 vanilla" was trained with adv windows.
  #2  Paired L_robust — KL(f(clean_mal), f(adv_of_that_clean_mal)) on MATCHED
      pairs (adv[k] is the mimicry of the k-th malicious train window), not two
      unrelated slices of a shuffled batch.
  #3  Validation-split checkpoint selection — best epoch chosen on a stratified
      val split carved from train, never on the test set.
  #4  Correct no-op token set (handled in adversarial_mimicry.py / vocab).

Canonicalization probe (step 4 — "does simple preprocessing already defeat the
attack?"): train + evaluate the vanilla CNN with input canonicalization applied
to the whole pipeline (train, eval, and INSIDE the model seen by the adaptive
attacker):
  - dedup : collapse consecutive identical syscalls (attacker-agnostic).
  - strip : drop the known no-op tokens entirely (action-set-aware).

Threat eval: static mimicry recall (DA/random/PA) AND greedy white-box adaptive
evasion at controlled budgets (<=10/20/30%), the latter against the deployed
pipeline (canon applied in forward).

Output: results/corrected_preliminary.json

NOTE: single-/few-seed PRELIMINARY numbers. NOT for pasting into the thesis.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from syscall_hids.training.baselines_ablation import (
    MultiScaleEncoder, loss_cls, loss_robust, loss_contrastive,
    augment_batch, evaluate,
    LR, WEIGHT_DECAY, NUM_EPOCHS, WARMUP_EPOCHS, BATCH_SIZE, MAX_LEN,
)
from syscall_hids.diagnostics.diagnostic_shortcut_learning import load_model, THRESHOLD, DEVICE
from syscall_hids.diagnostics.diagnostic_e4_artifact_only import load_noop_tokens
from syscall_hids.attacks.adaptive_attack import attack_one

PREPARED_DIR = Path("prepared_data")
ADVERSARIAL_DIR = Path("adversarial_data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
STRATEGIES = ["distribution_aware", "random", "position_aware"]

# No-op TOKEN ids (post-encode), from the corrected vocab mapping.
NOOP_TOKENS, _ = load_noop_tokens()
NOOP_TOKENS = sorted(set(int(t) for t in NOOP_TOKENS))


# ---------------------------------------------------------------------------
# Canonicalization (applied to encoded windows; PAD=0 trails)
# ---------------------------------------------------------------------------
def _canon_seq(seq, mode):
    if mode == "dedup":
        out = []
        for t in seq:
            if not out or out[-1] != t:
                out.append(t)
        return out
    if mode == "strip":
        s = set(NOOP_TOKENS)
        return [t for t in seq if t not in s]
    return seq


def canon_batch(X, mode, max_len=MAX_LEN):
    """X: (B, L) long tensor of token ids. Returns (B, max_len) canonicalized."""
    if mode == "none":
        return X if X.size(1) == max_len else _pad(X, max_len)
    arr = X.cpu().numpy()
    out = np.zeros((arr.shape[0], max_len), dtype=np.int64)
    for i, row in enumerate(arr):
        seq = row[row != 0].tolist()
        seq = _canon_seq(seq, mode)[:max_len]
        out[i, :len(seq)] = seq
    return torch.tensor(out, dtype=torch.long)


def _pad(x, length):
    if x.size(1) >= length:
        return x[:, :length]
    pad = torch.zeros(x.size(0), length - x.size(1), dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


class CanonModel(torch.nn.Module):
    """Wrap an inference model so canonicalization is part of the pipeline the
    adaptive attacker sees. inner(x) must return logits."""
    def __init__(self, inner, mode):
        super().__init__()
        self.inner = inner
        self.mode = mode

    def forward(self, x):
        xc = canon_batch(x, self.mode).to(x.device)
        return self.inner(xc)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_all():
    with open(PREPARED_DIR / "metadata.json") as f:
        meta = json.load(f)
    tr = np.load(PREPARED_DIR / "train.npz")
    te = np.load(PREPARED_DIR / "test.npz")
    X_train = _pad(torch.tensor(tr["windows"], dtype=torch.long), MAX_LEN)
    y_train = torch.tensor(tr["labels"], dtype=torch.long)
    cat_train = torch.tensor(tr["cat_ids"], dtype=torch.long)
    X_test = _pad(torch.tensor(te["windows"], dtype=torch.long), MAX_LEN)
    y_test = torch.tensor(te["labels"], dtype=torch.long)
    cat_test = torch.tensor(te["cat_ids"], dtype=torch.long)

    # adv train/test (already encoded, 130 wide). adv_train[k] is the mimicry of
    # the k-th malicious TRAIN window (train order); adv_test similarly for test.
    adv_train, adv_test = {}, {}
    for s in STRATEGIES:
        adv_train[s] = _pad(torch.tensor(
            np.load(ADVERSARIAL_DIR / f"{s}_train.npz")["windows"], dtype=torch.long), MAX_LEN)
        d = np.load(ADVERSARIAL_DIR / f"{s}_test.npz")
        adv_test[s] = (_pad(torch.tensor(d["windows"], dtype=torch.long), MAX_LEN),
                       torch.tensor(d["cat_ids"], dtype=torch.long))

    # malicious rank per train row -> index into adv_train arrays (-1 if benign)
    mal_rank = torch.full((len(y_train),), -1, dtype=torch.long)
    mal_pos = (y_train == 1).nonzero(as_tuple=True)[0]
    mal_rank[mal_pos] = torch.arange(len(mal_pos))

    return dict(vocab=meta["vocab_size"], X_train=X_train, y_train=y_train,
                cat_train=cat_train, X_test=X_test, y_test=y_test,
                cat_test=cat_test, adv_train=adv_train, adv_test=adv_test,
                mal_rank=mal_rank, trace_train=tr["trace_ids"],
                trace_test=te["trace_ids"])


def trace_split(data, seed, val_frac=0.15):
    """Validation split at TRACE level, stratified by category: every window of a
    trace lands on the same side, so overlapping windows cannot leak between
    training and model selection. Returns (train_idx, val_idx) over windows."""
    tids = data["trace_train"]
    cats = data["cat_train"].numpy()
    uniq, first = np.unique(tids, return_index=True)
    tr_t, va_t = train_test_split(uniq, test_size=val_frac, random_state=seed,
                                  stratify=cats[first])
    va_mask = np.isin(tids, va_t)
    split_dir = RESULTS_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"val_traces_seed{seed}.json"
    if not path.exists():
        json.dump({"seed": seed, "val_frac": val_frac,
                   "val_trace_ids": sorted(int(t) for t in va_t),
                   "n_train_traces": int(len(tr_t)), "n_val_traces": int(len(va_t)),
                   "n_train_windows": int((~va_mask).sum()),
                   "n_val_windows": int(va_mask.sum())},
                  open(path, "w"), indent=2)
    return np.where(~va_mask)[0], np.where(va_mask)[0]


# ---------------------------------------------------------------------------
# Corrected training
# ---------------------------------------------------------------------------
def train_corrected(data, alpha, beta, augment_adv, canon, seed, tag,
                    epochs=NUM_EPOCHS, val_frac=0.15, sym_aug=False,
                    encoder=MultiScaleEncoder):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    print(f"\n  [{tag}] seed={seed} alpha={alpha} beta={beta} "
          f"augment_adv={augment_adv} canon={canon}")

    tr_idx, va_idx = trace_split(data, seed, val_frac)
    tr_idx = torch.tensor(tr_idx); va_idx = torch.tensor(va_idx)

    # canonicalize everything the pipeline will see
    X_tr = canon_batch(data["X_train"][tr_idx], canon)
    y_tr = data["y_train"][tr_idx]
    cat_tr = data["cat_train"][tr_idx]
    rank_tr = data["mal_rank"][tr_idx]            # into adv_train DA
    adv_full = canon_batch(data["adv_train"]["distribution_aware"], canon).to(DEVICE)

    X_va = canon_batch(data["X_train"][va_idx], canon)
    y_va = data["y_train"][va_idx]; cat_va = data["cat_train"][va_idx]
    X_te = canon_batch(data["X_test"], canon)

    # Extra classification rows (all get is_aug=1 so they are NOT used as KL
    # pairs). Two independent options, both from the CLEAN train snapshot:
    #   sym_aug  — symmetric no-op augmentation: no-op-injected copies of BOTH
    #              classes with ORIGINAL labels (teaches that no-ops aren't
    #              class-discriminative). This is the B0-sym baseline.
    #   augment_adv — true "AT": add the DA mimicry of malicious rows, label 1.
    X0, y0, cat0, rank0 = X_tr, y_tr, cat_tr, rank_tr
    extra_X, extra_y, extra_cat = [], [], []
    if sym_aug:
        from syscall_hids.diagnostics.diagnostic_e4_artifact_only import inject_noop_tokens
        rng = np.random.default_rng(seed + 7)
        Xs = np.stack([inject_noop_tokens(r, NOOP_TOKENS, 0.30, rng)
                       for r in X0.numpy()])
        extra_X.append(torch.tensor(Xs, dtype=torch.long)); extra_y.append(y0)
        extra_cat.append(cat0)
    if augment_adv:
        mal = y0 == 1
        extra_X.append(adv_full[rank0[mal]].cpu())
        extra_y.append(torch.ones(int(mal.sum()), dtype=torch.long))
        extra_cat.append(cat0[mal])
    if extra_X:
        X_tr = torch.cat([X0] + extra_X, 0)
        y_tr = torch.cat([y0] + extra_y, 0)
        cat_tr = torch.cat([cat0] + extra_cat, 0)
        rank_tr = torch.cat([rank0] + [torch.full((len(x),), -1, dtype=torch.long)
                                       for x in extra_X], 0)
        is_aug = torch.cat([torch.zeros(len(X0), dtype=torch.long)] +
                           [torch.ones(len(x), dtype=torch.long) for x in extra_X], 0)
    else:
        is_aug = torch.zeros(len(X_tr), dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr, cat_tr, is_aug, rank_tr),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va, cat_va),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, data["y_test"], data["cat_test"]),
                             batch_size=BATCH_SIZE, shuffle=False)

    model = encoder(vocab_size=data["vocab"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n_ben = int((y_tr == 0).sum()); n_att = int((y_tr == 1).sum())
    cw = torch.tensor([1.0, n_ben / max(n_att, 1)], dtype=torch.float32).to(DEVICE)
    noop_dev = torch.tensor(NOOP_TOKENS, dtype=torch.long, device=DEVICE)

    best_f1, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        wf = min(1.0, (ep + 1) / WARMUP_EPOCHS)
        a, b = alpha * wf, beta * wf
        for x, yy, _, aug, rank in train_loader:
            x, yy, aug, rank = x.to(DEVICE), yy.to(DEVICE), aug.to(DEVICE), rank.to(DEVICE)
            logits, h = model(x)
            loss = loss_cls(logits, yy, weight=cw)
            if a > 0:
                mal_clean = (aug == 0) & (yy == 1)
                if mal_clean.any():
                    adv_x = adv_full[rank[mal_clean]]
                    logits_adv, _ = model(adv_x)
                    loss = loss + a * loss_robust(logits[mal_clean], logits_adv)
            if b > 0 and x.size(0) >= 4:
                _, h_aug = model(augment_batch(x, noop_dev))
                loss = loss + b * loss_contrastive(model.project(h), model.project(h_aug))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        vf1 = evaluate(model, val_loader, DEVICE)["f1"]   # <-- VAL, not test
        if vf1 > best_f1:
            best_f1, best_ep = vf1, ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    clean = evaluate(model, test_loader, DEVICE)
    adv = {}
    for s in STRATEGIES:
        xa, ca = data["adv_test"][s]
        xa = canon_batch(xa, canon)
        adv[s] = evaluate(model, DataLoader(
            TensorDataset(xa, torch.ones(len(xa), dtype=torch.long), ca),
            batch_size=BATCH_SIZE), DEVICE)

    torch.save({"model_state": best_state, "val_f1": best_f1, "epoch": best_ep,
                "tag": tag, "seed": seed}, MODEL_DIR / f"{tag}_seed{seed}.pt")
    print(f"    val_f1={best_f1:.4f}  test: F1={clean['f1']:.4f} "
          f"recall={clean['recall']:.4f} FPR={clean['fpr']:.4f} "
          f"prec={clean['precision']:.4f} | DA_rec={adv['distribution_aware']['recall']:.4f}")
    return {"val_f1": best_f1, "clean": clean,
            "adv": {s: adv[s] for s in STRATEGIES},
            "config": dict(alpha=alpha, beta=beta, augment_adv=augment_adv,
                           canon=canon, seed=seed)}


# ---------------------------------------------------------------------------
# Adaptive evasion at controlled budgets
# ---------------------------------------------------------------------------
@torch.no_grad()
def _probs(model, X):
    out = []
    for (b,) in DataLoader(TensorDataset(X), batch_size=512):
        out.append(torch.softmax(model(b.to(DEVICE)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def _malicious_test():
    test = np.load(PREPARED_DIR / "test.npz")
    m = test["labels"] == 1
    mal_enc = test["windows"][m]
    return (mal_enc, test["trace_ids"][m],
            _pad(torch.tensor(mal_enc, dtype=torch.long), MAX_LEN))


def common_attack_set(models, seed, n_traces=200):
    """Pick the windows every model is attacked on: among malicious test windows
    detected by ALL `models` on clean input, take one window per trace (so the
    attacked samples are independent traces) and sample `n_traces` of them with a
    seed-fixed RNG. Every model is attacked on exactly the same windows."""
    _, tids, Xmc = _malicious_test()
    det_all = np.ones(len(tids), dtype=bool)
    clean_recall = {}
    for name, m in models.items():
        d = _probs(m, Xmc) >= THRESHOLD
        clean_recall[name] = float(d.mean())
        det_all &= d
    rng = np.random.RandomState(seed)
    cand = np.where(det_all)[0]
    rng.shuffle(cand)
    _, first = np.unique(tids[cand], return_index=True)   # one window per trace
    per_trace = cand[np.sort(first)]
    if len(per_trace) > n_traces:
        per_trace = rng.choice(per_trace, n_traces, replace=False)
    info = {"n_malicious_windows": int(len(tids)),
            "n_malicious_traces": int(len(np.unique(tids))),
            "n_detected_by_all_windows": int(det_all.sum()),
            "n_detected_by_all_traces": int(len(np.unique(tids[det_all]))),
            "n_attacked_traces": int(len(per_trace))}
    return np.sort(per_trace), clean_recall, info


def run_adaptive(model, sample_idx, clean_recall, caps=(0.10, 0.20, 0.30)):
    """Attack `model` on the common windows `sample_idx`. Evasion is CONDITIONAL on
    those commonly-detected windows (same population for every model). The
    unconditional post-attack recall over the full malicious test set is
    estimated as clean_recall * (1 - evasion) and labelled as an estimate."""
    mal_enc, _, _ = _malicious_test()
    seqs = [w[w != 0].tolist() for w in mal_enc[sample_idx]]
    res = [attack_one(model, s, NOOP_TOKENS) for s in seqs]
    evaded = np.array([r["evaded"] for r in res])
    ratio = np.array([r["ratio_used"] for r in res])
    out = {"recall_clean_full_test": clean_recall,
           "n_attacked": int(len(res)),
           "evasion_final": float(evaded.mean())}
    for c in caps:
        e = float((evaded & (ratio <= c + 1e-9)).mean())
        out[f"evasion_le_{int(c*100)}pct"] = e
        out[f"post_attack_recall_est_le_{int(c*100)}pct"] = clean_recall * (1.0 - e)
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--out", default="corrected_preliminary.json")
    args = ap.parse_args()

    MODEL_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
    print("=" * 72)
    print(f"Corrected pipeline | device={DEVICE} | seeds={args.seeds} "
          f"| epochs={args.epochs} | noop_tokens={NOOP_TOKENS}")
    print("=" * 72)
    data = load_all()

    # (alpha, beta, augment_adv, sym_aug, canon, tag)
    configs = [
        (0.0, 0.0, False, False, "none",  "C_B0_vanilla"),      # true no-defense
        (0.0, 0.0, False, True,  "none",  "C_B0_sym"),          # symmetric no-op aug
        (0.0, 0.0, True,  False, "none",  "C_AT_augment"),      # adv-augmentation (old B3)
        (1.0, 0.0, False, False, "none",  "C_A2_pairedKL"),     # paired robust loss
        (0.0, 0.1, False, False, "none",  "C_A3_contrastive"),  # contrastive only
        (1.0, 0.1, False, False, "none",  "C_A4_full"),         # full
        (0.0, 0.0, False, False, "dedup", "C_CANON_dedup"),     # canon: run-collapse
        (0.0, 0.0, False, False, "strip", "C_CANON_strip"),     # canon: drop no-ops
    ]

    report = {"config": {"seeds": args.seeds, "epochs": args.epochs,
                         "noop_tokens": NOOP_TOKENS, "threshold": THRESHOLD,
                         "val_split": "trace-level, stratified by category (15%)"},
              "runs": {}}
    for alpha, beta, aug, sym, canon, tag in configs:
        report["runs"][tag] = []
        for seed in args.seeds:
            r = train_corrected(data, alpha, beta, aug, canon, seed, tag,
                                epochs=args.epochs, sym_aug=sym)
            report["runs"][tag].append(r)
    # Adaptive evaluation lives in adaptive_eval.py (common sample set across models).

    out = RESULTS_DIR / args.out
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
