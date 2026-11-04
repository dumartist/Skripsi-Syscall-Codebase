"""
Stage 2: Adversarial Mimicry Generation

Generates adversarial syscall windows from the malicious windows produced by
Stage 1.  Three strategies are implemented:

  1. DistributionAwareMimicryAttack (primary)
     Greedily injects no-op syscalls at positions where the resulting trigrams
     best match the benign reference distribution.  Each candidate (position,
     no-op) is scored by the sum of benign-reference probabilities for the
     new trigrams it creates — a fast O(1) proxy for JSD improvement.

  2. RandomMimicryAttack (ablation baseline)
     Inserts no-ops at uniformly random positions with random no-op selection.

  3. PositionAwareMimicryAttack (experimental)
     Prioritises insertion positions by local trigram salience (rare in
     benign → high-value target), choosing no-ops that maximise benign-trigram
     overlap.

All strategies enforce:
  PhiF  — Functional preservation (original syscalls stay in order).
  PhiO  — Operational feasibility (only valid no-op syscalls injected).
  PhiM  — Minimality (<= MAX_INJECTION_RATIO injection budget).
"""

import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PREPARED_DIR = Path("prepared_data")
OUTPUT_DIR = Path("adversarial_data")
MAX_INJECTION_RATIO = 0.30
SEED = 42
TRIGRAM_N = 3

# RAW i386 syscall numbers. getuid/geteuid on i386 glibc are the *32 variants
# (199/201); the 16-bit 24/49 never appear in ADFA-LD, so vocab.get("24",1) used
# to encode them as <UNK> (token 1) — a ΦO violation (not a valid no-op). Fixed
# to the numbers that actually occur in the corpus and map to real tokens.
# getpid=20, getuid32=199, geteuid32=201, getppid=64, time=13, gettimeofday=78.
NOOP_SYSCALLS = [20, 199, 201, 64, 13, 78]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def build_bigram_ref(traces):
    counts = Counter()
    for t in traces:
        for i in range(len(t) - 1):
            counts[(t[i], t[i + 1])] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def compute_jsd(window, ref_keys, ref_vec):
    """Vectorised JSD between a window's trigram distribution and reference."""
    counts = Counter()
    for i in range(len(window) - TRIGRAM_N + 1):
        counts[tuple(window[i:i + TRIGRAM_N])] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p = np.array([counts.get(k, 0) / total for k in ref_keys])
    return float(jensenshannon(p, ref_vec) ** 2)


def encode_window(window, vocab, max_len=None):
    encoded = [vocab.get(str(s), 1) for s in window]
    if max_len and len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    elif max_len and len(encoded) > max_len:
        encoded = encoded[:max_len]
    return encoded


# ---------------------------------------------------------------------------
# Strategy 1: Distribution-Aware Mimicry Attack
# ---------------------------------------------------------------------------
class DistributionAwareMimicryAttack:
    """
    Greedy no-op injection that maximises local benign-trigram overlap.

    For each candidate (position, no-op), the score is the sum of
    benign reference probabilities for the trigrams that would be
    newly introduced by the insertion.  This is O(1) per candidate
    (a few dict lookups) and serves as a fast proxy for JSD reduction.
    """

    def __init__(self, benign_trigrams, benign_bigrams, noop_list,
                 max_ratio=MAX_INJECTION_RATIO):
        self.ref_tri = benign_trigrams
        self.ref_bi = benign_bigrams
        self.noops = noop_list
        self.max_ratio = max_ratio

    def _score_insertion(self, window, pos, noop):
        """Score = sum of benign ref probs for new trigrams created."""
        score = 0.0
        w = window
        # Trigrams that include the new no-op at position pos
        for offset in range(-TRIGRAM_N + 1, 1):
            start = pos + offset
            if start < 0 or start + TRIGRAM_N > len(w) + 1:
                continue
            # Build the trigram with the no-op spliced in
            tri = []
            for j in range(TRIGRAM_N):
                idx = start + j
                if idx == pos:
                    tri.append(noop)
                elif idx < pos:
                    tri.append(w[idx])
                else:  # idx > pos (shifted by 1 due to insertion)
                    tri.append(w[idx - 1])
            score += self.ref_tri.get(tuple(tri), 0)

        # Bigram context bonus
        if pos > 0:
            score += 0.5 * self.ref_bi.get((w[pos - 1], noop), 0)
        if pos < len(w):
            score += 0.5 * self.ref_bi.get((noop, w[pos]), 0)

        return score

    def generate(self, window):
        window = list(window)
        budget = int(len(window) * self.max_ratio)

        for _ in range(budget):
            best_score = -1
            best_pos = -1
            best_noop = None

            # Sample up to 15 candidate positions
            n_pos = len(window)
            if n_pos > 15:
                step = max(1, n_pos // 15)
                positions = list(range(0, n_pos, step))
            else:
                positions = list(range(n_pos))

            for pos in positions:
                for noop in self.noops:
                    s = self._score_insertion(window, pos, noop)
                    if s > best_score:
                        best_score = s
                        best_pos = pos
                        best_noop = noop

            if best_noop is None:
                break
            window.insert(best_pos, best_noop)

        return window


# ---------------------------------------------------------------------------
# Strategy 2: Random Mimicry Attack
# ---------------------------------------------------------------------------
class RandomMimicryAttack:
    """Uniformly random no-op insertion (ablation baseline)."""

    def __init__(self, noop_list, max_ratio=MAX_INJECTION_RATIO, rng=None):
        self.noops = noop_list
        self.max_ratio = max_ratio
        self.rng = rng or np.random.default_rng(SEED)

    def generate(self, window):
        window = list(window)
        budget = int(len(window) * self.max_ratio)
        for _ in range(budget):
            pos = self.rng.integers(0, len(window) + 1)
            noop = self.noops[self.rng.integers(0, len(self.noops))]
            window.insert(pos, noop)
        return window


# ---------------------------------------------------------------------------
# Strategy 3: Position-Aware Mimicry Attack
# ---------------------------------------------------------------------------
class PositionAwareMimicryAttack:
    """Inserts no-ops at high-salience positions (rare benign trigrams)."""

    def __init__(self, benign_trigrams, noop_list, max_ratio=MAX_INJECTION_RATIO):
        self.ref_tri = benign_trigrams
        self.noops = noop_list
        self.max_ratio = max_ratio

    def generate(self, window):
        window = list(window)
        budget = int(len(window) * self.max_ratio)

        # Salience per position: 1 - benign_prob of the trigram starting here
        saliences = []
        for i in range(len(window) - TRIGRAM_N + 1):
            tri = tuple(window[i:i + TRIGRAM_N])
            saliences.append(1.0 - self.ref_tri.get(tri, 0))
        while len(saliences) < len(window):
            saliences.append(0.0)

        ranked = sorted(range(len(window)),
                        key=lambda i: saliences[i], reverse=True)

        inserted_at = set()
        for pos in ranked:
            if len(inserted_at) >= budget:
                break
            adj = pos + sum(1 for p in inserted_at if p <= pos)

            best_noop = self.noops[0]
            best_sc = -1
            for noop in self.noops:
                sc = 0.0
                # New trigrams that would appear
                for off in range(-TRIGRAM_N + 1, 1):
                    start = adj + off
                    if start < 0 or start + TRIGRAM_N > len(window) + 1:
                        continue
                    tri = []
                    for j in range(TRIGRAM_N):
                        idx = start + j
                        if idx == adj:
                            tri.append(noop)
                        elif idx < adj:
                            tri.append(window[idx])
                        else:
                            tri.append(window[idx - 1])
                    sc += self.ref_tri.get(tuple(tri), 0)
                if sc > best_sc:
                    best_sc = sc
                    best_noop = noop

            window.insert(adj, best_noop)
            inserted_at.add(pos)

        return window


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Stage 2: Adversarial Mimicry Generation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load prepared data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading prepared data...")

    with open(PREPARED_DIR / "vocabulary.json") as f:
        vocab = json.load(f)
    with open(PREPARED_DIR / "metadata.json") as f:
        meta = json.load(f)
    with open(PREPARED_DIR / "benign_trigrams.pkl", "rb") as f:
        benign_tri = pickle.load(f)

    train_data = np.load(PREPARED_DIR / "train.npz", allow_pickle=True)
    test_data = np.load(PREPARED_DIR / "test.npz", allow_pickle=True)

    train_raw = train_data["raw_windows"]
    train_labels = train_data["labels"]
    train_cats = train_data["cat_ids"]
    test_raw = test_data["raw_windows"]
    test_labels = test_data["labels"]
    test_cats = test_data["cat_ids"]

    # Load benign traces for bigram reference
    from syscall_hids.data.adfa_ld_preparation import load_traces, DATA_ROOT
    benign_traces = load_traces(DATA_ROOT / "Training_Data_Master")
    benign_bigrams = build_bigram_ref(benign_traces)

    # Filter malicious windows
    train_mal = train_raw[train_labels == 1]
    train_mal_cats = train_cats[train_labels == 1]
    test_mal = test_raw[test_labels == 1]
    test_mal_cats = test_cats[test_labels == 1]

    print(f"  Train malicious: {len(train_mal)} windows")
    print(f"  Test malicious:  {len(test_mal)} windows")
    print(f"  Benign trigrams: {len(benign_tri)}")

    # Build vectorised JSD reference for reporting
    ref_keys = sorted(benign_tri.keys())
    ref_vec = np.array([benign_tri[k] for k in ref_keys])

    # ------------------------------------------------------------------
    # 2. Initialise strategies
    # ------------------------------------------------------------------
    print("\n[2/4] Initialising strategies...")

    strategies = {
        "distribution_aware": DistributionAwareMimicryAttack(
            benign_tri, benign_bigrams, NOOP_SYSCALLS),
        "random": RandomMimicryAttack(NOOP_SYSCALLS),
        "position_aware": PositionAwareMimicryAttack(benign_tri, NOOP_SYSCALLS),
    }

    # ------------------------------------------------------------------
    # 3. Generate adversarial windows
    # ------------------------------------------------------------------
    print("\n[3/4] Generating adversarial windows...")
    max_len = int(meta["window_size"] * (1 + MAX_INJECTION_RATIO))

    all_results = {}

    for name, strategy in strategies.items():
        t0 = time.time()
        print(f"\n  --- {name} ---")

        adv_train = []
        for i, w in enumerate(train_mal):
            adv_train.append(strategy.generate(w.tolist()))
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"    train {i+1}/{len(train_mal)}  ({elapsed:.1f}s)")

        if name == "random":
            strategy.rng = np.random.default_rng(SEED + 1)

        adv_test = []
        for i, w in enumerate(test_mal):
            adv_test.append(strategy.generate(w.tolist()))
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"    test  {i+1}/{len(test_mal)}  ({elapsed:.1f}s)")

        elapsed = time.time() - t0

        # Statistics
        orig_lens = np.array([TRIGRAM_N] * len(train_mal))  # placeholder
        tr_orig = np.array([len(w) for w in train_mal])
        tr_adv = np.array([len(w) for w in adv_train])
        te_orig = np.array([len(w) for w in test_mal])
        te_adv = np.array([len(w) for w in adv_test])

        tr_inj = ((tr_adv - tr_orig) / tr_orig * 100).mean()
        te_inj = ((te_adv - te_orig) / te_orig * 100).mean()

        # Sample JSD for a few windows
        sample_jsd_before = np.mean([
            compute_jsd(train_mal[i].tolist(), ref_keys, ref_vec)
            for i in range(min(50, len(train_mal)))
        ])
        sample_jsd_after = np.mean([
            compute_jsd(adv_train[i], ref_keys, ref_vec)
            for i in range(min(50, len(adv_train)))
        ])

        print(f"    Done in {elapsed:.1f}s")
        print(f"    Train injection: {tr_inj:.1f}%  "
              f"(len {tr_orig.mean():.0f} -> {tr_adv.mean():.0f})")
        print(f"    Test injection:  {te_inj:.1f}%")
        print(f"    JSD sample (train): {sample_jsd_before:.4f} -> {sample_jsd_after:.4f}")

        # Encode + pad
        adv_train_enc = np.array(
            [encode_window(w, vocab, max_len) for w in adv_train], dtype=np.int32)
        adv_test_enc = np.array(
            [encode_window(w, vocab, max_len) for w in adv_test], dtype=np.int32)

        all_results[name] = {
            "train": adv_train_enc,
            "test": adv_test_enc,
            "train_cats": train_mal_cats,
            "test_cats": test_mal_cats,
            "train_inj_pct": float(tr_inj),
            "test_inj_pct": float(te_inj),
            "jsd_before": float(sample_jsd_before),
            "jsd_after": float(sample_jsd_after),
        }

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    print("\n[4/4] Saving...")

    for name, data in all_results.items():
        np.savez_compressed(OUTPUT_DIR / f"{name}_train.npz",
                            windows=data["train"], cat_ids=data["train_cats"])
        np.savez_compressed(OUTPUT_DIR / f"{name}_test.npz",
                            windows=data["test"], cat_ids=data["test_cats"])

    adv_meta = {
        "strategies": list(strategies.keys()),
        "max_injection_ratio": MAX_INJECTION_RATIO,
        "noop_syscalls": NOOP_SYSCALLS,
        "window_size": meta["window_size"],
        "max_padded_length": max_len,
        "train_malicious_count": int(len(train_mal)),
        "test_malicious_count": int(len(test_mal)),
        "injection_stats": {
            n: {
                "train_pct": d["train_inj_pct"],
                "test_pct": d["test_inj_pct"],
                "jsd_before": d["jsd_before"],
                "jsd_after": d["jsd_after"],
            }
            for n, d in all_results.items()
        },
        "seed": SEED,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(adv_meta, f, indent=2)

    print(f"\n  Saved to {OUTPUT_DIR}/")
    for name in strategies:
        r = all_results[name]
        print(f"    {name}_train.npz  {r['train'].shape}")
        print(f"    {name}_test.npz   {r['test'].shape}")
    print(f"    metadata.json")

    print("\n" + "=" * 60)
    print("Stage 2 complete. Next: python -m syscall_hids.training.robust_encoder (Stage 3)")
    print("=" * 60)


if __name__ == "__main__":
    main()
