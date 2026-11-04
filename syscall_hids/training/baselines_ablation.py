"""
Baselines & Ablation Experiments

Runs baseline comparisons and ablation studies across multiple seeds.
Produces results/baselines_report.json and results/ablation_report.json.

CORRECTED runner: neural training delegates to corrected_pipeline.train_corrected
(paired TRADES KL, stratified validation-split checkpoint selection, true no-defense
baseline, corrected no-op token set). The in-place trainer that previously lived here
had three method defects (unpaired L_robust, test-set checkpoint selection, and a
"vanilla" baseline that was actually trained with adversarial data); those are fixed.

Baselines:
  B1 — N-gram TF-IDF + Random Forest / SVM (classical)
  B2 — LSTM encoder, L_cls only (neural, no defense)
  B3 — Multi-Scale CNN, L_cls only, CLEAN data only (true no-defense baseline B0, same arch as the full model) [= Ablation A1]

Ablations (same CNN architecture, different loss components; no adv-as-labeled data —
adversarial windows are used only as paired targets for L_robust):
  A1 — L_cls only              (alpha=0, beta=0)              [= B3/B0]
  A2 — L_cls + L_robust        (alpha=1, beta=0, paired KL)
  A3 — L_cls + L_contrastive   (alpha=0, beta=0.1)
  A4 — Full model              (alpha=1, beta=0.1, paired KL + contrastive)

Usage:
  python -m syscall_hids.training.baselines_ablation --all              # run everything
  python -m syscall_hids.training.baselines_ablation --baselines        # run B1, B2, B3 only
  python -m syscall_hids.training.baselines_ablation --ablations        # run A1, A2, A3 only
  python -m syscall_hids.training.baselines_ablation --models ngram lstm # run specific models
  python -m syscall_hids.training.baselines_ablation --seeds 42 123     # override default seeds
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PREPARED_DIR = Path("prepared_data")
ADVERSARIAL_DIR = Path("adversarial_data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

VOCAB_SIZE = json.load(open(PREPARED_DIR / "metadata.json"))["vocab_size"]
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_CLASSES = 2
DROPOUT = 0.3
CONV_KERNELS = [3, 5, 7]
CONV_CHANNELS = 64

BATCH_SIZE = 256
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTH = 0.1
TEMPERATURE = 0.5

CONTRAST_AUG_RATIO = 0.08
CONTRAST_MASK_RATIO = 0.05

# TOKEN ids (post-encode) of the i386 no-ops getpid, getuid32, geteuid32,
# getppid, time, gettimeofday — derived from vocabulary.json so they track the
# (training-only) vocabulary. A no-op missing from the vocabulary is an error,
# not a silent <UNK> (the original bug: raw 24/49 encoded to <UNK>).
_VOCAB = json.load(open(PREPARED_DIR / "vocabulary.json"))
NOOP_RAW = (20, 199, 201, 64, 13, 78)
_missing = [s for s in NOOP_RAW if str(s) not in _VOCAB]
assert not _missing, f"no-op syscalls absent from vocabulary: {_missing}"
NOOP_SYSCALLS = [_VOCAB[str(s)] for s in NOOP_RAW]
DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
MAX_LEN = 130

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Model Definitions
# ===========================================================================
class MultiScaleEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 conv_channels=CONV_CHANNELS, kernels=CONV_KERNELS,
                 hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, conv_channels, k, padding=k // 2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),
            )
            for k in kernels
        ])
        rep_dim = conv_channels * len(kernels)
        self.fc_rep = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

    def encode(self, x):
        e = self.embedding(x).permute(0, 2, 1)
        conv_outs = [block(e).squeeze(-1) for block in self.conv_blocks]
        h = torch.cat(conv_outs, dim=1)
        return self.fc_rep(h)

    def forward(self, x):
        h = self.encode(x)
        return self.classifier(h), h

    def project(self, h):
        return self.projector(h)


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 lstm_hidden=100, hidden_dim=HIDDEN_DIM,
                 num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, x):
        e = self.embedding(x)
        output, _ = self.lstm(e)  # (B, L, hidden*2)
        # Mean pooling over sequence (ignoring padding at position 0)
        mask = (x != 0).unsqueeze(-1).float()  # (B, L, 1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(pooled)

    def forward(self, x):
        h = self.encode(x)
        return self.classifier(h), h


# ===========================================================================
# Loss Functions
# ===========================================================================
def loss_cls(logits, labels, weight=None):
    return F.cross_entropy(logits, labels, weight=weight, label_smoothing=LABEL_SMOOTH)


def loss_robust(clean_logits, adv_logits):
    clean_prob = F.softmax(clean_logits.detach(), dim=1)
    clean_prob = clean_prob.clamp(min=1e-6)
    adv_log_prob = F.log_softmax(adv_logits, dim=1)
    return F.kl_div(adv_log_prob, clean_prob, reduction="batchmean")


def loss_contrastive(z_i, z_j, temperature=TEMPERATURE):
    batch = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * batch, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    pos_i = torch.arange(batch, device=z.device)
    pos_j = torch.arange(batch, 2 * batch, device=z.device)
    labels_ct = torch.cat([pos_j, pos_i])
    return F.cross_entropy(sim, labels_ct)


def augment_batch(windows, noop_tensor, aug_ratio=CONTRAST_AUG_RATIO,
                  mask_ratio=CONTRAST_MASK_RATIO):
    B, L = windows.shape
    aug = windows.clone()
    num_swap = max(1, int(L * aug_ratio))
    for _ in range(num_swap):
        pos = torch.randint(1, L, (B,), device=windows.device)
        idx = torch.randint(0, noop_tensor.size(0), (B,), device=windows.device)
        aug[torch.arange(B, device=windows.device), pos] = noop_tensor[idx]
    num_mask = max(1, int(L * mask_ratio))
    mask_pos = torch.randint(1, L, (B, num_mask), device=windows.device)
    aug.scatter_(1, mask_pos, 0)
    return aug


# ===========================================================================
# Data Loading
# ===========================================================================
def load_data():
    with open(PREPARED_DIR / "metadata.json") as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]

    train = np.load(PREPARED_DIR / "train.npz")
    test = np.load(PREPARED_DIR / "test.npz")

    X_train = torch.tensor(train["windows"], dtype=torch.long)
    y_train = torch.tensor(train["labels"], dtype=torch.long)
    cat_train = torch.tensor(train["cat_ids"], dtype=torch.long)

    X_test = torch.tensor(test["windows"], dtype=torch.long)
    y_test = torch.tensor(test["labels"], dtype=torch.long)
    cat_test = torch.tensor(test["cat_ids"], dtype=torch.long)

    adv_strategies = ["distribution_aware", "random", "position_aware"]
    adv_train_sets = {}
    adv_test_sets = {}
    for strat in adv_strategies:
        f_train = ADVERSARIAL_DIR / f"{strat}_train.npz"
        f_test = ADVERSARIAL_DIR / f"{strat}_test.npz"
        if f_train.exists():
            d = np.load(f_train)
            adv_train_sets[strat] = torch.tensor(d["windows"], dtype=torch.long)
        if f_test.exists():
            d = np.load(f_test)
            adv_test_sets[strat] = (torch.tensor(d["windows"], dtype=torch.long),
                                    torch.tensor(d["cat_ids"], dtype=torch.long))

    return {
        "vocab_size": vocab_size,
        "X_train": X_train, "y_train": y_train, "cat_train": cat_train,
        "X_test": X_test, "y_test": y_test, "cat_test": cat_test,
        "adv_train": adv_train_sets, "adv_test": adv_test_sets,
    }


def pad_to(tensor, target_len):
    if tensor.size(1) >= target_len:
        return tensor
    pad = torch.zeros(tensor.size(0), target_len - tensor.size(1), dtype=torch.long)
    return torch.cat([tensor, pad], dim=1)


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels, all_cats = [], [], []
    for batch in loader:
        x, y, c = [t.to(device) for t in batch]
        logits, _ = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
        all_cats.append(c.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    cats = torch.cat(all_cats)
    preds = logits.argmax(dim=1)
    probs = F.softmax(logits, dim=1)[:, 1]

    acc = (preds == labels).float().mean().item()
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0

    cat_metrics = {}
    for cat_id in cats.unique().tolist():
        mask = cats == cat_id
        cat_acc = (preds[mask] == labels[mask]).float().mean().item()
        cat_metrics[cat_id] = {"accuracy": cat_acc, "count": int(mask.sum())}

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels.numpy(), probs.numpy())
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc, "f1": f1, "precision": precision, "recall": recall,
        "fpr": fpr, "auc": auc, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "per_category": cat_metrics,
    }


# ===========================================================================
# Neural Model Training (parametric)
# ===========================================================================
_CP_DATA = None


def _corrected_data():
    """Load the corrected-pipeline data format once (cached)."""
    global _CP_DATA
    if _CP_DATA is None:
        from syscall_hids.training.corrected_pipeline import load_all
        _CP_DATA = load_all()
    return _CP_DATA


def train_neural_model(model_class, model_kwargs, data, alpha=0.0, beta=0.0,
                       use_adv_train=False, seed=42, tag="model"):
    """[CORRECTED] Delegates to corrected_pipeline.train_corrected so this ablation
    runner uses the same training and validation splits. Changes from the
    original in-place trainer (which had three method defects): (1) L_robust is a
    paired TRADES KL between each clean-malicious window and its own adversarial
    version (not two unrelated slices of a shuffled batch); (2) the best checkpoint
    is selected on a stratified validation split carved from train (not the test
    set); (3) `use_adv_train` now controls only adv-as-labeled augmentation (AT),
    while the paired robust loss is applied whenever alpha>0. The no-op token set is
    the corrected i386 vocabulary (see NOOP_SYSCALLS)."""
    from syscall_hids.training.corrected_pipeline import train_corrected
    data_all = _corrected_data()
    r = train_corrected(data_all, alpha=alpha, beta=beta, augment_adv=use_adv_train,
                        canon="none", seed=seed, tag=tag, encoder=model_class)
    n_params = sum(p.numel()
                   for p in model_class(vocab_size=data_all["vocab"]).parameters()
                   if p.requires_grad)
    return {
        "params": n_params,
        "clean": r["clean"],
        "adversarial": r["adv"],
        "best_f1": r["val_f1"],
        "config": {"alpha": alpha, "beta": beta,
                   "use_adv_train": use_adv_train, "seed": seed},
    }


# ===========================================================================
# N-gram Baseline
# ===========================================================================
def train_ngram_baseline(data, seed=42):
    """N-gram TF-IDF + Random Forest baseline."""
    print(f"\n  [B1_ngram] seed={seed}")
    np.random.seed(seed)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_auc_score

    def windows_to_strings(windows):
        return [" ".join(map(str, w)) for w in windows.numpy()]

    X_train_str = windows_to_strings(data["X_train"])
    X_test_str = windows_to_strings(data["X_test"])
    y_train_np = data["y_train"].numpy()
    y_test_np = data["y_test"].numpy()

    # Fit TF-IDF on 3-grams through 7-grams
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(3, 7), max_features=5000,
        sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train_str)
    X_test_tfidf = vectorizer.transform(X_test_str)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    rf.fit(X_train_tfidf, y_train_np)

    rf_preds = rf.predict(X_test_tfidf)
    rf_probs = rf.predict_proba(X_test_tfidf)[:, 1]

    # Linear SVM (calibrated for probabilities)
    svm = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", max_iter=5000, random_state=seed),
        cv=3)
    svm.fit(X_train_tfidf, y_train_np)

    svm_preds = svm.predict(X_test_tfidf)
    svm_probs = svm.predict_proba(X_test_tfidf)[:, 1]

    def compute_sklearn_metrics(preds, probs, labels):
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        acc = (tp + tn) / (tp + fp + fn + tn)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")
        return {"accuracy": acc, "f1": f1, "precision": precision,
                "recall": recall, "fpr": fpr, "auc": auc,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    # Pick the better classifier by F1
    rf_metrics = compute_sklearn_metrics(rf_preds, rf_probs, y_test_np)
    svm_metrics = compute_sklearn_metrics(svm_preds, svm_probs, y_test_np)

    best_name = "RandomForest" if rf_metrics["f1"] >= svm_metrics["f1"] else "SVM"
    best_clf = rf if best_name == "RandomForest" else svm
    clean_metrics = rf_metrics if best_name == "RandomForest" else svm_metrics

    print(f"    Best classifier: {best_name} "
          f"(RF F1={rf_metrics['f1']:.4f}, SVM F1={svm_metrics['f1']:.4f})")
    print(f"    Clean: F1={clean_metrics['f1']:.4f}  AUC={clean_metrics['auc']:.4f}  "
          f"Recall={clean_metrics['recall']:.4f}  FPR={clean_metrics['fpr']:.4f}")

    # Adversarial evaluation
    adv_metrics = {}
    for strat, (x_adv, c_adv) in data["adv_test"].items():
        adv_str = windows_to_strings(x_adv)
        adv_tfidf = vectorizer.transform(adv_str)
        adv_preds = best_clf.predict(adv_tfidf)
        adv_labels = np.ones(len(adv_preds), dtype=int)
        adv_m = compute_sklearn_metrics(adv_preds, np.zeros(len(adv_preds)),
                                         adv_labels)
        adv_metrics[strat] = adv_m
        print(f"    Adv ({strat[:12]:12s}): Recall={adv_m['recall']:.4f}")

    return {
        "params": f"N/A ({best_name})",
        "classifier": best_name,
        "clean": clean_metrics,
        "rf_metrics": rf_metrics,
        "svm_metrics": svm_metrics,
        "adversarial": adv_metrics,
        "config": {"seed": seed, "ngram_range": [3, 7], "max_features": 5000},
    }


# ===========================================================================
# Experiment Runners
# ===========================================================================
def aggregate_runs(runs, keys_to_agg=None):
    """Aggregate metrics across seeds: compute mean and std."""
    if keys_to_agg is None:
        keys_to_agg = ["accuracy", "f1", "precision", "recall", "fpr", "auc"]
    result = {}
    for key in keys_to_agg:
        values = []
        for run in runs:
            v = run.get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                values.append(v)
        if values:
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }
    return result


def run_baselines(seeds):
    data = load_data()
    results = {}

    # B1: N-gram
    print("\n" + "=" * 60)
    print("BASELINE B1: N-gram TF-IDF + RF/SVM")
    print("=" * 60)
    ngram_runs = []
    for seed in seeds:
        ngram_runs.append(train_ngram_baseline(data, seed=seed))
    results["B1_ngram"] = {
        "clean_agg": aggregate_runs([r["clean"] for r in ngram_runs]),
        "runs": [{"seed": s, "clean": r["clean"], "adversarial": r["adversarial"],
                   "classifier": r["classifier"], "config": r["config"]}
                  for s, r in zip(seeds, ngram_runs)],
    }
    for strat in ["distribution_aware", "random", "position_aware"]:
        adv_runs = [r["adversarial"].get(strat, {}) for r in ngram_runs]
        results["B1_ngram"][f"adv_{strat}_agg"] = aggregate_runs(adv_runs)

    # B2: LSTM
    print("\n" + "=" * 60)
    print("BASELINE B2: LSTM Encoder (L_cls only)")
    print("=" * 60)
    lstm_kwargs = {"vocab_size": data["vocab_size"]}
    lstm_runs = []
    for seed in seeds:
        r = train_neural_model(LSTMEncoder, lstm_kwargs, data,
                               alpha=0, beta=0, use_adv_train=False,
                               seed=seed, tag="B2_lstm")
        lstm_runs.append(r)
    results["B2_lstm"] = {
        "clean_agg": aggregate_runs([r["clean"] for r in lstm_runs]),
        "params": lstm_runs[0]["params"],
        "runs": [{"seed": s, "clean": r["clean"],
                   "adversarial": r["adversarial"], "config": r["config"]}
                  for s, r in zip(seeds, lstm_runs)],
    }
    for strat in ["distribution_aware", "random", "position_aware"]:
        adv_runs = [r["adversarial"].get(strat, {}) for r in lstm_runs]
        results["B2_lstm"][f"adv_{strat}_agg"] = aggregate_runs(adv_runs)

    # B3: Vanilla CNN (= Ablation A1)
    print("\n" + "=" * 60)
    print("BASELINE B3 / ABLATION A1: Vanilla CNN (L_cls only)")
    print("=" * 60)
    cnn_kwargs = {"vocab_size": data["vocab_size"]}
    cnn_runs = []
    for seed in seeds:
        r = train_neural_model(MultiScaleEncoder, cnn_kwargs, data,
                               alpha=0, beta=0, use_adv_train=False,
                               seed=seed, tag="B3_vanilla_cnn")
        cnn_runs.append(r)
    results["B3_vanilla_cnn"] = {
        "clean_agg": aggregate_runs([r["clean"] for r in cnn_runs]),
        "params": cnn_runs[0]["params"],
        "runs": [{"seed": s, "clean": r["clean"],
                   "adversarial": r["adversarial"], "config": r["config"]}
                  for s, r in zip(seeds, cnn_runs)],
    }
    for strat in ["distribution_aware", "random", "position_aware"]:
        adv_runs = [r["adversarial"].get(strat, {}) for r in cnn_runs]
        results["B3_vanilla_cnn"][f"adv_{strat}_agg"] = aggregate_runs(adv_runs)

    return results, cnn_runs


def run_ablations(seeds, cnn_runs=None, data=None):
    if data is None:
        data = load_data()

    results = {}
    cnn_kwargs = {"vocab_size": data["vocab_size"]}

    # A1 = B3 (already done if baselines ran)
    if cnn_runs is not None:
        a1_runs = cnn_runs
    else:
        print("\n" + "=" * 60)
        print("ABLATION A1: L_cls only (= B3)")
        print("=" * 60)
        a1_runs = []
        for seed in seeds:
            r = train_neural_model(MultiScaleEncoder, cnn_kwargs, data,
                                   alpha=0, beta=0, use_adv_train=False,
                                   seed=seed, tag="A1_cls_only")
            a1_runs.append(r)

    results["A1_cls_only"] = {
        "config": {"alpha": 0, "beta": 0, "use_adv_train": False},
        "clean_agg": aggregate_runs([r["clean"] for r in a1_runs]),
        "params": a1_runs[0]["params"],
        "runs": [{"seed": s, "clean": r["clean"],
                   "adversarial": r["adversarial"], "config": r["config"]}
                  for s, r in zip(seeds, a1_runs)],
    }
    for strat in ["distribution_aware", "random", "position_aware"]:
        adv_runs = [r["adversarial"].get(strat, {}) for r in a1_runs]
        results["A1_cls_only"][f"adv_{strat}_agg"] = aggregate_runs(adv_runs)

    # A2: L_cls + L_robust
    print("\n" + "=" * 60)
    print("ABLATION A2: L_cls + L_robust (no contrastive)")
    print("=" * 60)
    a2_runs = []
    for seed in seeds:
        r = train_neural_model(MultiScaleEncoder, cnn_kwargs, data,
                               alpha=1.0, beta=0, use_adv_train=False,
                               seed=seed, tag="A2_cls_robust")
        a2_runs.append(r)
    results["A2_cls_robust"] = {
        "config": {"alpha": 1.0, "beta": 0, "use_adv_train": False},
        "clean_agg": aggregate_runs([r["clean"] for r in a2_runs]),
        "params": a2_runs[0]["params"],
        "runs": [{"seed": s, "clean": r["clean"],
                   "adversarial": r["adversarial"], "config": r["config"]}
                  for s, r in zip(seeds, a2_runs)],
    }
    for strat in ["distribution_aware", "random", "position_aware"]:
        adv_runs = [r["adversarial"].get(strat, {}) for r in a2_runs]
        results["A2_cls_robust"][f"adv_{strat}_agg"] = aggregate_runs(adv_runs)

    # A3: L_cls + L_contrastive (no adversarial training data)
    print("\n" + "=" * 60)
    print("ABLATION A3: L_cls + L_contrastive (no adv data)")
    print("=" * 60)
    a3_runs = []
    for seed in seeds:
        r = train_neural_model(MultiScaleEncoder, cnn_kwargs, data,
                               alpha=0, beta=0.1, use_adv_train=False,
                               seed=seed, tag="A3_cls_contrastive")
        a3_runs.append(r)
    results["A3_cls_contrastive"] = {
        "config": {"alpha": 0, "beta": 0.1, "use_adv_train": False},
        "clean_agg": aggregate_runs([r["clean"] for r in a3_runs]),
        "params": a3_runs[0]["params"],
        "runs": [{"seed": s, "clean": r["clean"],
                   "adversarial": r["adversarial"], "config": r["config"]}
                  for s, r in zip(seeds, a3_runs)],
    }
    for strat in ["distribution_aware", "random", "position_aware"]:
        adv_runs = [r["adversarial"].get(strat, {}) for r in a3_runs]
        results["A3_cls_contrastive"][f"adv_{strat}_agg"] = aggregate_runs(adv_runs)

    # A4: Full model — re-run for all seeds
    print("\n" + "=" * 60)
    print("ABLATION A4: Full model (L_cls + L_robust + L_contrastive)")
    print("=" * 60)
    a4_runs = []
    for seed in seeds:
        r = train_neural_model(MultiScaleEncoder, cnn_kwargs, data,
                               alpha=1.0, beta=0.1, use_adv_train=False,
                               seed=seed, tag="A4_full")
        a4_runs.append(r)
    results["A4_full"] = {
        "config": {"alpha": 1.0, "beta": 0.1, "use_adv_train": False},
        "clean_agg": aggregate_runs([r["clean"] for r in a4_runs]),
        "params": a4_runs[0]["params"],
        "runs": [{"seed": s, "clean": r["clean"],
                   "adversarial": r["adversarial"], "config": r["config"]}
                  for s, r in zip(seeds, a4_runs)],
    }
    for strat in ["distribution_aware", "random", "position_aware"]:
        adv_runs = [r["adversarial"].get(strat, {}) for r in a4_runs]
        results["A4_full"][f"adv_{strat}_agg"] = aggregate_runs(adv_runs)

    return results


# ===========================================================================
# Reporting
# ===========================================================================
def print_comparison_table(title, models_data, metric_keys=None):
    """Print a formatted comparison table."""
    if metric_keys is None:
        metric_keys = ["auc", "f1", "recall", "fpr", "precision"]

    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    header = f"  {'Model':20s}"
    for mk in metric_keys:
        header += f"  {mk:>10s}"
    # Add adversarial recall columns
    header += f"  {'AdvRec_DA':>10s}  {'AdvRec_Rnd':>10s}  {'AdvRec_PA':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model_name, mdata in models_data:
        agg = mdata.get("clean_agg", {})
        row = f"  {model_name:20s}"
        for mk in metric_keys:
            if mk in agg:
                m, s = agg[mk]["mean"], agg[mk]["std"]
                row += f"  {m:>6.4f}±{s:<3.3f}"
            else:
                row += f"  {'N/A':>10s}"

        # Adversarial recall
        for strat_key in ["adv_distribution_aware_agg", "adv_random_agg",
                          "adv_position_aware_agg"]:
            adv = mdata.get(strat_key, {})
            if "recall" in adv:
                m, s = adv["recall"]["mean"], adv["recall"]["std"]
                row += f"  {m:>6.4f}±{s:<3.3f}"
            else:
                row += f"  {'N/A':>10s}"
        print(row)


def save_reports(baselines, ablations, seeds):
    """Save JSON reports and print summary tables."""
    RESULTS_DIR.mkdir(exist_ok=True)

    report = {"seeds": seeds, "baselines": baselines, "ablations": ablations}

    # Print baseline table
    bl_models = [
        ("B1 N-gram+RF/SVM", baselines.get("B1_ngram", {})),
        ("B2 LSTM", baselines.get("B2_lstm", {})),
        ("B3 Vanilla CNN", baselines.get("B3_vanilla_cnn", {})),
        ("Full Model (Ours)", ablations.get("A4_full", {})),
    ]
    print_comparison_table("BASELINE COMPARISON (mean±std across seeds)", bl_models)

    # Print ablation table
    abl_models = [
        ("A1 L_cls only", ablations.get("A1_cls_only", {})),
        ("A2 L_cls+L_robust", ablations.get("A2_cls_robust", {})),
        ("A3 L_cls+L_contrast", ablations.get("A3_cls_contrastive", {})),
        ("A4 Full model", ablations.get("A4_full", {})),
    ]
    print_comparison_table("ABLATION STUDY (mean±std across seeds)", abl_models)

    # Save
    # Flatten for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    with open(RESULTS_DIR / "baselines_report.json", "w") as f:
        json.dump(make_serializable(baselines), f, indent=2)
    with open(RESULTS_DIR / "ablation_report.json", "w") as f:
        json.dump(make_serializable(ablations), f, indent=2)
    with open(RESULTS_DIR / "full_comparison_report.json", "w") as f:
        json.dump(make_serializable(report), f, indent=2)

    print(f"\n  Saved: {RESULTS_DIR}/baselines_report.json")
    print(f"  Saved: {RESULTS_DIR}/ablation_report.json")
    print(f"  Saved: {RESULTS_DIR}/full_comparison_report.json")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Baselines & Ablation Experiments")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--baselines", action="store_true", help="Run baselines only")
    parser.add_argument("--ablations", action="store_true", help="Run ablations only")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds")
    args = parser.parse_args()

    if not any([args.all, args.baselines, args.ablations]):
        args.all = True

    seeds = args.seeds
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Baselines & Ablation Experiments")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seeds: {seeds}")
    print("=" * 60)

    baselines = {}
    cnn_runs = None

    if args.all or args.baselines:
        baselines, cnn_runs = run_baselines(seeds)

    if args.all or args.ablations:
        ablations = run_ablations(seeds, cnn_runs=cnn_runs)
    else:
        ablations = {}

    save_reports(baselines, ablations, seeds)

    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
