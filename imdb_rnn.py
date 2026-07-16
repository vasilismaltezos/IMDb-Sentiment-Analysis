

import os
import re
import tarfile
import urllib.request
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import gensim.downloader as api

# ----------------------------
# Fixed CPU device (no print)
# ----------------------------
device = torch.device("cpu")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# CPU threads: tweak 4/6/8 depending on your CPU
try:
    torch.set_num_threads(6)
except Exception:
    pass

# --------------------------
# Download + Load IMDB
# --------------------------
def download_imdb(d="./imdb_data"):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    t = d / "aclImdb_v1.tar.gz"

    if not t.exists():
        print("Downloading IMDB dataset...")
        urllib.request.urlretrieve(
            "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", t
        )

    if not (d / "aclImdb").exists():
        print("Extracting IMDB dataset...")
        with tarfile.open(t, "r:gz") as tf:
            tf.extractall(d)

    return d / "aclImdb"


def load_split(base_path: Path, split="train"):
    texts, labels = [], []
    for sub, lab in [("pos", 1), ("neg", 0)]:
        for f in (base_path / split / sub).glob("*.txt"):
            texts.append(f.read_text(encoding="utf-8", errors="ignore"))
            labels.append(lab)
    return texts, labels


# Fast tokenizer: keep only lowercase a-z sequences
_WORD_RE = re.compile(r"[a-z]+")


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return _WORD_RE.findall(text)


# --------------------------
# Dataset
# --------------------------
class IMDBDataset(Dataset):
    def __init__(self, texts_tok, labels, word2idx, max_len=160):
        self.texts = texts_tok
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        self.pad = word2idx["<PAD>"]
        self.unk = word2idx["<UNK>"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx][: self.max_len]
        ids = [self.word2idx.get(w, self.unk) for w in words]
        length = len(ids)
        if length < self.max_len:
            ids += [self.pad] * (self.max_len - length)

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# --------------------------
# Model
# --------------------------
class BiRNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        embeddings=None,
        pooling="max",      # 'max' or 'attention'
        cell_type="GRU",    # 'GRU' faster on CPU, or 'LSTM'
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(embeddings, dtype=torch.float32)
            )

        self.pooling = pooling
        cell_type = cell_type.upper()
        rnn_cls = nn.LSTM if cell_type == "LSTM" else nn.GRU

        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )

        if pooling == "attention":
            self.attn = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x, lengths):
        emb = self.embedding(x)  # (B, T, E)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B, T, 2H)

        if self.pooling == "max":
            pooled = out.max(dim=1).values
        else:
            w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
            pooled = torch.sum(w * out, dim=1)

        pooled = self.dropout(pooled)
        return self.fc(pooled)


# --------------------------
# Metrics + Plots
# --------------------------
def get_metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
    pM, rM, fM, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return {
        "neg": {"p": p[0], "r": r[0], "f": f[0], "s": s[0]},
        "pos": {"p": p[1], "r": r[1], "f": f[1], "s": s[1]},
        "mi": {"p": pm, "r": rm, "f": fm},
        "ma": {"p": pM, "r": rM, "f": fM},
        "acc": accuracy_score(y_true, y_pred),
    }


def print_metrics(res, title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(f"{'Class':<10}{'Prec':<10}{'Rec':<10}{'F1':<10}{'Sup':<10}")
    print("-" * 60)
    for c in ["neg", "pos"]:
        rr = res[c]
        print(
            f"{c.upper():<10}{rr['p']:<10.4f}{rr['r']:<10.4f}{rr['f']:<10.4f}{rr['s']:<10}"
        )
    print("-" * 60)
    for a in ["mi", "ma"]:
        rr = res[a]
        print(f"{a.upper():<10}{rr['p']:<10.4f}{rr['r']:<10.4f}{rr['f']:<10.4f}")
    print("-" * 60)
    print(f"Accuracy: {res['acc']:.4f}\n{'='*60}\n")


def plot_losses(train_losses, dev_losses, save_path="loss_curves.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(dev_losses, label="Development Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Development Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# --------------------------
# Train / Eval (CPU)
# --------------------------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, lens, y in loader:
        x = x.to(device)
        lens = lens.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, lens)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, labels = [], []

    for x, lens, y in loader:
        x = x.to(device)
        lens = lens.to(device)
        y = y.to(device)

        out = model(x, lens)
        loss = criterion(out, y)

        total_loss += loss.item() * y.size(0)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    return total_loss / len(labels), acc, preds, labels


# --------------------------
# Embeddings + caching
# --------------------------
def set_gensim_cache_dir(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GENSIM_DATA_DIR"] = str(cache_dir.resolve())


def build_vocab_and_embeddings(texts_tok, wv, min_freq=3, max_vocab=25000):
    counts = {}
    for doc in texts_tok:
        for w in doc:
            counts[w] = counts.get(w, 0) + 1

    candidates = [w for w, c in counts.items() if c >= min_freq and w in wv]
    candidates.sort(key=lambda w: counts[w], reverse=True)
    candidates = candidates[:max_vocab]

    vocab = ["<PAD>", "<UNK>"] + candidates
    word2idx = {w: i for i, w in enumerate(vocab)}

    embed_dim = wv.vector_size
    emb = np.zeros((len(vocab), embed_dim), dtype=np.float32)
    emb[1] = (np.random.randn(embed_dim).astype(np.float32) * 0.1)  # <UNK>
    for i, w in enumerate(vocab[2:], start=2):
        emb[i] = wv[w]

    return word2idx, emb


def main():
    # ----------------------------
    # Hyperparameters (CPU FAST)
    # ----------------------------
    MAX_LEN = 160        # smaller -> faster
    BATCH_SIZE = 128     # increase if RAM allows (try 64 if memory tight)
    EMBED_DIM = 100
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    EPOCHS = 6           # fewer epochs -> faster
    LR = 1e-3
    POOLING = "max"      # attention slower, use max for speed
    CELL_TYPE = "GRU"    # GRU faster on CPU

    # Early stopping
    PATIENCE = 2

    # ----------------------------
    # Caches
    # ----------------------------
    ROOT = Path(".")
    CACHE = ROOT / "cache_b"
    CACHE.mkdir(exist_ok=True)

    TOK_CACHE = CACHE / "imdb_tok_split.pkl"
    VOC_CACHE = CACHE / "vocab_emb_glove100_min3_max25k.pkl"

    set_gensim_cache_dir(CACHE / "gensim_data")

    print("=" * 60)
    print("IMDB Bidirectional RNN - PyTorch (CPU FAST)")
    print("=" * 60)
    print(f"Hyperparameters: max_len={MAX_LEN}, batch={BATCH_SIZE}, epochs={EPOCHS}, "
          f"cell={CELL_TYPE}, layers={NUM_LAYERS}, hidden={HIDDEN_DIM}, pooling={POOLING}")
    print(f"Early stopping: patience={PATIENCE}")

    # ----------------------------
    # Load/tokenize (cached)
    # ----------------------------
    if TOK_CACHE.exists():
        print("\n[1] Loading cached tokenized data...")
        tr_tok, tr_lab, dev_tok, dev_lab, test_tok, test_lab = pickle.loads(TOK_CACHE.read_bytes())
    else:
        print("\n[1] Downloading/loading IMDB...")
        path = download_imdb("./imdb_data")
        train_rev, train_lab = load_split(path, "train")
        test_rev, test_lab = load_split(path, "test")
        print(f"Train: {len(train_rev)}, Test: {len(test_rev)}")

        print("\n[2] Tokenizing (first time only)...")
        train_tok = [tokenize(t) for t in train_rev]
        test_tok = [tokenize(t) for t in test_rev]

        print("\n[3] Splitting train/dev (80/20)...")
        n_dev = int(0.2 * len(train_tok))
        idx = np.random.RandomState(SEED).permutation(len(train_tok))
        train_idx, dev_idx = idx[:-n_dev], idx[-n_dev:]

        tr_tok = [train_tok[i] for i in train_idx]
        tr_lab = [train_lab[i] for i in train_idx]
        dev_tok = [train_tok[i] for i in dev_idx]
        dev_lab = [train_lab[i] for i in dev_idx]

        TOK_CACHE.write_bytes(pickle.dumps((tr_tok, tr_lab, dev_tok, dev_lab, test_tok, test_lab)))
        print(f"Saved token cache -> {TOK_CACHE}")

    print(f"Final: train={len(tr_tok)}, dev={len(dev_tok)}, test={len(test_tok)}")

    # ----------------------------
    # Load embeddings + build vocab (cached)
    # ----------------------------
    if VOC_CACHE.exists():
        print("\n[4] Loading cached vocab/embeddings...")
        word2idx, embeddings = pickle.loads(VOC_CACHE.read_bytes())
    else:
        print("\n[4] Loading GloVe-100 (first time may take a while)...")
        wv = api.load("glove-wiki-gigaword-100")
        print(f"Loaded vectors: {len(wv)} | dim={wv.vector_size}")

        print("\n[5] Building vocab+embedding matrix (min_freq=3, max_vocab=25k)...")
        word2idx, embeddings = build_vocab_and_embeddings(tr_tok, wv, min_freq=3, max_vocab=25000)

        VOC_CACHE.write_bytes(pickle.dumps((word2idx, embeddings)))
        print(f"Saved vocab cache -> {VOC_CACHE}")

    print(f"Vocab size: {len(word2idx)}")

    # ----------------------------
    # DataLoaders (CPU)
    # ----------------------------
    # On Windows, too many workers can hurt; 2 is a safe default.
    num_workers = 2

    train_ds = IMDBDataset(tr_tok, tr_lab, word2idx, max_len=MAX_LEN)
    dev_ds = IMDBDataset(dev_tok, dev_lab, word2idx, max_len=MAX_LEN)
    test_ds = IMDBDataset(test_tok, test_lab, word2idx, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=num_workers)

    # ----------------------------
    # Model
    # ----------------------------
    print("\n[6] Creating model...")
    model = BiRNNClassifier(
        vocab_size=len(word2idx),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        embeddings=embeddings,
        pooling=POOLING,
        cell_type=CELL_TYPE,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ----------------------------
    # Training (best epoch by dev loss) + Early stopping
    # ----------------------------
    print("\n[7] Training...")
    train_losses, dev_losses = [], []
    best_dev_loss, best_epoch = float("inf"), -1
    bad = 0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        dv_loss, dv_acc, _, _ = evaluate(model, dev_loader, criterion)

        train_losses.append(tr_loss)
        dev_losses.append(dv_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
              f"Dev Loss={dv_loss:.4f} Acc={dv_acc:.4f}")

        if dv_loss < best_dev_loss:
            best_dev_loss = dv_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model.pt")
            print("  -> New best model saved!")
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("Early stopping!")
                break

    print(f"\nBest epoch: {best_epoch+1} | dev loss: {best_dev_loss:.4f}")

    # ----------------------------
    # Plot losses
    # ----------------------------
    print("\n[8] Saving loss curves...")
    plot_losses(train_losses, dev_losses, save_path="loss_curves.png")

    # ----------------------------
    # Evaluate best model
    # ----------------------------
    print("\n[9] Evaluating best model...")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    _, _, train_preds, train_labels = evaluate(model, train_loader, criterion)
    print_metrics(get_metrics(train_labels, train_preds), "Training Set (Best Epoch)")

    _, _, dev_preds, dev_labels = evaluate(model, dev_loader, criterion)
    print_metrics(get_metrics(dev_labels, dev_preds), "Development Set (Best Epoch)")

    _, _, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print_metrics(get_metrics(test_labels, test_preds), "Test Set (Best Epoch)")

    print("\nFINAL HYPERPARAMETERS:")
    print(f"  Cell type: {CELL_TYPE} (stacked bidirectional)")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Embed dim: {EMBED_DIM} (GloVe-100)")
    print(f"  Pooling: {POOLING}")
    print(f"  Max seq length: {MAX_LEN}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max epochs: {EPOCHS} (early stopping patience={PATIENCE})")
    print(f"  Optimizer: Adam | lr={LR}")
    print(f"  Best epoch: {best_epoch+1}")
    print("DONE!")


if __name__ == "__main__":
    main()
