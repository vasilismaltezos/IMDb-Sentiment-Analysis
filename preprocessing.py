# texniti2/preprocessing.py
from __future__ import annotations

import os
import re
from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _read_text_file(path: str) -> str:
    """Read a text file robustly (IMDB reviews are plain text)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_split(split_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load one IMDB split (train or test) from:
      split_dir/pos/*.txt and split_dir/neg/*.txt
    Labels: pos -> 1, neg -> 0
    """
    pos_dir = os.path.join(split_dir, "pos")
    neg_dir = os.path.join(split_dir, "neg")

    if not os.path.isdir(pos_dir) or not os.path.isdir(neg_dir):
        raise FileNotFoundError(
            f"Could not find IMDB folders 'pos'/'neg' under: {split_dir}\n"
            f"Expected: {pos_dir} and {neg_dir}"
        )

    texts: List[str] = []
    labels: List[int] = []

    # sort for determinism
    for fname in sorted(os.listdir(pos_dir)):
        if fname.endswith(".txt"):
            texts.append(_read_text_file(os.path.join(pos_dir, fname)))
            labels.append(1)

    for fname in sorted(os.listdir(neg_dir)):
        if fname.endswith(".txt"):
            texts.append(_read_text_file(os.path.join(neg_dir, fname)))
            labels.append(0)

    return texts, labels


def load_imdb_data(base_dir: str = "../aclImdb"):
    """
    Loads IMDB dataset from base_dir:
      base_dir/train/pos, base_dir/train/neg
      base_dir/test/pos,  base_dir/test/neg

    Returns:
      x_train, y_train, x_test, y_test
    """
    # Allow calling from either A/ or A/texniti2/
    base_dir = os.path.normpath(base_dir)

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"IMDB base_dir not found or missing train/test.\n"
            f"Given base_dir: {base_dir}\n"
            f"Expected: {train_dir} and {test_dir}"
        )

    x_train, y_train = _load_split(train_dir)
    x_test, y_test = _load_split(test_dir)
    return x_train, y_train, x_test, y_test


def split_train_dev(
    x_train: List[str],
    y_train: List[int],
    dev_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Splits training into train/dev, stratified by labels.

    Returns:
      x_tr, y_tr, x_dev, y_dev
    """
    if not (0.0 < dev_ratio < 1.0):
        raise ValueError("dev_ratio must be in (0, 1).")

    x_tr, x_dev, y_tr, y_dev = train_test_split(
        x_train,
        y_train,
        test_size=dev_ratio,
        random_state=seed,
        stratify=y_train,
    )
    return x_tr, y_tr, x_dev, y_dev


def preprocess_texts(
    texts: List[str],
    lowercase: bool = True,
    keep_apostrophes: bool = True,
    min_token_len: int = 2,
) -> List[List[str]]:
    """
    Tokenize and normalize texts.
    Returns list of token lists (one list per document).

    - lowercase: lowercases tokens
    - keep_apostrophes: keeps words like "don't" as one token (regex already allows ')
    - min_token_len: filters very short tokens
    """
    processed: List[List[str]] = []

    for t in texts:
        if lowercase:
            t = t.lower()

        # basic tokenization
        tokens = _WORD_RE.findall(t)

        if not keep_apostrophes:
            tokens = [tok.replace("'", "") for tok in tokens]

        if min_token_len is not None and min_token_len > 1:
            tokens = [tok for tok in tokens if len(tok) >= min_token_len]

        processed.append(tokens)

    return processed
# --- helpers needed by feature_extraction.py ---

from collections import Counter


def get_word_document_frequency(tokenized_texts):
    """
    Document frequency: in how many documents each word appears.
    tokenized_texts: List[List[str]]
    returns: Counter(word -> df)
    """
    df = Counter()
    for doc in tokenized_texts:
        for w in set(doc):
            df[w] += 1
    return df

def filter_vocabulary(
    doc_freq,
    n_most_common,
    k_most_rare,
    max_vocab_size,
    min_doc_freq=1,
):
    """
    Filter vocabulary according to:
    - n_most_common: remove n most frequent words
    - k_most_rare: remove k rarest words
    - max_vocab_size: keep at most this many words
    - min_doc_freq: minimum document frequency
    """
    # keep only words with sufficient document frequency
    items = [(w, df) for w, df in doc_freq.items() if df >= min_doc_freq]

    # sort by document frequency (descending)
    items.sort(key=lambda x: x[1], reverse=True)

    # remove n most frequent
    if n_most_common > 0:
        items = items[n_most_common:]

    # remove k rarest
    if k_most_rare > 0 and len(items) > k_most_rare:
        items = items[:-k_most_rare]

    # keep up to max_vocab_size
    items = items[:max_vocab_size]

    vocab = [w for w, _ in items]
    return vocab

