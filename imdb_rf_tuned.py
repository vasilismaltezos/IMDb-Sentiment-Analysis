

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
import matplotlib.pyplot as plt
from collections import Counter
import re
import tarfile
import urllib.request
from pathlib import Path
import pickle

print_gr = print


def download_imdb(data_dir='./imdb_data'):
    """Fortosi IMDB dataset"""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    tar_path = data_dir / 'aclImdb_v1.tar.gz'

    if not tar_path.exists():
        print_gr("Fortosi dataset...")
        urllib.request.urlretrieve('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', tar_path)

    if not (data_dir / 'aclImdb').exists():
        print_gr("Apoksinosi...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)

    return data_dir / 'aclImdb'


def load_reviews(path, split='train'):
    """Fortosi reviews"""
    reviews, labels = [], []
    for label_type, label in [('pos', 1), ('neg', 0)]:
        for f in (path / split / label_type).glob('*.txt'):
            reviews.append(f.read_text(encoding='utf-8'))
            labels.append(label)
    return reviews, labels


def clean_tokenize(text):
    """Katharismos + tokenization"""
    return re.sub(r'[^a-z\s]', '', re.sub(r'<[^>]+>', '', text.lower())).split()


def compute_ig(word, word_sets, labels, h_parent):
    """Information Gain"""
    mask = np.array([word in ws for ws in word_sets], dtype=bool)
    n, n_has = len(labels), mask.sum()

    if n_has == 0 or n_has == n:
        return 0

    def entropy(y):
        if len(y) == 0:
            return 0
        p = y.sum() / len(y)
        return 0 if p == 0 or p == 1 else -p*np.log2(p) - (1-p)*np.log2(1-p)

    return h_parent - (n_has/n * entropy(labels[mask]) + (n-n_has)/n * entropy(labels[~mask]))


def build_vocab(docs, labels, n=50, k=5, m=2000):
    """Kataskevi lexiloyiou me IG"""
    print_gr("Ypologismos IG...")

    # Document frequency
    df = Counter()
    for words in docs:
        df.update(set(words))

    # Afairesi sixnon/spanion
    words = sorted(df.items(), key=lambda x: x[1], reverse=True)[n:-k if k>0 else None]
    print_gr(f"Ypopsifies lexeis: {len(words)}")

    # IG computation
    word_sets = [set(w) for w in docs]
    labels_arr = np.array(labels)
    p = labels_arr.mean()
    h_parent = -p*np.log2(p) - (1-p)*np.log2(1-p)

    ig = {}
    for i, (w, _) in enumerate(words):
        if i % 500 == 0:
            print_gr(f"  {i}/{len(words)}")
        ig[w] = compute_ig(w, word_sets, labels_arr, h_parent)

    vocab = [w for w, _ in sorted(ig.items(), key=lambda x: x[1], reverse=True)[:m]]
    print_gr(f"Teliko lexiloyio: {len(vocab)}")
    return vocab


def vectorize(docs, vocab):
    """Binary feature vectors"""
    w2i = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)), dtype=np.int8)
    for i, words in enumerate(docs):
        for w in set(words):
            if w in w2i:
                X[i, w2i[w]] = 1
    return X


def evaluate(y_true, y_pred):
    """Metrics"""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1])
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    pM, rM, fM, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    return {
        'negative': {'precision': p[0], 'recall': r[0], 'f1': f[0], 'support': s[0]},
        'positive': {'precision': p[1], 'recall': r[1], 'f1': f[1], 'support': s[1]},
        'micro_avg': {'precision': pm, 'recall': rm, 'f1': fm},
        'macro_avg': {'precision': pM, 'recall': rM, 'f1': fM},
        'accuracy': accuracy_score(y_true, y_pred)
    }


def print_eval(res, name):
    """Ektyposi results"""
    print_gr(f"\n{'='*60}\n{name}\n{'='*60}")
    print_gr(f"{'Klasi':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print_gr("-"*60)

    for cls in ['negative', 'positive']:
        r = res[cls]
        print_gr(f"{cls.capitalize():<15} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f} {r['support']:<10}")

    print_gr("-"*60)
    for avg in ['micro_avg', 'macro_avg']:
        r = res[avg]
        print_gr(f"{avg.replace('_','-').capitalize():<15} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")

    print_gr(f"-"*60)
    print_gr(f"Accuracy: {res['accuracy']:.4f}\n{'='*60}\n")


def grid_search(X_train, y_train, X_dev, y_dev):
    """Grid Search gia hyperparameters"""
    print_gr("\n[FINE TUNING] Grid Search sta dev data...")

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    best_score = 0
    best_params = None

    total = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])
    current = 0

    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_samp in param_grid['min_samples_split']:
                current += 1
                print_gr(f"  [{current}/{total}] Testing: n_est={n_est}, max_depth={max_d}, min_samp={min_samp}")

                clf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_samp,
                    random_state=42,
                    n_jobs=-1
                )
                clf.fit(X_train, y_train)
                score = f1_score(y_dev, clf.predict(X_dev), average='macro')

                print_gr(f"      F1-macro: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = {'n_estimators': n_est, 'max_depth': max_d, 'min_samples_split': min_samp}
                    print_gr(f"      *** NEW BEST! ***")

    print_gr(f"\nBest params: {best_params}")
    print_gr(f"Best F1-macro: {best_score:.4f}\n")
    return best_params


def learning_curves(X_train, y_train, X_dev, y_dev, params):
    """Learning curves"""
    print_gr("\n[6] Learning Curves...")
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    res = {
        'train_sizes': [int(len(X_train)*s) for s in sizes],
        'train_precision': [], 'train_recall': [], 'train_f1': [],
        'dev_precision': [], 'dev_recall': [], 'dev_f1': []
    }

    for size in res['train_sizes']:
        print_gr(f"  Training me {size} samples...")
        idx = np.random.RandomState(42).permutation(len(X_train))[:size]

        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        clf.fit(X_train[idx], y_train[idx])

        train_eval = evaluate(y_train[idx], clf.predict(X_train[idx]))
        dev_eval = evaluate(y_dev, clf.predict(X_dev))

        res['train_precision'].append(train_eval['positive']['precision'])
        res['train_recall'].append(train_eval['positive']['recall'])
        res['train_f1'].append(train_eval['positive']['f1'])
        res['dev_precision'].append(dev_eval['positive']['precision'])
        res['dev_recall'].append(dev_eval['positive']['recall'])
        res['dev_f1'].append(dev_eval['positive']['f1'])

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, label in zip(axes, ['precision', 'recall', 'f1'], ['Precision', 'Recall', 'F1']):
        ax.plot(res['train_sizes'], res[f'train_{metric}'], 'o-', label='Training', linewidth=2, markersize=6)
        ax.plot(res['train_sizes'], res[f'dev_{metric}'], 's-', label='Development', linewidth=2, markersize=6)
        ax.set_xlabel('Plithos Paradeigmaton')
        ax.set_ylabel(label)
        ax.set_title(f'{label} (Positive Class)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curves_rf.png', dpi=300, bbox_inches='tight')
    print_gr("Saved: learning_curves_rf.png")
    plt.close()


def main():
    print_gr("="*60)
    print_gr("IMDB Classification - Random Forest + Fine Tuning")
    print_gr("="*60)

    # [1] Load data
    print_gr("\n[1] Fortosi IMDB...")
    path = download_imdb()
    train_rev, train_lab = load_reviews(path, 'train')
    test_rev, test_lab = load_reviews(path, 'test')
    print_gr(f"Train: {len(train_rev)}, Test: {len(test_rev)}")

    # [2] Preprocess
    print_gr("\n[2] Proepeksergasia...")
    train_tok = [clean_tokenize(r) for r in train_rev]
    test_tok = [clean_tokenize(r) for r in test_rev]

    # [3] Split train/dev
    print_gr("\n[3] Split train/dev (80/20)...")
    n_dev = int(len(train_rev) * 0.2)
    idx = np.random.RandomState(42).permutation(len(train_rev))
    train_idx, dev_idx = idx[:-n_dev], idx[-n_dev:]

    tr_tok = [train_tok[i] for i in train_idx]
    tr_lab = [train_lab[i] for i in train_idx]
    dev_tok = [train_tok[i] for i in dev_idx]
    dev_lab = [train_lab[i] for i in dev_idx]
    print_gr(f"Final train: {len(tr_tok)}, Dev: {len(dev_tok)}")

    # [4] Build vocab
    print_gr("\n[4] Kataskevi lexiloyiou...")
    vocab = build_vocab(tr_tok, tr_lab, n=50, k=5, m=2000)
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # [5] Extract features
    print_gr("\n[5] Eksagogi features...")
    X_train = vectorize(tr_tok, vocab)
    X_dev = vectorize(dev_tok, vocab)
    X_test = vectorize(test_tok, vocab)
    y_train, y_dev, y_test = np.array(tr_lab), np.array(dev_lab), np.array(test_lab)
    print_gr(f"Shapes: X_train={X_train.shape}, X_dev={X_dev.shape}, X_test={X_test.shape}")

    # [6] Fine tuning - Grid Search
    best_params = grid_search(X_train, y_train, X_dev, y_dev)

    # [7] Learning curves me best params
    learning_curves(X_train, y_train, X_dev, y_dev, best_params)

    # [8] Final training me best params
    print_gr("\n[7] Teliki ekpaideysi me best params...")
    clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # [9] Evaluation
    print_gr("\n[8] Axiologisi...")
    print_eval(evaluate(y_train, clf.predict(X_train)), "Training Set")
    print_eval(evaluate(y_dev, clf.predict(X_dev)), "Development Set")
    print_eval(evaluate(y_test, clf.predict(X_test)), "Test Set")

    print_gr("\n" + "="*60)
    print_gr(f"BEST HYPERPARAMETERS:")
    print_gr(f"  n_estimators: {best_params['n_estimators']}")
    print_gr(f"  max_depth: {best_params['max_depth']}")
    print_gr(f"  min_samples_split: {best_params['min_samples_split']}")
    print_gr(f"  n (top frequent removed): 50")
    print_gr(f"  k (bottom rare removed): 5")
    print_gr(f"  m (vocabulary size): 2000")
    print_gr("="*60)
    print_gr("OLOKLIROSI!")
    print_gr("="*60)


if __name__ == "__main__":
    main()
