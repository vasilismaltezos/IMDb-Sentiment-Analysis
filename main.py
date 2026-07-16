"""
IMDB Logistic Regression - FINAL FIXED MAIN
"""
import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from preprocessing import split_train_dev, preprocess_texts
from feature_extraction import extract_features
from logistic_regression import train_logistic_regression
from evaluation import (
    evaluate_model,
    print_evaluation_results,
    create_results_table_latex,
    calculate_metrics_per_class,
)

from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================
# HYPERPARAMETERS
# ==============================
HYPERPARAMETERS = {
    "n_most_common": 100,
    "k_most_rare": 100,
    "min_doc_freq": 10,
    "m_features": 1000,
    "dev_ratio": 0.2,
    "C": 1.0,
    "max_iter": 1000,
    "random_state": 42,
}

TRAIN_SIZES = [0.1, 0.3, 0.5, 0.7, 1.0]
TARGET_CLASS = 1


# ==============================
# READ DATASET DIRECTLY
# ==============================
def read_imdb_split(base_dir, split):
    x, y = [], []

    for label, sub in [(1, "pos"), (0, "neg")]:
        folder = os.path.join(base_dir, split, sub)

        if not os.path.isdir(folder):
            continue

        for fn in os.listdir(folder):
            if fn.endswith(".txt"):
                with open(os.path.join(folder, fn), "r", encoding="utf-8", errors="ignore") as f:
                    x.append(f.read())
                    y.append(label)

    return x, y


# ==============================
# LEARNING CURVES
# ==============================
def generate_learning_curves(X_train, y_train, X_dev, y_dev):

    n_train_full = len(X_train)

    results = {
        "train_sizes": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "dev_precision": [],
        "dev_recall": [],
        "dev_f1": [],
    }

    for ratio in TRAIN_SIZES:
        n_train = int(n_train_full * ratio)

        X_subset = X_train[:n_train]
        y_subset = y_train[:n_train]

        model = LogisticRegression(
            penalty="l2",
            C=HYPERPARAMETERS["C"],
            solver="lbfgs",
            max_iter=HYPERPARAMETERS["max_iter"],
            random_state=HYPERPARAMETERS["random_state"],
        )

        model.fit(X_subset, y_subset)

        y_pred_train = model.predict(X_subset)
        tp, tr, tf = calculate_metrics_per_class(y_subset, y_pred_train, TARGET_CLASS)

        y_pred_dev = model.predict(X_dev)
        dp, dr, df = calculate_metrics_per_class(y_dev, y_pred_dev, TARGET_CLASS)

        results["train_sizes"].append(n_train)
        results["train_precision"].append(tp)
        results["train_recall"].append(tr)
        results["train_f1"].append(tf)
        results["dev_precision"].append(dp)
        results["dev_recall"].append(dr)
        results["dev_f1"].append(df)

        os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Precision
    axes[0].plot(results["train_sizes"], results["train_precision"], label="Train Precision")
    axes[0].plot(results["train_sizes"], results["dev_precision"], label="Dev Precision")
    axes[0].set_title("Precision vs Training Size")
    axes[0].set_xlabel("Training size")
    axes[0].set_ylabel("Precision")
    axes[0].grid(True)
    axes[0].legend()

    # Recall
    axes[1].plot(results["train_sizes"], results["train_recall"], label="Train Recall")
    axes[1].plot(results["train_sizes"], results["dev_recall"], label="Dev Recall")
    axes[1].set_title("Recall vs Training Size")
    axes[1].set_xlabel("Training size")
    axes[1].set_ylabel("Recall")
    axes[1].grid(True)
    axes[1].legend()

    # F1
    axes[2].plot(results["train_sizes"], results["train_f1"], label="Train F1")
    axes[2].plot(results["train_sizes"], results["dev_f1"], label="Dev F1")
    axes[2].set_title("F1 vs Training Size")
    axes[2].set_xlabel("Training size")
    axes[2].set_ylabel("F1")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("results/learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    return results



# ==============================
# MAIN
# ==============================
def main():

    print("\n=== IMDB LOGISTIC REGRESSION ===\n")

    # ΚΛΕΙΔΩΜΕΝΟ DATASET PATH
    base_dir = r"C:\Users\vasil\Desktop\kainourio\3220267_32300114_3230071\B\imdb_data\aclImdb"

    print("Dataset path:", base_dir)

    # DEBUG COUNTS
    for split in ["train", "test"]:
        for sub in ["pos", "neg"]:
            folder = os.path.join(base_dir, split, sub)
            if os.path.isdir(folder):
                count = sum(1 for f in os.listdir(folder) if f.endswith(".txt"))
                print(f"{split}/{sub} txt:", count)
            else:
                print(f"{split}/{sub} MISSING")

    # READ DATA
    x_train, y_train = read_imdb_split(base_dir, "train")
    x_test, y_test = read_imdb_split(base_dir, "test")

    print("\nLoaded samples:")
    print("Train:", len(x_train))
    print("Test :", len(x_test))

    if len(x_train) == 0:
        raise RuntimeError("TRAIN dataset is empty.")
    if len(x_test) == 0:
        raise RuntimeError("TEST dataset is empty.")

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # SPLIT TRAIN / DEV
    train_texts, train_labels, dev_texts, dev_labels = split_train_dev(
        x_train, y_train, dev_ratio=HYPERPARAMETERS["dev_ratio"]
    )

    train_labels = np.array(train_labels)
    dev_labels = np.array(dev_labels)

    # PREPROCESS
    train_tok = preprocess_texts(train_texts)
    dev_tok = preprocess_texts(dev_texts)
    test_tok = preprocess_texts(x_test)

    # FEATURES
    vocab, X_train, X_dev, X_test = extract_features(
        train_tok,
        train_labels,
        dev_tok,
        test_tok,
        n_most_common=HYPERPARAMETERS["n_most_common"],
        k_most_rare=HYPERPARAMETERS["k_most_rare"],
        m=HYPERPARAMETERS["m_features"],
        min_doc_freq=HYPERPARAMETERS["min_doc_freq"],
    )

    # TRAIN MODEL
    model = train_logistic_regression(
        X_train,
        train_labels,
        C=HYPERPARAMETERS["C"],
        max_iter=HYPERPARAMETERS["max_iter"],
        random_state=HYPERPARAMETERS["random_state"],
    )

    # EVALUATION
    print("\n=== EVALUATION ===\n")

    train_res = evaluate_model(model, X_train, train_labels, "TRAIN")
    print_evaluation_results(train_res, "TRAIN")

    dev_res = evaluate_model(model, X_dev, dev_labels, "DEV")
    print_evaluation_results(dev_res, "DEV")

    test_res = evaluate_model(model, X_test, y_test, "TEST")
    print_evaluation_results(test_res, "TEST")

    # LEARNING CURVES
    lc = generate_learning_curves(X_train, train_labels, X_dev, dev_labels)

    # SAVE
    os.makedirs("models", exist_ok=True)

    with open("models/logistic_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("models/results.pkl", "wb") as f:
        pickle.dump({"train": train_res, "dev": dev_res, "test": test_res, "lc": lc}, f)

    print("\n✔ DONE — model, results, curves saved.\n")


if __name__ == "__main__":
    main()
