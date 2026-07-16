"""
Script gia ti dimiourgia learning curves (kamyles mathisis)
Deiknoyme tin exeliksi tis apodosis toy model synartisei tou plithous ton dedomenwn ekpaideysis
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from preprocessing import (load_imdb_data, split_train_dev, preprocess_texts)
from feature_extraction import extract_features
from logistic_regression import train_logistic_regression
from evaluation import calculate_metrics_per_class


# YPERPARAMETROI - Prepei na einai oi idies me to main.py
HYPERPARAMETERS = {
    'n_most_common': 100,
    'k_most_rare': 100,
    'min_doc_freq': 10,
    'm_features': 1000,
    'dev_ratio': 0.2,
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42
}

# Pososta dedomenwn ekpaideysis gia ta experiments (ligoteres times gia taxytita)
TRAIN_SIZES = [0.1, 0.3, 0.5, 0.7, 1.0]

# Epilogoume gia poia klasi tha deiksoume ta metrics (0=negative, 1=positive)
TARGET_CLASS = 1  # Positive class


def run_learning_curve_experiment():
    """Trexoume ta experiments gia tis learning curves"""

    print("="*70)
    print("DIMIOURGIA LEARNING CURVES")
    print("="*70)

    # 1. Fortosi kai proepeksergasia dedomenwn
    print("\n" + "="*70)
    print("BHMA 1: PROETOIMASIA DEDOMENWN")
    print("="*70)

    train_data, test_data = load_imdb_data()

    np.random.seed(HYPERPARAMETERS['random_state'])
    train_texts, train_labels, dev_texts, dev_labels = split_train_dev(
        train_data,
        dev_ratio=HYPERPARAMETERS['dev_ratio']
    )

    train_labels = np.array(train_labels)
    dev_labels = np.array(dev_labels)

    print("\nProepeksergasia keimenwn...")
    train_texts_tokenized = preprocess_texts(train_texts)
    dev_texts_tokenized = preprocess_texts(dev_texts)

    test_texts = [test_data[i]['text'] for i in range(len(test_data))]
    test_labels = np.array([test_data[i]['label'] for i in range(len(test_data))])
    test_texts_tokenized = preprocess_texts(test_texts)

    # 2. Epilogi idioteton
    print("\nEpilogi idioteton...")
    vocabulary, X_train_full, X_dev, X_test = extract_features(
        train_texts_tokenized,
        train_labels,
        dev_texts_tokenized,
        test_texts_tokenized,
        n_most_common=HYPERPARAMETERS['n_most_common'],
        k_most_rare=HYPERPARAMETERS['k_most_rare'],
        m=HYPERPARAMETERS['m_features'],
        min_doc_freq=HYPERPARAMETERS['min_doc_freq']
    )

    # 3. Trexoume experiments gia diafores megethoi training set
    print("\n" + "="*70)
    print("BHMA 2: EKTELESI EXPERIMENTS")
    print("="*70)

    results = {
        'train_sizes': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'dev_precision': [],
        'dev_recall': [],
        'dev_f1': []
    }

    n_train_full = len(X_train_full)

    for train_size_ratio in TRAIN_SIZES:
        n_train = int(n_train_full * train_size_ratio)

        print(f"\n{'-'*70}")
        print(f"Training me {n_train}/{n_train_full} paradeigmata ({train_size_ratio*100:.0f}%)")
        print(f"{'-'*70}")

        # Epilogi yposynolou training data
        X_train_subset = X_train_full[:n_train]
        y_train_subset = train_labels[:n_train]

        # Ekpaideysi model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            penalty='l2',
            C=HYPERPARAMETERS['C'],
            solver='lbfgs',
            max_iter=HYPERPARAMETERS['max_iter'],
            random_state=HYPERPARAMETERS['random_state'],
            verbose=0
        )
        model.fit(X_train_subset, y_train_subset)

        # Axiologisi sta train data (mono to yposynolo pou xrisimopoiithike)
        y_train_pred = model.predict(X_train_subset)
        train_precision, train_recall, train_f1 = calculate_metrics_per_class(
            y_train_subset, y_train_pred, TARGET_CLASS
        )

        # Axiologisi sta dev data (ola ta dev data)
        y_dev_pred = model.predict(X_dev)
        dev_precision, dev_recall, dev_f1 = calculate_metrics_per_class(
            dev_labels, y_dev_pred, TARGET_CLASS
        )

        # Apothikeusi apotelesmaton
        results['train_sizes'].append(n_train)
        results['train_precision'].append(train_precision)
        results['train_recall'].append(train_recall)
        results['train_f1'].append(train_f1)
        results['dev_precision'].append(dev_precision)
        results['dev_recall'].append(dev_recall)
        results['dev_f1'].append(dev_f1)

        print(f"Train - Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Dev   - Precision: {dev_precision:.4f}, Recall: {dev_recall:.4f}, F1: {dev_f1:.4f}")

    # 4. Apothikeusi apotelesmaton
    os.makedirs('results', exist_ok=True)
    with open('results/learning_curves_data.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nApotelesmata apothikeythikan sto: results/learning_curves_data.pkl")

    return results


def plot_learning_curves(results):
    """Dimioyrgoume ta diagrmmata twn learning curves"""

    print("\n" + "="*70)
    print("BHMA 3: DIMIOURGIA DIAGRAMMATWN")
    print("="*70)

    class_name = "Positive" if TARGET_CLASS == 1 else "Negative"

    # Dimiourgia figure me 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Learning Curves - Logistic Regression (Klasi: {class_name})', fontsize=16)

    train_sizes = results['train_sizes']

    # Plot 1: Precision
    axes[0].plot(train_sizes, results['train_precision'], 'o-', label='Train', linewidth=2, markersize=6)
    axes[0].plot(train_sizes, results['dev_precision'], 's-', label='Dev', linewidth=2, markersize=6)
    axes[0].set_xlabel('Arithmos Dedomenwn Ekpaideysis', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision vs Training Size', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Plot 2: Recall
    axes[1].plot(train_sizes, results['train_recall'], 'o-', label='Train', linewidth=2, markersize=6)
    axes[1].plot(train_sizes, results['dev_recall'], 's-', label='Dev', linewidth=2, markersize=6)
    axes[1].set_xlabel('Arithmos Dedomenwn Ekpaideysis', fontsize=12)
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_title('Recall vs Training Size', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    # Plot 3: F1
    axes[2].plot(train_sizes, results['train_f1'], 'o-', label='Train', linewidth=2, markersize=6)
    axes[2].plot(train_sizes, results['dev_f1'], 's-', label='Dev', linewidth=2, markersize=6)
    axes[2].set_xlabel('Arithmos Dedomenwn Ekpaideysis', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('F1 Score vs Training Size', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1.05])

    plt.tight_layout()

    # Apothikeusi diagrammatos
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"Diagramma apothikeythike sto: results/learning_curves.png")

    plt.show()
    print("Diagramma emfanizetai!")


def main():
    """Kyria synartisi"""

    print("\nPROSOKHI: Autos o kwdikas mporei na parei arketa lepta na trexei!")
    print(f"Tha ektelestoun {len(TRAIN_SIZES)} experiments.\n")

    # Ektelesi experiments
    results = run_learning_curve_experiment()

    # Dimiourgia diagrammatwn
    plot_learning_curves(results)

    print("\n" + "="*70)
    print("OLOKLHROTHI!")
    print("="*70)


if __name__ == "__main__":
    main()
