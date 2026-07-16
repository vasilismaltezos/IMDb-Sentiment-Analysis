"""
Module gia tin axiologisi toy model (metrics)
"""
import numpy as np


def calculate_confusion_matrix(y_true, y_pred, n_classes=2):
    """Ypologizoume ton confusion matrix"""
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1

    return cm


def calculate_metrics_per_class(y_true, y_pred, class_label):
    """
    Ypologizoume precision, recall, F1 gia mia sigkekrimeni klasi

    Returns:
        precision, recall, f1
    """
    # True Positives: provlepsi = class_label KAI pragmatiko = class_label
    tp = np.sum((y_pred == class_label) & (y_true == class_label))

    # False Positives: provlepsi = class_label ALLA pragmatiko != class_label
    fp = np.sum((y_pred == class_label) & (y_true != class_label))

    # False Negatives: provlepsi != class_label ALLA pragmatiko = class_label
    fn = np.sum((y_pred != class_label) & (y_true == class_label))

    # Precision
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    # F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def calculate_micro_averaged_metrics(y_true, y_pred, n_classes=2):
    """
    Ypologizoume micro-averaged precision, recall, F1

    Sto micro-averaging, athroizoume ola ta TP, FP, FN apo oles tis klasis
    kai ypologizoume ta metrics
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for class_label in range(n_classes):
        tp = np.sum((y_pred == class_label) & (y_true == class_label))
        fp = np.sum((y_pred == class_label) & (y_true != class_label))
        fn = np.sum((y_pred != class_label) & (y_true == class_label))

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Micro-averaged precision
    if total_tp + total_fp == 0:
        precision = 0.0
    else:
        precision = total_tp / (total_tp + total_fp)

    # Micro-averaged recall
    if total_tp + total_fn == 0:
        recall = 0.0
    else:
        recall = total_tp / (total_tp + total_fn)

    # Micro-averaged F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def calculate_macro_averaged_metrics(y_true, y_pred, n_classes=2):
    """
    Ypologizoume macro-averaged precision, recall, F1

    Sto macro-averaging, ypologizoume ta metrics gia kathe klasi
    kai kanoume ton meso oro tous
    """
    precisions = []
    recalls = []
    f1s = []

    for class_label in range(n_classes):
        precision, recall, f1 = calculate_metrics_per_class(y_true, y_pred, class_label)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    return macro_precision, macro_recall, macro_f1


def evaluate_model(model, X, y, dataset_name=""):
    """
    Axiologoume ena model kai epistrefoume ola ta metrics

    Returns:
        dictionary me ola ta metrics
    """
    # Provlepseis
    y_pred = model.predict(X)

    # Accuracy
    accuracy = np.mean(y_pred == y)

    # Metrics gia kathe klasi
    metrics_class_0 = calculate_metrics_per_class(y, y_pred, class_label=0)
    metrics_class_1 = calculate_metrics_per_class(y, y_pred, class_label=1)

    # Micro kai macro averaged
    micro_metrics = calculate_micro_averaged_metrics(y, y_pred)
    macro_metrics = calculate_macro_averaged_metrics(y, y_pred)

    results = {
        'accuracy': accuracy,
        'class_0': {
            'precision': metrics_class_0[0],
            'recall': metrics_class_0[1],
            'f1': metrics_class_0[2]
        },
        'class_1': {
            'precision': metrics_class_1[0],
            'recall': metrics_class_1[1],
            'f1': metrics_class_1[2]
        },
        'micro_avg': {
            'precision': micro_metrics[0],
            'recall': micro_metrics[1],
            'f1': micro_metrics[2]
        },
        'macro_avg': {
            'precision': macro_metrics[0],
            'recall': macro_metrics[1],
            'f1': macro_metrics[2]
        }
    }

    return results


def print_evaluation_results(results, dataset_name=""):
    """Ektypwnoume ta apotelesmata axiologisis"""
    print(f"\n{'='*60}")
    print(f"APOTELESMATA {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"\n{'-'*60}")
    print(f"{'Klasi':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"{'-'*60}")

    # Class 0 (Negative)
    print(f"{'Negative (0)':<15} "
          f"{results['class_0']['precision']:<12.4f} "
          f"{results['class_0']['recall']:<12.4f} "
          f"{results['class_0']['f1']:<12.4f}")

    # Class 1 (Positive)
    print(f"{'Positive (1)':<15} "
          f"{results['class_1']['precision']:<12.4f} "
          f"{results['class_1']['recall']:<12.4f} "
          f"{results['class_1']['f1']:<12.4f}")

    print(f"{'-'*60}")

    # Micro average
    print(f"{'Micro-avg':<15} "
          f"{results['micro_avg']['precision']:<12.4f} "
          f"{results['micro_avg']['recall']:<12.4f} "
          f"{results['micro_avg']['f1']:<12.4f}")

    # Macro average
    print(f"{'Macro-avg':<15} "
          f"{results['macro_avg']['precision']:<12.4f} "
          f"{results['macro_avg']['recall']:<12.4f} "
          f"{results['macro_avg']['f1']:<12.4f}")

    print(f"{'='*60}\n")


def create_results_table_latex(results, dataset_name="Test"):
    """Dimioyrgoume pinaka me ta apotelesmata se LaTeX format"""
    latex = f"% Apotelesmata gia {dataset_name} set\n"
    latex += "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\begin{tabular}{|l|c|c|c|}\n"
    latex += "\\hline\n"
    latex += "Klasi & Precision & Recall & F1 \\\\\n"
    latex += "\\hline\n"

    latex += f"Negative (0) & {results['class_0']['precision']:.4f} & "
    latex += f"{results['class_0']['recall']:.4f} & {results['class_0']['f1']:.4f} \\\\\n"

    latex += f"Positive (1) & {results['class_1']['precision']:.4f} & "
    latex += f"{results['class_1']['recall']:.4f} & {results['class_1']['f1']:.4f} \\\\\n"

    latex += "\\hline\n"

    latex += f"Micro-avg & {results['micro_avg']['precision']:.4f} & "
    latex += f"{results['micro_avg']['recall']:.4f} & {results['micro_avg']['f1']:.4f} \\\\\n"

    latex += f"Macro-avg & {results['macro_avg']['precision']:.4f} & "
    latex += f"{results['macro_avg']['recall']:.4f} & {results['macro_avg']['f1']:.4f} \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += f"\\caption{{Apotelesmata gia {dataset_name} set (Accuracy: {results['accuracy']:.4f})}}\n"
    latex += "\\end{table}\n"

    return latex
