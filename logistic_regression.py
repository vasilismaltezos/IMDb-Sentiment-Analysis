"""
Logistic Regression gia IMDB Sentiment Analysis
Xrisimopoioyme to scikit-learn LogisticRegression me SGD solver
"""
from sklearn.linear_model import LogisticRegression


def create_logistic_regression_model(C=1.0, max_iter=1000, random_state=42, verbose=False):
    """
    Dimiourgisi Logistic Regression model me SGD solver

    Parameters:
    -----------
    C : float
        Inverse of regularization strength (mikrotero = megalyteri regularization)
        C = 1/lambda, opou lambda einai i parametros L2 regularization
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random seed gia reproducibility
    verbose : bool
        An True, emfanizoume progress messages

    Returns:
    --------
    model : LogisticRegression
        To LogisticRegression model toy scikit-learn
    """
    if verbose:
        print(f"\nDimiourgia Logistic Regression model:")
        print(f"  Solver: lbfgs (quasi-Newton method)")
        print(f"  C (inverse regularization): {C}")
        print(f"  Max iterations: {max_iter}")
        print(f"  Random state: {random_state}")

    # Dimiourgoume to model me L2 regularization (default)
    model = LogisticRegression(
        penalty='l2',           # L2 regularization
        C=C,                    # Inverse regularization strength
        solver='lbfgs',         # Optimization algorithm (kalyteri apo SGD gia mikra datasets)
        max_iter=max_iter,      # Maximum epochs
        random_state=random_state,
        verbose=1 if verbose else 0
    )

    return model


def train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000, random_state=42):
    """
    Ekpaideysi Logistic Regression model

    Parameters:
    -----------
    X_train : numpy array
        Training features
    y_train : numpy array
        Training labels
    C : float
        Inverse regularization strength
    max_iter : int
        Maximum iterations
    random_state : int
        Random seed

    Returns:
    --------
    model : trained LogisticRegression model
    """
    print("\n" + "="*70)
    print("EKPAIDEYSI LOGISTIC REGRESSION")
    print("="*70)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"C (1/lambda): {C}")
    print(f"Max iterations: {max_iter}")

    # Dimiourgia kai ekpaideysi toy model
    model = create_logistic_regression_model(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        verbose=False
    )

    print("\nEkpaideysi model...")
    model.fit(X_train, y_train)

    # Ypologismos training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print("Ekpaideysi oloklirothi!")

    return model
