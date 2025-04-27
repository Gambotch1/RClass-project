# src/rclass/experiment.py

import time
import csv
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from rclass.data import load_mnist_train_and_test
from rclass.features import preprocess_data, create_polynomial_features
from rclass.bisection import solve_for_class
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def _solve_for_class_helper(args):
    """
    Unpack a tuple of arguments and call solve_for_class.
    This function must be at module‐level to be picklable.
    """
    return solve_for_class(*args)

def run_experiment(
    N_samples: int,
    num_degree: int,
    den_degree: int,
    n_pca_components: int,
):
    """
    1. Subsample N_samples from MNIST training set
    2. Preprocess (scale + PCA)
    3. Generate numerator/denominator polynomial features
    4. Solve rational approx for each digit in parallel
    5. Classify full test set & compute metrics

    Returns: (runtime_sec, accuracy, macro_f1, class_report, conf_matrix)
    """
    # 1. Load
    X_train_full, y_train_full, X_test_full, y_test_full = load_mnist_train_and_test()

    # 2. Subsample
    idx = np.random.choice(len(X_train_full), N_samples, replace=False)
    X_train, y_train = X_train_full[idx], y_train_full[idx]

    # 3. Preprocess
    X_tr_pca, X_te_pca, scaler, pca = preprocess_data(
        X_train, X_test_full, n_components=n_pca_components
    )

    # 4. Poly features
    p_Phi_train, p_poly = create_polynomial_features(X_tr_pca, num_degree)
    q_Phi_train, q_poly = create_polynomial_features(X_tr_pca, den_degree)

    p_Phi_test = p_poly.transform(X_te_pca)
    q_Phi_test = q_poly.transform(X_te_pca)

    # 5. One‐vs‐rest label dict
    classes = list(range(10))
    f_vals = {c: np.array([1 if y==c else 0 for y in y_train]) for c in classes}

    # 6. Solve per class
    args = [
        (c, p_Phi_train, q_Phi_train, f_vals)
        for c in classes
    ]

    start = time.time()
    with ProcessPoolExecutor() as ex:
        results = ex.map(_solve_for_class_helper, args)
        models = {c: (p, q, z) for c, p, q, z in results}
    runtime = time.time() - start

    # 7. Classify test set
    y_pred = []
    for x in X_test_full:
        # preprocess + pca
        x_scaled = scaler.transform(x.reshape(1, -1))
        x_pca = pca.transform(x_scaled)[0]

        # compute scores
        scores = {}
        for c, m in models.items():
            p_coef, q_coef, _ = m
            if p_coef is None:
                scores[c] = -np.inf
            else:
                num = p_poly.transform(x_pca.reshape(1,-1)).dot(p_coef)[0]
                den = q_poly.transform(x_pca.reshape(1,-1)).dot(q_coef)[0]
                scores[c] = num/den if abs(den)>1e-12 else (-np.inf if num<0 else np.inf)
        y_pred.append(max(scores, key=scores.get))

    # 8. Metrics
    acc = accuracy_score(y_test_full, y_pred)
    f1  = f1_score(y_test_full, y_pred, average="macro")
    report = classification_report(y_test_full, y_pred)
    cm = confusion_matrix(y_test_full, y_pred)

    return runtime, acc, f1, report, cm


if __name__ == "__main__":
    # Example grid
    sample_sizes   = [1000]
    num_degrees    = [2]
    den_degrees    = [1]
    pca_components = [10]

    # CSV output
    fname = "results.csv"
    fields = ["N_samples","num_degree","den_degree","pca_comp","runtime","accuracy","macro_f1"]

    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

        for N, nd, dd, pc in [
            (N, nd, dd, pc)
            for N in sample_sizes
            for nd in num_degrees
            for dd in den_degrees
            for pc in pca_components
        ]:
            print(f"Running {N=} {nd=} {dd=} {pc=}")
            rt, ac, mf, _, _ = run_experiment(N, nd, dd, pc)
            writer.writerow([N, nd, dd, pc, f"{rt:.1f}", f"{ac:.4f}", f"{mf:.4f}"])
            print(f" → acc={ac:.3f}, f1={mf:.3f}, t={rt:.1f}s")
