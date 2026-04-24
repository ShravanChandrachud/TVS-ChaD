"""
Logistic Regression Classifier
================================
Loads cached CLIP embeddings and trains a multinomial Logistic Regression
classifier using scikit-learn. Evaluates on both test and validation sets,
runs stratified k-fold cross-validation on train+val combined, and generates
a confusion matrix heatmap, per-class F1 bar chart, and t-SNE visualization
with decision boundaries.

The model learns 12 weight vectors (one per class, each 512-dim) and uses
softmax to produce class probabilities. Training minimizes cross-entropy
loss with L2 regularization controlled by the C hyperparameter.

Hyperparameters are defined as constants at the top of the file.
The trained model is saved to models/logreg_model.pkl (includes the
fitted scaler and label encoder for standalone inference).

Input:  data/embeddings/clip_vit_b32/embeddings_*.npy
Output: outputs/logreg_<timestamp>/  (results JSON, confusion matrix,
        per-class F1 chart, t-SNE plot)
        models/logreg_model.pkl
"""

import json
import joblib
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression as LogReg2D

from utils import (
    OUTPUTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    prepare_data,
    evaluate_classifier,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_tsne,
    save_results,
)


C = 1.0 
SOLVER = "lbfgs" 
MAX_ITER = 5000 
CV_FOLDS = 5
EMBEDDING_MODEL = "clip_vit_b32"


def main():
    print("=" * 60)
    print("Logistic Regression — CLIP Embeddings")
    print("=" * 60)
    print(f"   C          = {C}")
    print(f"   solver     = {SOLVER}")
    print(f"   max_iter   = {MAX_ITER}")
    print(f"   cv_folds   = {CV_FOLDS}")

    data = prepare_data(EMBEDDING_MODEL)
    label_names = data["label_names"]

    print(f"\nTraining Logistic Regression")
    clf = LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        solver=SOLVER,
        multi_class="multinomial",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(data["X_train"], data["y_train"])

    test_results = evaluate_classifier(
        clf,
        data["X_test"],
        data["y_test"],
        label_names,
        "Logistic Regression",
        split_name="test",
    )

    val_results = evaluate_classifier(
        clf,
        data["X_val"],
        data["y_val"],
        label_names,
        "Logistic Regression",
        split_name="val",
    )

    print(f"\nRunning {CV_FOLDS}-fold cross-validation (train+val)")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        LogisticRegression(
            C=C,
            max_iter=MAX_ITER,
            solver=SOLVER,
            multi_class="multinomial",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        data["X_trainval"],
        data["y_trainval"],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Per-fold:    {cv_scores}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUTS_DIR / f"logreg_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "classifier": "Logistic Regression",
        "embedding_model": EMBEDDING_MODEL,
        "hyperparameters": {
            "C": C,
            "solver": SOLVER,
            "max_iter": MAX_ITER,
            "cv_folds": CV_FOLDS,
        },
        "test": {
            "accuracy": test_results["accuracy"],
            "f1_macro": test_results["f1_macro"],
            "f1_weighted": test_results["f1_weighted"],
            "classification_report": test_results["classification_report"],
            "confusion_matrix": test_results["confusion_matrix"],
        },
        "val": {
            "accuracy": val_results["accuracy"],
            "f1_macro": val_results["f1_macro"],
            "f1_weighted": val_results["f1_weighted"],
        },
        "cross_validation": {
            "mean_accuracy": float(cv_scores.mean()),
            "std_accuracy": float(cv_scores.std()),
            "fold_scores": cv_scores.tolist(),
        },
    }
    save_results(combined, run_dir, "logreg")

    cm = np.array(test_results["confusion_matrix"])
    plot_confusion_matrix(
        cm,
        label_names,
        title=f"Logistic Regression [Test] (C={C}, solver={SOLVER})\n"
        f"Acc: {test_results['accuracy']:.3f} | F1: {test_results['f1_macro']:.3f}",
        save_path=run_dir / "confusion_matrix_test.png",
    )
    plot_per_class_f1(test_results, label_names, run_dir / "per_class_f1_test.png")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "logreg_model.pkl"
    joblib.dump(
        {
            "classifier": clf,
            "scaler": data["scaler"],
            "label_encoder": data["label_encoder"],
            "label_names": label_names,
            "hyperparameters": {"C": C, "solver": SOLVER, "max_iter": MAX_ITER},
        },
        model_path,
    )
    print(f"   Model saved: {model_path}")

    print("\nGenerating t-SNE visualization")
    from sklearn.manifold import TSNE as TSNE_local

    tsne = TSNE_local(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_test_2d = tsne.fit_transform(data["X_test"])

    logreg_2d = LogisticRegression(
        C=C,
        solver=SOLVER,
        max_iter=MAX_ITER,
        multi_class="multinomial",
        random_state=RANDOM_STATE,
    )
    logreg_2d.fit(X_test_2d, data["y_test"])

    plot_tsne(
        data["X_test"],
        data["y_test"],
        clf.predict(data["X_test"]),
        label_names,
        f"Logistic Regression (C={C})\nAcc: {test_results['accuracy']:.3f} | F1: {test_results['f1_macro']:.3f}",
        run_dir / "tsne_logreg.png",
        clf_2d=logreg_2d,
        X_2d=X_test_2d,
    )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — Logistic Regression")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'Test':>10} {'Val':>10}")
    print("-" * 47)
    print(
        f"{'Accuracy':<25} {test_results['accuracy']:>10.4f} {val_results['accuracy']:>10.4f}"
    )
    print(
        f"{'F1 (macro)':<25} {test_results['f1_macro']:>10.4f} {val_results['f1_macro']:>10.4f}"
    )
    print(
        f"{'F1 (weighted)':<25} {test_results['f1_weighted']:>10.4f} {val_results['f1_weighted']:>10.4f}"
    )
    print(f"\nCV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
