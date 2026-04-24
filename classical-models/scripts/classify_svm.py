"""
SVM Classifier (Linear + RBF Kernels)
=======================================
Loads cached CLIP embeddings and trains Support Vector Machine classifiers
with both linear and RBF kernels. Evaluates on test and validation sets,
runs stratified k-fold cross-validation, and generates confusion matrices,
per-class F1 charts, and t-SNE visualizations with decision boundaries.

Linear SVM finds maximum-margin hyperplanes in the 512-dim embedding space.
RBF SVM uses the kernel trick to implicitly map data to a higher-dimensional
space where curved decision boundaries become possible, controlled by the
gamma parameter.

t-SNE is computed once and reused for both kernel plots to ensure
consistent 2D projections across visualizations.

Hyperparameters are defined as constants at the top of the file.
Each kernel's trained model is saved separately to models/.

Input:  data/embeddings/clip_vit_b32/embeddings_*.npy
Output: outputs/svm_<timestamp>/  (per-kernel results JSON, confusion
        matrices, F1 charts, t-SNE plots, comparison JSON)
        models/svm_linear_model.pkl
        models/svm_rbf_model.pkl
"""

import json
import joblib
import numpy as np
from datetime import datetime

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import (
    OUTPUTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    prepare_data,
    evaluate_classifier,
    plot_confusion_matrix,
    plot_per_class_f1,
    save_results,
    plot_tsne,
)

KERNELS = "both" 

C_LINEAR = 1.0  
C_RBF = 10.0  
GAMMA = "scale"  

CV_FOLDS = 5 
EMBEDDING_MODEL = "clip_vit_b32"

def main():
    configs = {}
    if KERNELS in ("linear", "both"):
        configs["SVM (Linear)"] = {"kernel": "linear", "C": C_LINEAR, "gamma": "scale"}
    if KERNELS in ("rbf", "both"):
        configs["SVM (RBF)"] = {"kernel": "rbf", "C": C_RBF, "gamma": GAMMA}

    print("=" * 60)
    print("SVM Classification — CLIP Embeddings")
    print("=" * 60)
    for name, cfg in configs.items():
        print(f"   {name}: C={cfg['C']}, gamma={cfg['gamma']}")
    print(f"   cv_folds = {CV_FOLDS}")

    print("\nLoading embeddings")
    data = prepare_data(EMBEDDING_MODEL)
    label_names = data["label_names"]
    print("Data loaded.\n")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUTS_DIR / f"svm_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\nComputing t-SNE for visualization")
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_test_2d = tsne.fit_transform(data["X_test"])

    all_results = {}

    for name, cfg in configs.items():
        print(f"{'=' * 60}")
        print(f"Training: {name} (C={cfg['C']}, gamma={cfg['gamma']})")
        print(f"{'=' * 60}")

        print("   Fitting model (this may take a few minutes)")
        clf = SVC(
            kernel=cfg["kernel"],
            C=cfg["C"],
            gamma=cfg["gamma"],
            decision_function_shape="ovr",
            random_state=RANDOM_STATE,
        )
        clf.fit(data["X_train"], data["y_train"])
        print(" Model trained.")

        print("   Evaluating on test set")
        test_results = evaluate_classifier(
            clf,
            data["X_test"],
            data["y_test"],
            label_names,
            name,
            split_name="test",
        )
        print(f" Test done. Accuracy: {test_results['accuracy']:.4f}")

        print("   Evaluating on val set")
        val_results = evaluate_classifier(
            clf,
            data["X_val"],
            data["y_val"],
            label_names,
            name,
            split_name="val",
        )
        print(f" Val done. Accuracy: {val_results['accuracy']:.4f}")

        # Cross-validation
        print(f"   Running {CV_FOLDS}-fold CV")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            SVC(
                kernel=cfg["kernel"],
                C=cfg["C"],
                gamma=cfg["gamma"],
                decision_function_shape="ovr",
                random_state=RANDOM_STATE,
            ),
            data["X_trainval"],
            data["y_trainval"],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        print(f" CV done. Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Save per-kernel
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

        kernel_results = {
            "classifier": name,
            "embedding_model": EMBEDDING_MODEL,
            "hyperparameters": cfg,
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

        print("   Saving results and plots")
        save_results(kernel_results, run_dir, safe_name)

        cm = np.array(test_results["confusion_matrix"])
        plot_confusion_matrix(
            cm,
            label_names,
            title=f"{name} [Test] (C={cfg['C']}, gamma={cfg['gamma']})\n"
            f"Acc: {test_results['accuracy']:.3f} | F1: {test_results['f1_macro']:.3f}",
            save_path=run_dir / f"{safe_name}_confusion_matrix_test.png",
        )
        plot_per_class_f1(
            test_results, label_names, run_dir / f"{safe_name}_per_class_f1_test.png"
        )

        print(f"   Generating t-SNE for {name}")
        svm_2d = SVC(
            kernel=cfg["kernel"],
            C=cfg["C"],
            gamma=cfg["gamma"],
            random_state=RANDOM_STATE,
        )
        svm_2d.fit(X_test_2d, data["y_test"])

        plot_tsne(
            data["X_test"],
            data["y_test"],
            clf.predict(data["X_test"]),
            label_names,
            f"{name} (C={cfg['C']}, gamma={cfg['gamma']})\nAcc: {test_results['accuracy']:.3f} | F1: {test_results['f1_macro']:.3f}",
            run_dir / f"{safe_name}_tsne.png",
            clf_2d=svm_2d,
            X_2d=X_test_2d,
        )

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{safe_name}_model.pkl"
        joblib.dump(
            {
                "classifier": clf,
                "scaler": data["scaler"],
                "label_encoder": data["label_encoder"],
                "label_names": label_names,
                "hyperparameters": cfg,
            },
            model_path,
        )
        print(f"   Model saved: {model_path}")

        print(f" {name} complete.\n")

        all_results[name] = {
            "test": test_results,
            "val": val_results,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "params": cfg,
        }

    if len(all_results) > 1:
        print(f"{'=' * 60}")
        print("COMPARISON")
        print(f"{'=' * 60}")
        print(
            f"{'Kernel':<20} {'Test Acc':>10} {'Val Acc':>10} {'Test F1':>10} {'CV Acc':>10}"
        )
        print("-" * 64)
        for name, r in all_results.items():
            print(
                f"{name:<20} "
                f"{r['test']['accuracy']:>10.4f} "
                f"{r['val']['accuracy']:>10.4f} "
                f"{r['test']['f1_macro']:>10.4f} "
                f"{r['cv_mean']:>10.4f}"
            )

    # Save comparison
    comparison = {
        "timestamp": timestamp,
        "embedding_model": EMBEDDING_MODEL,
        "kernels": {
            name: {
                "test_accuracy": r["test"]["accuracy"],
                "val_accuracy": r["val"]["accuracy"],
                "test_f1_macro": r["test"]["f1_macro"],
                "cv_mean": r["cv_mean"],
                "cv_std": r["cv_std"],
                "hyperparameters": r["params"],
            }
            for name, r in all_results.items()
        },
    }
    with open(run_dir / "svm_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"\n{'!' * 60}")
        print(f"ERROR: {e}")
        print(f"{'!' * 60}")
        traceback.print_exc()
