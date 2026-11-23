"""Train and evaluate the machine-learning models on retinal image features."""
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, dump
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from config import (DATA_DIR, CLASS_NAMES, OUTPUT_DIR, FEATURE_CSV, MODEL_PATH,
                    RANDOM_STATE, TEST_SIZE)
from features import extract_features, FEATURE_NAMES

warnings.filterwarnings("ignore")


def list_images():
    paths, labels = [], []
    for label, cls in enumerate(CLASS_NAMES):
        files = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.*")))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        paths += files
        labels += [label] * len(files)
    if not paths:
        raise SystemExit("No images found. Run:  python generate_dataset.py")
    return paths, np.array(labels)


def build_feature_table():
    paths, labels = list_images()
    print(f"[INFO] Extracting features from {len(paths)} images ...")
    rows = Parallel(n_jobs=-1)(delayed(extract_features)(p) for p in paths)
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    df["label"] = labels
    df["path"] = paths
    df.to_csv(FEATURE_CSV, index=False)
    return df


def get_models():
    rs = RANDOM_STATE
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "SVM (RBF)": make_pipeline(StandardScaler(),
                                   SVC(kernel="rbf", C=2.0, probability=True, random_state=rs)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
    }


def main():
    df = build_feature_table()
    X, y = df[FEATURE_NAMES].values, df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"[INFO] Train = {len(y_tr)}   Test = {len(y_te)}   Features = {X.shape[1]}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted, probs = [], {}, {}
    for name, model in get_models().items():
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="roc_auc").mean()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)[:, 1]
        results.append({
            "Model": name, "CV AUC": cv_auc,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred),
            "F1-score": f1_score(y_te, pred),
            "ROC-AUC": roc_auc_score(y_te, prob)})
        fitted[name], probs[name] = model, prob
        print(f"  {name:20s} acc={results[-1]['Accuracy']:.3f}   auc={results[-1]['ROC-AUC']:.3f}")

    res = pd.DataFrame(results).round(3)
    res.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    best_name = res.sort_values("CV AUC", ascending=False).iloc[0]["Model"]
    best = fitted[best_name]
    dump({"model": best, "name": best_name, "features": FEATURE_NAMES}, MODEL_PATH)
    print(f"\n[RESULT] Best model : {best_name}   (saved to {MODEL_PATH})")
    print(res.to_string(index=False))

    y_pred = best.predict(X_te)
    report = classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=3)
    open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w").write(report)
    print("\nClassification report (test set):\n" + report)

    # ---- graphs ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred),
                           display_labels=["Low risk", "High risk"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for name, pr in probs.items():
        fpr, tpr, _ = roc_curve(y_te, pr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, pr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves"); ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    idx = np.arange(len(res))
    ax.bar(idx - 0.2, res["Accuracy"], 0.4, label="Accuracy")
    ax.bar(idx + 0.2, res["ROC-AUC"], 0.4, label="ROC-AUC")
    ax.set_xticks(idx); ax.set_xticklabels(res["Model"], rotation=20, fontsize=7)
    ax.set_ylim(0.5, 1.0); ax.legend(fontsize=7); ax.set_title("Model comparison")
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=200)
    plt.close(fig)

    rf = fitted["Random Forest"]
    order = np.argsort(rf.feature_importances_)[::-1][:12]
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.barh([FEATURE_NAMES[i] for i in order][::-1], rf.feature_importances_[order][::-1])
    ax.set_title("Top 12 feature importances (Random Forest)"); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=200)
    plt.close(fig)
    print("[INFO] Graphs saved in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
