"""Predict heart-attack risk for one retinal image (command line)."""
import os
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import load
from config import MODEL_PATH, OUTPUT_DIR, RISK_LOW, RISK_HIGH
from preprocess import run_pipeline
from features import extract_features_from_result
import database

warnings.filterwarnings("ignore")


def risk_category(prob):
    if prob < RISK_LOW:
        return "LOW RISK"
    if prob > RISK_HIGH:
        return "HIGH RISK"
    return "MODERATE RISK"


def predict_image(path, bundle=None):
    """Returns (probability of high risk, category, preprocessing results)."""
    bundle = bundle or load(MODEL_PATH)
    res = run_pipeline(path)
    feats = extract_features_from_result(res).reshape(1, -1)
    prob = float(bundle["model"].predict_proba(feats)[0, 1])
    return prob, risk_category(prob), res


def save_report(path, prob, label, res, out_path):
    """One-page picture : original, vessels, skeleton and the final decision."""
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.8))
    ax[0].imshow(res["image"][:, :, ::-1]); ax[0].set_title("Retinal image")
    ax[1].imshow(res["vessels"], cmap="gray"); ax[1].set_title("Segmented vessels")
    ax[2].imshow(res["skeleton"], cmap="gray"); ax[2].set_title("Vessel skeleton")
    for a in ax:
        a.axis("off")
    colour = {"LOW RISK": "green", "MODERATE RISK": "darkorange", "HIGH RISK": "red"}[label]
    fig.suptitle(f"{os.path.basename(path)}   |   Risk probability = {prob * 100:.1f}%   |   {label}",
                 color=colour, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path of the retinal image")
    ap.add_argument("--name", default="Demo Patient")
    ap.add_argument("--age", type=int, default=45)
    ap.add_argument("--gender", default="Male")
    args = ap.parse_args()

    bundle = load(MODEL_PATH)
    prob, label, res = predict_image(args.image, bundle)
    out = os.path.join(OUTPUT_DIR, "report_" + os.path.basename(args.image).split(".")[0] + ".png")
    save_report(args.image, prob, label, res, out)

    database.init_db()
    pid = database.add_patient(args.name, args.age, args.gender)
    database.add_prediction(pid, args.image, prob, label, bundle["name"])

    print("=" * 52)
    print(" HEART ATTACK RISK PREDICTION - RETINAL IMAGE")
    print("=" * 52)
    print(f" Patient       : {args.name}  ({args.age} yrs, {args.gender})")
    print(f" Image         : {args.image}")
    print(f" Model used    : {bundle['name']}")
    print(f" Risk score    : {prob * 100:.1f} %")
    print(f" Risk category : {label}")
    print(f" Report saved  : {out}")
    print("=" * 52)
    print(" Note: screening aid only - not a medical diagnosis.")


if __name__ == "__main__":
    main()
     