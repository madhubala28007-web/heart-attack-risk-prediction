"""Simple test cases for the project (run:  python run_tests.py)."""
import os, sqlite3, tempfile, warnings
import numpy as np, cv2
warnings.filterwarnings("ignore")
from joblib import load
from config import MODEL_PATH, DATA_DIR, FEATURE_CSV
from preprocess import load_image, run_pipeline
from features import extract_features, FEATURE_NAMES
from predict import predict_image, risk_category
import database

results = []


def case(no, desc, fn):
    try:
        out = fn()
        results.append((no, desc, "PASS", out))
    except Exception as e:
        results.append((no, desc, "FAIL", repr(e)[:60]))


hi = os.path.join(DATA_DIR, "high_risk", "high_risk_0003.jpg")
lo = os.path.join(DATA_DIR, "low_risk", "low_risk_0005.jpg")


def t1():
    img = load_image(hi); assert img.shape == (320, 320, 3); return "shape (320, 320, 3)"


def t2():
    r = run_pipeline(hi)
    assert r["vessels"].sum() > 1000
    return f"{int(r['vessels'].sum())} vessel pixels"


def t3():
    f = extract_features(hi)
    assert len(f) == len(FEATURE_NAMES) and np.isfinite(f).all()
    return f"{len(f)} finite features"


def t4():
    p, l, _ = predict_image(hi); assert l == "HIGH RISK"; return f"{p*100:.1f}% -> {l}"


def t5():
    p, l, _ = predict_image(lo); assert l == "LOW RISK"; return f"{p*100:.1f}% -> {l}"


def t6():
    got = [risk_category(x) for x in (0.2, 0.5, 0.8)]
    assert got == ["LOW RISK", "MODERATE RISK", "HIGH RISK"]
    return "3 thresholds correct"


def t7():
    try:
        load_image("no_such_file.jpg")
    except FileNotFoundError:
        return "FileNotFoundError raised"
    raise AssertionError("no error")


def t8():
    tmp = os.path.join(tempfile.gettempdir(), "bad.jpg"); open(tmp, "w").write("not an image")
    try:
        load_image(tmp)
    except FileNotFoundError:
        return "invalid file rejected"
    raise AssertionError


def t9():
    gray = cv2.cvtColor(cv2.imread(hi), cv2.COLOR_BGR2GRAY)
    tmp = os.path.join(tempfile.gettempdir(), "gray.jpg"); cv2.imwrite(tmp, gray)
    p, l, _ = predict_image(tmp); return f"grayscale accepted, {l}"


def t10():
    database.init_db(); pid = database.add_patient("Test User", 40, "Female")
    rid = database.add_prediction(pid, hi, 0.5, "MODERATE RISK")
    assert rid > 0
    return f"record id {rid} saved"


def t11():
    try:
        database.add_patient("Bad Age", 200, "Male")
    except sqlite3.IntegrityError:
        return "age 200 rejected (CHECK)"
    raise AssertionError


def t12():
    try:
        database.add_prediction(99999, hi, 0.5, "LOW RISK")
    except sqlite3.IntegrityError:
        return "unknown patient rejected (FK)"
    raise AssertionError


for i, (d, f) in enumerate([
        ("Load and resize a valid fundus image", t1), ("Segment vessels from a valid image", t2),
        ("Extract the feature vector", t3), ("Predict a high-risk image", t4),
        ("Predict a low-risk image", t5), ("Risk category thresholds", t6),
        ("Image path does not exist", t7), ("File is not an image", t8),
        ("Grayscale image given as input", t9), ("Save patient + prediction in database", t10),
        ("Invalid age (200) inserted", t11), ("Prediction for unknown patient id", t12)], start=1):
    case(i, d, f)
for r in results:
    print(f"TC{r[0]:02d} | {r[1]:42s} | {r[2]} | {r[3]}")
print("PASSED", sum(r[2] == "PASS" for r in results), "of", len(results))
