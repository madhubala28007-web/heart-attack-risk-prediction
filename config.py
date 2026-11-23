"""Global configuration for Heart Attack Risk Prediction using Retinal Images."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "retina")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DB_PATH = os.path.join(BASE_DIR, "heart_risk.db")
FEATURE_CSV = os.path.join(OUTPUT_DIR, "features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")

CLASS_NAMES = ["low_risk", "high_risk"]   # label 0 = low risk, label 1 = high risk
IMG_SIZE = 320                            # every fundus image is resized to 320 x 320
RANDOM_STATE = 42
TEST_SIZE = 0.20

RISK_LOW = 0.35    # probability below this -> LOW risk
RISK_HIGH = 0.65   # probability above this -> HIGH risk (in between -> MODERATE)

for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)
