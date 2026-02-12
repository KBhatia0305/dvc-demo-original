import json
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_score,recall_score,roc_auc_score,)

#  logging config 
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)


# -------------------- model loading --------------------
def load_model(path: str):
    try:
        clf = pickle.load(open(path, "rb"))
        logger.info("Model loaded successfully from %s", path)
        return clf
    except Exception:
        logger.error("Failed to load model", exc_info=True)
        raise


# -------------------- data loading --------------------
def load_test_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info("Test data loaded successfully from %s", path)
        return df
    except Exception:
        logger.error("Failed to load test data", exc_info=True)
        raise


# -------------------- evaluation --------------------
def evaluate_model(clf, test_data: pd.DataFrame) -> dict:
    try:
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        logger.info("Model evaluation completed successfully")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }

    except Exception:
        logger.error("Error during model evaluation", exc_info=True)
        raise


# -------------------- save metrics --------------------
def save_metrics(metrics: dict, path: str) -> None:
    try:
        with open(path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.info("Metrics saved successfully to %s", path)
    except Exception:
        logger.error("Failed to save metrics", exc_info=True)
        raise


# -------------------- main --------------------
def main() -> None:
    try:
        logger.info("Starting model evaluation stage")

        clf = load_model("models/model.pkl")
        test_data = load_test_data("data/processed/test_processed.csv")

        metrics = evaluate_model(clf, test_data)
        save_metrics(metrics, "reports/metrics.json")

        logger.info("Model evaluation completed successfully")

    except Exception:
        logger.error("Model evaluation pipeline failed", exc_info=True)


if __name__ == "__main__":
    main()
