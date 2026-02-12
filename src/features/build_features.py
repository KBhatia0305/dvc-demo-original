import yaml 
import numpy as np
import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer

# -------------------- logging config --------------------
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)


def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        logger.error(f"Failed to load data from {path}", exc_info=True)
        raise

def load_params(path: str = "params.yaml") -> dict:
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        max_features= params['feature_engineering']['max_features']
        logger.debug('max features retrieved')
        return max_features
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise
    except Exception as e:
        logger.error('some error occured')
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
        logger.info("Feature data saved successfully")
    except Exception:
        logger.error("Failed to save feature data", exc_info=True)
        raise

#  feature engineering 
def build_bow_features(train_df: pd.DataFrame, test_df: pd.DataFrame, max_features: int):
    try:
        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df.fillna("", inplace=True)
        test_df.fillna("", inplace=True)

        X_train = train_df["content"].values
        y_train = train_df["sentiment"].values

        X_test = test_df["content"].values
        y_test = test_df["sentiment"].values

        vectorizer = CountVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_features = pd.DataFrame(X_train_bow.toarray())
        train_features["label"] = y_train

        test_features = pd.DataFrame(X_test_bow.toarray())
        test_features["label"] = y_test

        return train_features, test_features

    except Exception:
        logger.error("Error during feature engineering", exc_info=True)
        raise

def main():
    try:
        logger.info("Starting feature engineering stage")

        max_features= load_params(path='params.yaml')

        train_data = load_data("data/interim/train_interim.csv")
        test_data = load_data("data/interim/test_interim.csv")

        train_features, test_features = build_bow_features(train_data, test_data,max_features)

        save_data(
            train_features,
            test_features,
            output_dir=os.path.join("data", "processed"),
        )

        logger.info("Feature engineering completed successfully")

    except Exception:
        logger.error("Feature engineering failed", exc_info=True)

if __name__ == "__main__":
    main()
