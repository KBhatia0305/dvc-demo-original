import logging
import pickle
from typing import Tuple
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


# ---------- logging config ----------
logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)


#  data loading 
def load_training_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        train_data = pd.read_csv(path)

        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values

        logger.info("Training data loaded successfully")
        return X_train, y_train

    except Exception:
        logger.error("Failed to load training data", exc_info=True)
        raise

def load_params(path: str = "params.yaml") -> dict:
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        n_estimators= params['model_building']['n_estimators']
        learning_rate= params['model_building']['learning_rate']
        logger.debug('n_estimators and learning_rate retrieved')
        return n_estimators,learning_rate
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise
    except Exception as e:
        logger.error('some error occured')
        raise



# model training 
def train_model(X_train: np.ndarray,y_train: np.ndarray, n_estimators, learning_rate) -> GradientBoostingClassifier:
    try:
        clf = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
        clf.fit(X_train, y_train)

        logger.info("Model training completed successfully")
        return clf

    except Exception:
        logger.error("Model training failed", exc_info=True)
        raise


# model saving 
def save_model(model: GradientBoostingClassifier, path: str) -> None:
    try:
        pickle.dump(model, open(path, "wb"))
        logger.info("Model saved successfully at %s", path)

    except Exception:
        logger.error("Failed to save model", exc_info=True)
        raise


# main pipeline 
def main() -> None:
    try:
        logger.info("Starting model training stage")

        n_estimators,learning_rate= load_params(path='params.yaml')

        X_train, y_train = load_training_data("./data/processed/train_processed.csv")

        model = train_model(X_train, y_train, n_estimators, learning_rate)

        save_model(model, "models/model.pkl")

        logger.info("Model training completed successfully")

    except Exception:
        logger.error("Model training pipeline failed", exc_info=True)


if __name__ == "__main__":
    main()
