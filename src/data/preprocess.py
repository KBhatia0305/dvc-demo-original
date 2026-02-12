import numpy as np
import pandas as pd
import os
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------- logging config --------------------
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)

# -------------------- text utils --------------------
def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

def removing_numbers(text: str) -> str:
    return "".join([char for char in text if not char.isdigit()])

def lower_case(text: str) -> str:
    return text.lower()

def removing_punctuations(text: str) -> str:
    text = re.sub(
        '[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""),
        " ",
        text,
    )
    text = re.sub("\s+", " ", text)
    return text.strip()

def removing_urls(text: str) -> str:
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)

# -------------------- preprocessing --------------------
def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df["content"] = df["content"].apply(lower_case)
        df["content"] = df["content"].apply(remove_stop_words)
        df["content"] = df["content"].apply(removing_numbers)
        df["content"] = df["content"].apply(removing_punctuations)
        df["content"] = df["content"].apply(removing_urls)
        df["content"] = df["content"].apply(lemmatization)
        return df
    except Exception as e:
        logger.error("Error during text normalization", exc_info=True)
        raise

# -------------------- io helpers --------------------
def load_data(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to load data from {path}", exc_info=True)
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train_interim.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_interim.csv"), index=False)
        logger.info("Processed data saved successfully")
    except Exception as e:
        logger.error("Failed to save processed data", exc_info=True)
        raise

# -------------------- main --------------------
def main():
    try:
        logger.info("Starting data preprocessing stage")

        nltk.download("wordnet")
        nltk.download("stopwords")

        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")

        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)

        save_data(
            train_processed,
            test_processed,
            output_dir=os.path.join("data", "interim"),
        )

        logger.info("Data preprocessing completed successfully")

    except Exception as e:
        logger.error("Data preprocessing failed", exc_info=True)

if __name__ == "__main__":
    main()
