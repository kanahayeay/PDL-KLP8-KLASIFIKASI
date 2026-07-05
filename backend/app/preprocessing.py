import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from app import config

# ---- Load sekali saat modul di-import ----
kamus_alay = pd.read_csv(
    config.SLANG_DICT_PATH, encoding="latin1", header=None, names=["slang", "formal"]
)
slang_dict = dict(zip(kamus_alay["slang"], kamus_alay["formal"]))

stopword_df = pd.read_csv(config.STOPWORD_PATH, header=None, names=["stopword"])
stopwords_set = set(stopword_df["stopword"].astype(str))


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub("RT", " ", text)
    text = re.sub("USER", " ", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def case_folding(text: str) -> str:
    return text.lower()


def normalize_tokens(tokens: list[str]) -> list[str]:
    return [slang_dict.get(word, word) for word in tokens]


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [word for word in tokens if word not in stopwords_set]


def preprocess(text: str) -> list[str]:
    text = clean_text(text)
    text = case_folding(text)
    tokens = word_tokenize(text)
    tokens = normalize_tokens(tokens)
    tokens = remove_stopwords(tokens)
    return tokens