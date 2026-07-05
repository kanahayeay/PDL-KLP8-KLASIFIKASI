import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model")
WORD_TO_ID_PATH = os.path.join(ARTIFACTS_DIR, "word_to_id.pkl")
EMBEDDING_PATH = os.path.join(ARTIFACTS_DIR, "glove_embeddings.npy")
LSTM_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "lstm_config.json")
MODEL_WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "best_lstm_model.pth")
SLANG_DICT_PATH = os.path.join(ARTIFACTS_DIR, "new_kamusalay.csv")
STOPWORD_PATH = os.path.join(ARTIFACTS_DIR, "stopwordbahasa.csv")
CORS_ALLOWED_ORIGINS = ["http://localhost:5173"]  