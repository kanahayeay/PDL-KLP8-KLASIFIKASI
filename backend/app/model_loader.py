import json
import pickle

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# LSTM Classifier
# Arsitektur HARUS identik dengan yang dipakai saat training
# (lihat notebook, cell "LSTM Classifier")
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


class ModelBundle:
    """
    Membungkus model + artifact yang dibutuhkan untuk inference,
    supaya cukup di-load sekali saat startup FastAPI.
    """

    def __init__(self, artifacts_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Vocabulary ----
        with open(f"{artifacts_dir}/word_to_id.pkl", "rb") as f:
            self.word_to_id = pickle.load(f)

        # ---- Embedding matrix (GloVe hasil training) ----
        embedding_matrix = np.load(f"{artifacts_dir}/glove_embeddings.npy")

        # ---- Config model ----
        with open(f"{artifacts_dir}/lstm_config.json") as f:
            config = json.load(f)

        self.max_length = config["MAX_LENGTH"]

        # ---- Bangun ulang arsitektur, lalu isi bobotnya ----
        self.model = LSTMClassifier(
            embedding_matrix=embedding_matrix,
            hidden_dim=config["HIDDEN_DIM"],
            num_layers=config["NUM_LAYERS"],
            output_dim=config["OUTPUT_DIM"],
            dropout=config["DROPOUT"],
        ).to(self.device)

        state_dict = torch.load(
            f"{artifacts_dir}/best_lstm_model.pth",
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()  # penting: matikan dropout saat inference

    def text_to_sequence(self, tokens: list[str]) -> list[int]:
        """Ubah list token (hasil preprocessing) jadi list index angka."""
        unk_id = self.word_to_id["<UNK>"]
        return [self.word_to_id.get(word, unk_id) for word in tokens]

    def pad_sequence(self, sequence: list[int]) -> list[int]:
        """Samakan panjang sequence dengan MAX_LENGTH (pad/truncate)."""
        if len(sequence) < self.max_length:
            return sequence + [0] * (self.max_length - len(sequence))
        return sequence[: self.max_length]

    def predict(self, tokens: list[str]) -> dict:
        """
        tokens: hasil akhir preprocessing (list kata, SUDAH melalui
        cleaning, case folding, normalisasi slang, stopword removal).
        """
        sequence = self.text_to_sequence(tokens)
        padded = self.pad_sequence(sequence)

        input_tensor = torch.LongTensor([padded]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        hs_prob, abusive_prob = float(probs[0]), float(probs[1])

        return {
            "hate_speech": {
                "label": int(hs_prob >= 0.5),
                "probability": hs_prob,
            },
            "abusive": {
                "label": int(abusive_prob >= 0.5),
                "probability": abusive_prob,
            },
        }


# ============================================================
# Instance global — di-load sekali saat FastAPI startup,
# dipakai berulang kali oleh endpoint /predict
# ============================================================
model_bundle: ModelBundle | None = None


def load_model_bundle(artifacts_dir: str = "model_artifacts") -> ModelBundle:
    global model_bundle
    model_bundle = ModelBundle(artifacts_dir)
    return model_bundle