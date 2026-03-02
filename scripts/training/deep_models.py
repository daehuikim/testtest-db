#!/usr/bin/env python3
"""
LSTM, Transformer 등 Deep 모델 (PyTorch).
lag 컬럼을 시퀀스로 사용.
"""

import logging
import re
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LAG_PATTERN = re.compile(r"^(.+)_lag(\d+)$")


def _get_lag_sequence_cols(features: List[str], base: str = "price_per_kg_mean", max_lag: int = 60) -> List[str]:
    """base의 lag1..lagN 컬럼을 lag 순으로 정렬."""
    cols = []
    for f in features:
        m = LAG_PATTERN.match(f)
        if m and m.group(1) == base and int(m.group(2)) <= max_lag:
            cols.append((int(m.group(2)), f))
    cols.sort(key=lambda x: x[0])
    return [c[1] for c in cols]


def build_sequence_array(df: pd.DataFrame, seq_cols: List[str], fill_val: float = 0.0) -> np.ndarray:
    """(n_samples, seq_len, 1) 배열 생성."""
    arr = df[seq_cols].fillna(fill_val).values
    return arr[:, :, np.newaxis]  # (n, seq_len, 1)


def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class LSTMWrapper:
    """LSTM Regressor (lag 시퀀스 입력)."""

    def __init__(self, seq_len: int = 30, hidden: int = 64, layers: int = 2, epochs: int = 50, lr: float = 0.001):
        self.seq_len = seq_len
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.seq_cols = None

    def fit(self, X_seq: np.ndarray, y: np.ndarray):
        torch = _try_import_torch()
        if torch is None:
            raise ImportError("PyTorch 필요: pip install torch")

        import torch.nn as nn

        class LSTMReg(nn.Module):
            def __init__(self, seq_len, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.model = LSTMReg(self.seq_len, self.hidden, self.layers)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        X_t = torch.FloatTensor(X_seq)
        y_t = torch.FloatTensor(y)

        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.model(X_t)
            loss = nn.MSELoss()(pred, y_t)
            loss.backward()
            opt.step()
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        torch = _try_import_torch()
        if self.model is None:
            raise RuntimeError("fit 먼저 호출")
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X_seq))
        return out.numpy()


class TransformerWrapper:
    """Simple Temporal Transformer (lag 시퀀스 입력)."""

    def __init__(self, seq_len: int = 30, d_model: int = 32, nhead: int = 4, num_layers: int = 2, epochs: int = 50, lr: float = 0.001):
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def fit(self, X_seq: np.ndarray, y: np.ndarray):
        torch = _try_import_torch()
        if torch is None:
            raise ImportError("PyTorch 필요: pip install torch")

        import torch.nn as nn

        class PosEnc(nn.Module):
            def __init__(self, seq_len, d_model):
                super().__init__()
                pe = torch.zeros(seq_len, d_model)
                for i in range(seq_len):
                    for j in range(0, d_model, 2):
                        pe[i, j] = np.sin(i / 10000 ** (j / d_model))
                        if j + 1 < d_model:
                            pe[i, j + 1] = np.cos(i / 10000 ** (j / d_model))
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, : x.size(1)]

        class TransReg(nn.Module):
            def __init__(self, seq_len, d_model, nhead, num_layers):
                super().__init__()
                self.proj = nn.Linear(1, d_model)
                self.pos = PosEnc(seq_len, d_model)
                enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, batch_first=True)
                self.trans = nn.TransformerEncoder(enc, num_layers)
                self.fc = nn.Linear(d_model, 1)

            def forward(self, x):
                x = self.proj(x)
                x = self.pos(x)
                x = self.trans(x)
                return self.fc(x[:, -1, :]).squeeze(-1)

        self.model = TransReg(self.seq_len, self.d_model, self.nhead, self.num_layers)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        X_t = torch.FloatTensor(X_seq)
        y_t = torch.FloatTensor(y)

        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.model(X_t)
            loss = nn.MSELoss()(pred, y_t)
            loss.backward()
            opt.step()
        return self

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        torch = _try_import_torch()
        if self.model is None:
            raise RuntimeError("fit 먼저 호출")
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X_seq))
        return out.numpy()
