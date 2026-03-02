#!/usr/bin/env python3
"""
LSTM, Transformer 등 Deep 모델 (PyTorch).
lag 컬럼을 시퀀스로 사용.
- eval during training, early stopping, checkpoint
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LAG_PATTERN = re.compile(r"^(.+)_lag(\d+)$")

# 체크포인트 저장 경로
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "temp" / "checkpoints"


def _get_lag_sequence_cols(features: List[str], base: str = "price_per_kg_mean", max_lag: int = 365) -> List[str]:
    """base의 lag1..lagN 컬럼을 lag 순으로 정렬. max_lag=365일 때 seasonal(364,365,366) 포함."""
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


def _mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE (평가용)."""
    return np.mean(np.abs(y_true - y_pred))


# Module-level LSTM for pickle support (must be top-level class)
try:
    import torch.nn as _nn
    class _LSTMRegModule(_nn.Module):
        def __init__(self, seq_len, hidden, layers, dropout=0.2):
            super().__init__()
            self.lstm = _nn.LSTM(1, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
            self.dropout = _nn.Dropout(dropout)
            self.fc = _nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            h = self.dropout(out[:, -1, :])
            return self.fc(h).squeeze(-1)
except ImportError:
    _LSTMRegModule = None


class LSTMWrapper:
    """LSTM Regressor (lag 시퀀스 입력). eval, early stopping, checkpoint 지원."""

    def __init__(
        self,
        seq_len: int = 30,
        hidden: int = 64,
        layers: int = 2,
        epochs: int = 50,
        lr: float = 0.001,
        patience: int = 10,
        dropout: float = 0.2,
        use_mape_loss: bool = False,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.seq_len = seq_len
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.dropout = dropout
        self.use_mape_loss = use_mape_loss
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        self.model = None
        self.seq_cols = None
        self.best_epoch = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def fit(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        torch = _try_import_torch()
        if torch is None:
            raise ImportError("PyTorch 필요: pip install torch")

        if _LSTMRegModule is None:
            raise ImportError("PyTorch 필요: pip install torch")
        self.model = _LSTMRegModule(self.seq_len, self.hidden, self.layers, self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        X_t = torch.FloatTensor(X_seq)
        y_train = np.log1p(y) if self.use_mape_loss else y
        y_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val) if X_val is not None else None
        y_val_np = y_val

        best_val_loss = float("inf")
        no_improve = 0
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            pred = self.model(X_t)
            # LSTM: MSE on log scale for stability when use_mape_loss (raw scale)
            loss = torch.nn.MSELoss()(pred, y_t)
            loss.backward()
            opt.step()
            train_loss = loss.item()
            self.train_losses.append(train_loss)

            val_loss = None
            if X_val_t is not None and y_val_np is not None:
                self.model.eval()
                with torch.no_grad():
                    pred_val = self.model(X_val_t).numpy()
                pred_raw = np.expm1(pred_val) if self.use_mape_loss else pred_val
                val_loss = _mae_loss(y_val_np, pred_raw)
                self.val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_epoch = epoch
                    no_improve = 0
                    self._save_checkpoint("lstm_best.pt", torch)
                else:
                    no_improve += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"  LSTM epoch {epoch+1}/{self.epochs} train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" val_mae={val_loss:.4f}"
                logger.info(msg)

            if self.patience > 0 and no_improve >= self.patience:
                logger.info("  LSTM early stop at epoch %d", epoch + 1)
                break

        if X_val_t is not None and (self.checkpoint_dir / "lstm_best.pt").exists():
            self._load_checkpoint("lstm_best.pt", torch)
        return self

    def fit_from_pretrained(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        pretrained: "LSTMWrapper",
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        patience: Optional[int] = None,
    ) -> "LSTMWrapper":
        """Pre-trained LSTM에서 가중치 로드 후 fine-tune (월별 데이터 등 소량 데이터용)."""
        torch = _try_import_torch()
        if torch is None:
            raise ImportError("PyTorch 필요: pip install torch")
        if _LSTMRegModule is None:
            raise ImportError("PyTorch 필요: pip install torch")
        if pretrained.model is None:
            raise ValueError("pretrained 모델이 fit되지 않음")

        self.model = _LSTMRegModule(self.seq_len, self.hidden, self.layers, self.dropout)
        self.model.load_state_dict(pretrained.model.state_dict().copy())

        _epochs = epochs if epochs is not None else 30
        _lr = lr if lr is not None else 0.0001
        _patience = patience if patience is not None else 10
        opt = torch.optim.Adam(self.model.parameters(), lr=_lr)
        X_t = torch.FloatTensor(X_seq)
        y_train = np.log1p(y) if self.use_mape_loss else y
        y_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val) if X_val is not None else None
        y_val_np = y_val

        best_val_loss = float("inf")
        no_improve = 0
        self.train_losses = []
        self.val_losses = []

        for epoch in range(_epochs):
            self.model.train()
            opt.zero_grad()
            pred = self.model(X_t)
            loss = torch.nn.MSELoss()(pred, y_t)
            loss.backward()
            opt.step()
            train_loss = loss.item()
            self.train_losses.append(train_loss)

            val_loss = None
            if X_val_t is not None and y_val_np is not None:
                self.model.eval()
                with torch.no_grad():
                    pred_val = self.model(X_val_t).numpy()
                pred_raw = np.expm1(pred_val) if self.use_mape_loss else pred_val
                val_loss = _mae_loss(y_val_np, pred_raw)
                self.val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1

            if self.patience > 0 and no_improve >= _patience:
                logger.info("  LSTM fine-tune early stop at epoch %d", epoch + 1)
                break

        return self

    def _save_checkpoint(self, name: str, torch) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / name
        torch.save(self.model.state_dict(), path)

    def _load_checkpoint(self, name: str, torch) -> None:
        path = self.checkpoint_dir / name
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location="cpu"))

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        torch = _try_import_torch()
        if self.model is None:
            raise RuntimeError("fit 먼저 호출")
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X_seq))
        pred = out.numpy()
        return np.expm1(pred) if self.use_mape_loss else pred


class TransformerWrapper:
    """Simple Temporal Transformer (lag 시퀀스 입력). eval, early stopping, checkpoint 지원."""

    def __init__(
        self,
        seq_len: int = 30,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        epochs: int = 50,
        lr: float = 0.001,
        patience: int = 10,
        dropout: float = 0.2,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.dropout = dropout
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        self.model = None
        self.best_epoch = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def fit(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
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
            def __init__(self, seq_len, d_model, nhead, num_layers, dropout=0.2):
                super().__init__()
                self.proj = nn.Linear(1, d_model)
                self.pos = PosEnc(seq_len, d_model)
                enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
                self.trans = nn.TransformerEncoder(enc, num_layers)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(d_model, 1)

            def forward(self, x):
                x = self.proj(x)
                x = self.pos(x)
                x = self.trans(x)
                h = self.dropout(x[:, -1, :])
                return self.fc(h).squeeze(-1)

        self.model = TransReg(self.seq_len, self.d_model, self.nhead, self.num_layers, dropout=self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        X_t = torch.FloatTensor(X_seq)
        y_t = torch.FloatTensor(y)
        X_val_t = torch.FloatTensor(X_val) if X_val is not None else None
        y_val_np = y_val

        best_val_loss = float("inf")
        no_improve = 0
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            pred = self.model(X_t)
            loss = nn.MSELoss()(pred, y_t)
            loss.backward()
            opt.step()
            train_loss = loss.item()
            self.train_losses.append(train_loss)

            val_loss = None
            if X_val_t is not None and y_val_np is not None:
                self.model.eval()
                with torch.no_grad():
                    pred_val = self.model(X_val_t).numpy()
                val_loss = _mae_loss(y_val_np, pred_val)
                self.val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_epoch = epoch
                    no_improve = 0
                    self._save_checkpoint("transformer_best.pt", torch)
                else:
                    no_improve += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"  Transformer epoch {epoch+1}/{self.epochs} train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" val_mae={val_loss:.4f}"
                logger.info(msg)

            if self.patience > 0 and no_improve >= self.patience:
                logger.info("  Transformer early stop at epoch %d", epoch + 1)
                break

        if X_val_t is not None and (self.checkpoint_dir / "transformer_best.pt").exists():
            self._load_checkpoint("transformer_best.pt", torch)
        return self

    def _save_checkpoint(self, name: str, torch) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_dir / name)

    def _load_checkpoint(self, name: str, torch) -> None:
        path = self.checkpoint_dir / name
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location="cpu"))

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        torch = _try_import_torch()
        if self.model is None:
            raise RuntimeError("fit 먼저 호출")
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X_seq))
        return out.numpy()
