"""
LSTM-Based Forecasting Model for Fiber Degradation
Captures temporal dependencies in attenuation and PMD sequences.
Supports quantile-based uncertainty estimation.
"""

from generate_data import generate_fiber_dataset
from preprocessing import (
    clean_series, add_features, compute_health_index, THETA_1, THETA_2
)
from rf_model import FiberRFModel
# Try importing TensorFlow; gracefully fall back if unavailable
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (
        Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed. LSTM model unavailable. Using RF only.")


class FiberLSTMModel:
    """
    Bidirectional LSTM for multi-step-ahead forecasting of
    attenuation and PMD.

    Input:  (batch, lookback, n_features)
    Output: (batch, 2) → [attenuation, pmd]
    """

    def __init__(self,
                 lookback: int = 24,
                 n_features: int = 2,
                 lstm_units: int = 64,
                 dense_units: int = 32,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3):

        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for LSTM model.")

        self.lookback = lookback
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.lr = learning_rate
        self.model = None
        self.history = None

    def build(self):
        inp = Input(shape=(self.lookback, self.n_features))
        x = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(inp)
        x = Dropout(self.dropout)(x)
        x = Bidirectional(LSTM(self.lstm_units // 2))(x)
        x = Dropout(self.dropout)(x)
        x = Dense(self.dense_units, activation="relu")(x)
        x = BatchNormalization()(x)
        out = Dense(2)(x)   # attenuation + PMD

        self.model = Model(inputs=inp, outputs=out)
        self.model.compile(
            optimizer=Adam(self.lr),
            loss="mse",
            metrics=["mae"]
        )
        self.model.summary()
        return self

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 100, batch_size: int = 32):

        if self.model is None:
            self.build()

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, verbose=1),
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, 2) predictions."""
        return self.model.predict(X, verbose=0)

    def predict_with_intervals(self, X: np.ndarray,
                                n_samples: int = 50) -> dict:
        """
        Monte Carlo Dropout for uncertainty estimation.
        Runs n_samples stochastic forward passes (dropout active at test time).
        """
        # Enable MC Dropout by calling model in training mode
        preds = np.array([
            self.model(X, training=True).numpy()
            for _ in range(n_samples)
        ])  # (n_samples, N, 2)

        return {
            "attenuation_dB_km": {
                "mean":  preds[:, :, 0].mean(axis=0),
                "lower": np.percentile(preds[:, :, 0], 5, axis=0),
                "upper": np.percentile(preds[:, :, 0], 95, axis=0),
            },
            "pmd_ps_sqkm": {
                "mean":  preds[:, :, 1].mean(axis=0),
                "lower": np.percentile(preds[:, :, 1], 5, axis=0),
                "upper": np.percentile(preds[:, :, 1], 95, axis=0),
            },
        }

    def plot_training(self, save_path: str = "outputs/lstm_training.png"):
        if self.history is None:
            print("No training history.")
            return
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.history.history["loss"], label="Train Loss")
        axes[0].plot(self.history.history["val_loss"], label="Val Loss")
        axes[0].set_title("MSE Loss")
        axes[0].legend()
        axes[1].plot(self.history.history["mae"], label="Train MAE")
        axes[1].plot(self.history.history["val_mae"], label="Val MAE")
        axes[1].set_title("MAE")
        axes[1].legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training curve saved → {save_path}")

    def save(self, path: str = "outputs/lstm_model.keras"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"LSTM model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "FiberLSTMModel":
        obj = cls.__new__(cls)
        obj.model = load_model(path)
        return obj
