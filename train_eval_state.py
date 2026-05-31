import os
import copy
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Setup and Reproducibility
# ----------------------------------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ----------------------------------------------------
# 2. Model Architecture
# ----------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return x

class DeepTransformer(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1, d_model: int = 64, num_heads: int = 4,
                 ff_dim: int = 256, encoder_layers: int = 4, decoder_layers: int = 4,
                 dropout: float = 0.2):
        super(DeepTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.position_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    def generate_square_subsequent_mask(self, T, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, src, tgt):
        src = self.input_proj(src)
        src = self.position_encoder(src)
        memory = self.encoder(src)

        tgt = self.input_proj(tgt)
        tgt = self.position_encoder(tgt)

        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, tgt.device)

        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.output_proj(out)

# ----------------------------------------------------
# 3. Training Pipeline
# ----------------------------------------------------
class PipeLine:
    def __init__(self, model, train_loader, val_loader, warmup_steps=5000, d_model=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.model = model.to(self.device)
        self.d_model = d_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.warmup_steps = warmup_steps
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1.0,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self.lr_lambda
        )
        self.loss_func = nn.MSELoss()

    def lr_lambda(self, step):
        step = max(step, 1)
        lr = (self.d_model ** (-0.5)) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return lr

    def train_model(self, epochs):
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch()
            train_losses.append(train_loss)
            val_loss = self.evaluate(self.val_loader)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Load best weights
        self.model.load_state_dict(best_model_state)
        return self.model, train_losses, val_losses

    def train_one_epoch(self):
        self.model.train()
        total_loss, total_count = 0.0, 0
        for src, y in self.train_loader:
            src, y = src.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            # Teacher forcing target
            tgt = torch.cat([src[:, -1:, :], y[:, :-1, :]], dim=1)
            pred = self.model(src, tgt)
            loss = self.loss_func(pred, y)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item() * src.size(0)
            total_count += src.size(0)
        return total_loss / total_count

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss, total_count = 0.0, 0
        for src, y in dataloader:
            src, y = src.to(self.device), y.to(self.device)
            tgt = torch.cat([src[:, -1:, :], y[:, :-1, :]], dim=1)
            pred = self.model(src, tgt)
            loss = self.loss_func(pred, y)
            total_loss += loss.item() * src.size(0)
            total_count += src.size(0)
        return total_loss / total_count

    @torch.no_grad()
    def predict_autoregressive(self, src, future=4):
        self.model.eval()
        tgt = src[:, -1:, :]
        preds = []
        for _ in range(future):
            out = self.model(src, tgt)
            next_pred = out[:, -1:, :]
            preds.append(next_pred)
            tgt = torch.cat([tgt, next_pred], dim=1)
        return torch.cat(preds, dim=1)

# ----------------------------------------------------
# 4. ARIMA Predictor helper
# ----------------------------------------------------
def arima_predict_window(history_window, order=(3, 0, 3), steps=4):
    try:
        model = ARIMA(history_window, order=order, trend="c")
        fit = model.fit(method_kwargs={"maxiter": 200})
        pred = fit.forecast(steps=steps)
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            raise ValueError("Invalid ARIMA prediction")
        return pred, False
    except Exception:
        # Fallback: repeat last value
        return np.full(steps, history_window[-1]), True

# ----------------------------------------------------
# 5. Main Execution Script
# ----------------------------------------------------
if __name__ == "__main__":
    from dataset import Dataset

    # Load dataset to split per state identically
    dataset = Dataset(path="./data/state/ILINet.csv")
    train_loader, val_loader, test_loader = dataset.get_train_val_test_loader(
        history=10,
        future=4,
        batch_size=64
    )

    # Instantiate and train DeepTransformer on State-Level Dataset
    transformer_model = DeepTransformer()
    pipeline = PipeLine(model=transformer_model, train_loader=train_loader, val_loader=val_loader)
    
    if os.path.exists("best_state_transformer.pt"):
        print("\nFound saved weights best_state_transformer.pt. Loading instead of training...")
        transformer_model.load_state_dict(torch.load("best_state_transformer.pt", map_location=pipeline.device))
        train_losses = []
        val_losses = []
    else:
        print("\nTraining DeepTransformer on State-Level Dataset...")
        best_model, train_losses, val_losses = pipeline.train_model(200)
        # Save weights
        torch.save(best_model.state_dict(), "best_state_transformer.pt")
        print("Model saved to: best_state_transformer.pt")

    # Now prepare per-state evaluation
    df_state = dataset.clean_dataframe()
    all_states = sorted(df_state["REGION"].unique().tolist())
    print(f"\nEvaluating state-by-state for {len(all_states)} states...")

    HISTORY = 10
    FUTURE = 4
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1

    state_results = {}
    # Structure: { state_name: { "transformer": {"preds": (N, 4), "labels": (N, 4)}, "arima": ..., "persistence": ... } }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_transformer_preds, all_transformer_labels = [], []
    all_arima_preds, all_arima_labels = [], []
    all_persist_preds, all_persist_labels = [], []

    for state in all_states:
        # Group by and sort
        g = df_state[df_state["REGION"] == state].sort_values(["YEAR", "WEEK"])
        series = g["%UNWEIGHTED ILI"].values.astype(np.float32)

        if len(series) < HISTORY + FUTURE:
            continue

        # Get the split windows exactly as in dataset.py
        x_tr, y_tr, x_va, y_va, x_te, y_te = dataset.build_split_windows(
            series, HISTORY, FUTURE, TEST_SIZE, VAL_SIZE
        )

        n_samples = len(x_te)
        if n_samples == 0:
            continue

        # -- 1. Persistence baseline --
        p_last = x_te[:, -1]
        persist_preds = np.tile(p_last, (FUTURE, 1)).T
        persist_labels = y_te.squeeze(-1) if y_te.ndim == 3 else y_te

        # -- 2. ARIMA baseline --
        arima_preds = []
        for window in x_te:
            pred, _ = arima_predict_window(window, order=(3, 0, 3), steps=FUTURE)
            arima_preds.append(pred)
        arima_preds = np.array(arima_preds)
        arima_labels = y_te.squeeze(-1) if y_te.ndim == 3 else y_te

        # -- 3. Transformer Model --
        x_te_tensor = torch.from_numpy(x_te).float()
        if x_te_tensor.ndim == 2:
            x_te_tensor = x_te_tensor.unsqueeze(-1) # [N, 10, 1]
        x_te_tensor = x_te_tensor.to(device)

        with torch.no_grad():
            t_preds_tensor = pipeline.predict_autoregressive(x_te_tensor, future=FUTURE)
            t_preds = t_preds_tensor.cpu().numpy().squeeze(-1)
        t_labels = y_te.squeeze(-1) if y_te.ndim == 3 else y_te

        # Save lists for overall metrics
        all_transformer_preds.append(t_preds)
        all_transformer_labels.append(t_labels)
        all_arima_preds.append(arima_preds)
        all_arima_labels.append(arima_labels)
        all_persist_preds.append(persist_preds)
        all_persist_labels.append(persist_labels)

        # Helper to calculate step and overall metrics
        def compute_metrics_dict(p, l):
            m = {}
            for step in range(FUTURE):
                step_p = p[:, step]
                step_l = l[:, step]
                rmse = np.sqrt(mean_squared_error(step_l, step_p))
                mae = mean_absolute_error(step_l, step_p)
                if np.std(step_p) == 0 or np.std(step_l) == 0:
                    corr = np.nan
                else:
                    corr, _ = pearsonr(step_p, step_l)
                m[f"week{step+1}"] = {"rmse": rmse, "mae": mae, "pearson": corr}
            
            p_flat = p.flatten()
            l_flat = l.flatten()
            overall_rmse = np.sqrt(mean_squared_error(l_flat, p_flat))
            overall_mae = mean_absolute_error(l_flat, p_flat)
            if np.std(p_flat) == 0 or np.std(l_flat) == 0:
                overall_corr = np.nan
            else:
                overall_corr, _ = pearsonr(p_flat, l_flat)
            m["overall"] = {"rmse": overall_rmse, "mae": overall_mae, "pearson": overall_corr}
            return m

        state_results[state] = {
            "transformer": {
                "preds": t_preds,
                "labels": t_labels,
                "metrics": compute_metrics_dict(t_preds, t_labels)
            },
            "arima": {
                "preds": arima_preds,
                "labels": arima_labels,
                "metrics": compute_metrics_dict(arima_preds, arima_labels)
            },
            "persistence": {
                "preds": persist_preds,
                "labels": persist_labels,
                "metrics": compute_metrics_dict(persist_preds, persist_labels)
            }
        }

    # Concatenate all state metrics
    trans_preds_all = np.concatenate(all_transformer_preds, axis=0)
    trans_labels_all = np.concatenate(all_transformer_labels, axis=0)
    arima_preds_all = np.concatenate(all_arima_preds, axis=0)
    arima_labels_all = np.concatenate(all_arima_labels, axis=0)
    persist_preds_all = np.concatenate(all_persist_preds, axis=0)
    persist_labels_all = np.concatenate(all_persist_labels, axis=0)

    # Save globally
    np.save("transformer_state_preds.npy", trans_preds_all)
    np.save("transformer_state_labels.npy", trans_labels_all)
    np.save("arima_state_preds.npy", arima_preds_all)
    np.save("arima_state_labels.npy", arima_labels_all)
    np.save("persistence_state_preds.npy", persist_preds_all)
    np.save("persistence_state_labels.npy", persist_labels_all)

    # Compute overall metrics
    def get_overall_metrics(preds, labels):
        results = {}
        for step in range(FUTURE):
            p = preds[:, step]
            l = labels[:, step]
            rmse = np.sqrt(mean_squared_error(l, p))
            mae = mean_absolute_error(l, p)
            corr, _ = pearsonr(p, l)
            results[f"week{step+1}"] = {"rmse": rmse, "mae": mae, "pearson": corr}
        
        p_flat = preds.flatten()
        l_flat = labels.flatten()
        overall_rmse = np.sqrt(mean_squared_error(l_flat, p_flat))
        overall_mae = mean_absolute_error(l_flat, p_flat)
        overall_corr, _ = pearsonr(p_flat, l_flat)
        results["overall"] = {"rmse": overall_rmse, "mae": overall_mae, "pearson": overall_corr}
        return results

    global_metrics = {
        "transformer": get_overall_metrics(trans_preds_all, trans_labels_all),
        "arima": get_overall_metrics(arima_preds_all, arima_labels_all),
        "persistence": get_overall_metrics(persist_preds_all, persist_labels_all)
    }

    # Save comparative summary dictionary
    with open("state_evaluation_comparison.pkl", "wb") as f:
        pickle.dump({"state_results": state_results, "global_metrics": global_metrics}, f)
    print("Metrics dictionary saved to: state_evaluation_comparison.pkl")

    # Save numpy predictions/labels shapes info
    print("\nPredictions shapes:")
    print("  Transformer state predictions:", trans_preds_all.shape)
    print("  ARIMA state predictions:", arima_preds_all.shape)
    print("  Persistence state predictions:", persist_preds_all.shape)

    # Print Comparative Table
    print("\n" + "="*80)
    print(f"{'Model / Horizon':<25} {'RMSE':>10} {'MAE':>10} {'Pearson Corr':>15}")
    print("="*80)
    for model_name in ["persistence", "arima", "transformer"]:
        for step in range(FUTURE):
            m = global_metrics[model_name][f"week{step+1}"]
            print(f"  {f'Week+{step+1} {model_name.capitalize()}':<23}  {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['pearson']:>15.4f}")
        m_overall = global_metrics[model_name]["overall"]
        print("-" * 80)
        print(f"  {f'Overall {model_name.capitalize()}':<23}  {m_overall['rmse']:>10.4f} {m_overall['mae']:>10.4f} {m_overall['pearson']:>15.4f}")
        print("="*80)

    # ----------------------------------------------------
    # 6. Visualization
    # ----------------------------------------------------
    # 1. Training Losses Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="#1F77B4", linewidth=1.5)
    plt.plot(val_losses, label="Validation Loss", color="#FF7F0E", linewidth=1.5)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("MSE Loss", fontsize=10)
    plt.title("State-Level Transformer Training Curves", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("state_transformer_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot saved: state_transformer_training_curves.png")

    # 2. Model performance comparisons per horizon
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_to_plot = ["rmse", "mae", "pearson"]
    titles = ["Overall RMSE (Lower is Better)", "Overall MAE (Lower is Better)", "Overall Pearson Correlation (Higher is Better)"]
    models = ["persistence", "arima", "transformer"]
    colors = ["#FF9800", "#F44336", "#2196F3"]

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [global_metrics[m]["overall"][metric] for m in models]
        bars = ax.bar([m.capitalize() for m in models], values, color=colors, width=0.5)
        ax.set_title(titles[idx], fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        
        # Attach labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.suptitle("Model Evaluation Comparison — State Level (4332 test windows)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("state_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot saved: state_model_comparison.png")

    # 3. Example Visualizations for 4 states
    example_states = ["California", "Texas", "New York", "Florida"]
    fig, axes = plt.subplots(len(example_states), FUTURE, figsize=(16, 12))
    fig.suptitle("Model Predictions vs Ground Truth — State Level (4 Example States)", fontsize=14, fontweight="bold")

    for row, state in enumerate(example_states):
        labels = state_results[state]["transformer"]["labels"]
        t_preds = state_results[state]["transformer"]["preds"]
        a_preds = state_results[state]["arima"]["preds"]
        p_preds = state_results[state]["persistence"]["preds"]
        
        for step in range(FUTURE):
            ax = axes[row][step]
            ax.plot(labels[:, step], color="black", linewidth=1.2, label="Actual")
            ax.plot(p_preds[:, step], color="#FF9800", linewidth=1.0, linestyle=":", label="Persistence")
            ax.plot(a_preds[:, step], color="#F44336", linewidth=1.0, linestyle="-.", label="ARIMA")
            ax.plot(t_preds[:, step], color="#2196F3", linewidth=1.2, linestyle="--", label="Transformer")
            
            rmse_val = state_results[state]["transformer"]["metrics"][f"week{step+1}"]["rmse"]
            ax.set_title(f"{state} W+{step+1}\nTransformer RMSE={rmse_val:.3f}", fontsize=8)
            ax.set_ylabel("ILI %", fontsize=8)
            if row == 0 and step == 0:
                ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("state_model_predictions_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot saved: state_model_predictions_examples.png")
