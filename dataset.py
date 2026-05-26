import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class Dataset:
    def __init__(self, path: str):
        self.path = path
        self.ilinet = pd.read_csv(path, header=1)

    def clean_dataframe(self):
        df = self.ilinet.copy()

        df.columns = df.columns.str.strip()

        df["REGION TYPE"] = df["REGION TYPE"].astype(str).str.strip()
        df["REGION"] = df["REGION"].astype(str).str.strip()

        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce")
        df["%UNWEIGHTED ILI"] = pd.to_numeric(
            df["%UNWEIGHTED ILI"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["REGION TYPE", "REGION", "YEAR", "WEEK", "%UNWEIGHTED ILI"]
        )

        return df

    def sliding_window(self, series, history=10, future=4):
        x, y = [], []

        for i in range(len(series) - history - future + 1):
            x.append(series[i : i + history])
            y.append(series[i + history : i + history + future])

        return np.array(x), np.array(y)

    def build_split_windows(self, series, history, future, test_size, val_size):
        n = len(series)

        test_start = int(n * (1 - test_size))
        val_start = int(test_start * (1 - val_size))

        train_series = series[:val_start]
        val_series = series[val_start - history : test_start]
        test_series = series[test_start - history :]

        x_train, y_train = self.sliding_window(train_series, history, future)
        x_val, y_val = self.sliding_window(val_series, history, future)
        x_test, y_test = self.sliding_window(test_series, history, future)

        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_train_val_test_loader(
        self,
        history=10,
        future=4,
        test_size=0.2,
        val_size=0.1,
        batch_size=64
    ):
        df = self.clean_dataframe()

        train_x, train_y = [], []
        val_x, val_y = [], []
        test_x, test_y = [], []

        for state, g in df.groupby("REGION"):
            g = g.sort_values(["YEAR", "WEEK"])
            series = g["%UNWEIGHTED ILI"].values.astype(np.float32)

            if len(series) < history + future:
                continue

            x_tr, y_tr, x_va, y_va, x_te, y_te = self.build_split_windows(
                series,
                history,
                future,
                test_size,
                val_size
            )

            train_x.append(x_tr)
            train_y.append(y_tr)
            val_x.append(x_va)
            val_y.append(y_va)
            test_x.append(x_te)
            test_y.append(y_te)

        x_train = np.concatenate(train_x, axis=0)
        y_train = np.concatenate(train_y, axis=0)
        x_val = np.concatenate(val_x, axis=0)
        y_val = np.concatenate(val_y, axis=0)
        x_test = np.concatenate(test_x, axis=0)
        y_test = np.concatenate(test_y, axis=0)

        train_dataset = TensorDataset(
            torch.from_numpy(x_train).float().unsqueeze(-1),
            torch.from_numpy(y_train).float().unsqueeze(-1)
        )

        val_dataset = TensorDataset(
            torch.from_numpy(x_val).float().unsqueeze(-1),
            torch.from_numpy(y_val).float().unsqueeze(-1)
        )

        test_dataset = TensorDataset(
            torch.from_numpy(x_test).float().unsqueeze(-1),
            torch.from_numpy(y_test).float().unsqueeze(-1)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset = Dataset("./data/national/ILINet.csv")

    train_loader, val_loader, test_loader = dataset.get_train_val_test_loader(
        history=10,
        future=4,
        test_size=0.2,
        val_size=0.1,
        batch_size=64
    )

    xb, yb = next(iter(train_loader))
    print("x batch shape:", xb.shape)  # [B, 10, 1]
    print("y batch shape:", yb.shape)  # [B, 4, 1]