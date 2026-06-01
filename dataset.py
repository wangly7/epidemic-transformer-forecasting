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



class MultivariateDataset:
    def __init__(
        self,
        state_ilinet_path: str,
        national_ilinet_path: str,
        lab_path: str,
        target_region: str = None
    ):
        self.state_ilinet_path = state_ilinet_path
        self.national_ilinet_path = national_ilinet_path
        self.lab_path = lab_path
        self.target_region = target_region

        self.state_ilinet = pd.read_csv(state_ilinet_path, header=1)
        self.national_ilinet = pd.read_csv(national_ilinet_path, header=1)
        self.lab = pd.read_csv(lab_path, header=1)

    def clean_state_ilinet_dataframe(self):
        df = self.state_ilinet.copy()

        df.columns = df.columns.str.strip()

        df["REGION"] = df["REGION"].astype(str).str.strip()
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce")
        df["%UNWEIGHTED ILI"] = pd.to_numeric(
            df["%UNWEIGHTED ILI"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "REGION",
                "YEAR",
                "WEEK",
                "%UNWEIGHTED ILI"
            ]
        )

        if self.target_region is not None:
            df = df[df["REGION"] == self.target_region].copy()

        df = df[
            ["REGION", "YEAR", "WEEK", "%UNWEIGHTED ILI"]
        ].copy()

        return df

    def clean_national_ilinet_dataframe(self):
        df = self.national_ilinet.copy()

        df.columns = df.columns.str.strip()

        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce")
        df["%UNWEIGHTED ILI"] = pd.to_numeric(
            df["%UNWEIGHTED ILI"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "YEAR",
                "WEEK",
                "%UNWEIGHTED ILI"
            ]
        )

        df = df[
            ["YEAR", "WEEK", "%UNWEIGHTED ILI"]
        ].copy()

        df = df.rename(
            columns={"%UNWEIGHTED ILI": "national_ili"}
        )

        df = (
            df
            .groupby(["YEAR", "WEEK"], as_index=False)["national_ili"]
            .mean()
        )

        return df

    def clean_lab_dataframe(self):
        df = self.lab.copy()

        df.columns = df.columns.str.strip()

        df["REGION"] = df["REGION"].astype(str).str.strip()
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce")

        df["PERCENT POSITIVE"] = pd.to_numeric(
            df["PERCENT POSITIVE"],
            errors="coerce"
        )

        df = df.dropna(
            subset=[
                "REGION",
                "YEAR",
                "WEEK"
            ]
        )

        if self.target_region is not None:
            df = df[df["REGION"] == self.target_region].copy()

        df = df[
            ["REGION", "YEAR", "WEEK", "PERCENT POSITIVE"]
        ].copy()

        df = (
            df
            .groupby(["REGION", "YEAR", "WEEK"], as_index=False)["PERCENT POSITIVE"]
            .mean()
        )

        return df

    def build_multivariate_dataframe(self):
        state_df = self.clean_state_ilinet_dataframe()
        national_df = self.clean_national_ilinet_dataframe()
        lab_df = self.clean_lab_dataframe()

        df = state_df.merge(
            national_df,
            on=["YEAR", "WEEK"],
            how="left"
        )

        df = df.merge(
            lab_df,
            on=["REGION", "YEAR", "WEEK"],
            how="left"
        )

        df = df.sort_values(
            ["REGION", "YEAR", "WEEK"]
        ).reset_index(drop=True)

        df = df.dropna(
            subset=[
                "%UNWEIGHTED ILI",
                "national_ili"
            ]
        )

        return df

    def sliding_window_multivariate(
        self,
        features,
        target,
        history=10,
        future=4
    ):
        x, y = [], []

        for i in range(len(features) - history - future + 1):
            x_window = features[i : i + history]
            y_window = target[i + history : i + history + future]

            if np.isnan(x_window).any() or np.isnan(y_window).any():
                continue

            x.append(x_window)
            y.append(y_window)

        return np.array(x), np.array(y)

    def build_split_windows(
        self,
        features,
        target,
        history,
        future,
        test_size,
        val_size
    ):
        x_all, y_all = self.sliding_window_multivariate(
            features,
            target,
            history,
            future
        )
    
        n = len(x_all)
    
        if n == 0:
            return (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([])
            )
    
        test_start = int(n * (1 - test_size))
        val_start = int(test_start * (1 - val_size))
    
        x_train = x_all[:val_start]
        y_train = y_all[:val_start]
    
        x_val = x_all[val_start:test_start]
        y_val = y_all[val_start:test_start]
    
        x_test = x_all[test_start:]
        y_test = y_all[test_start:]
    
        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_train_val_test_loader(
        self,
        history=10,
        future=4,
        test_size=0.2,
        val_size=0.1,
        batch_size=64
    ):
        df = self.build_multivariate_dataframe()

        feature_cols = [
            "%UNWEIGHTED ILI",
            "national_ili",
            "PERCENT POSITIVE"
        ]

        target_col = "%UNWEIGHTED ILI"

        train_x, train_y = [], []
        val_x, val_y = [], []
        test_x, test_y = [], []

        for region, g in df.groupby("REGION"):
            g = g.sort_values(["YEAR", "WEEK"])

            features = g[feature_cols].values.astype(np.float32)
            target = g[[target_col]].values.astype(np.float32)

            if len(g) < history + future:
                continue

            x_tr, y_tr, x_va, y_va, x_te, y_te = self.build_split_windows(
                features,
                target,
                history,
                future,
                test_size,
                val_size
            )

            if len(x_tr) == 0 or len(x_va) == 0 or len(x_te) == 0:
                continue

            train_x.append(x_tr)
            train_y.append(y_tr)

            val_x.append(x_va)
            val_y.append(y_va)

            test_x.append(x_te)
            test_y.append(y_te)

        if len(train_x) == 0:
            raise ValueError(
                "No valid training samples were created. "
                "This may be caused by too many missing PERCENT POSITIVE values."
            )

        x_train = np.concatenate(train_x, axis=0)
        y_train = np.concatenate(train_y, axis=0)

        x_val = np.concatenate(val_x, axis=0)
        y_val = np.concatenate(val_y, axis=0)

        x_test = np.concatenate(test_x, axis=0)
        y_test = np.concatenate(test_y, axis=0)

        train_dataset = TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).float()
        )

        val_dataset = TensorDataset(
            torch.from_numpy(x_val).float(),
            torch.from_numpy(y_val).float()
        )

        test_dataset = TensorDataset(
            torch.from_numpy(x_test).float(),
            torch.from_numpy(y_test).float()
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

        print(f"Feature columns: {feature_cols}")
        print(f"Target column: {target_col}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        xb, yb = next(iter(train_loader))
        print("x batch shape:", xb.shape)
        print("y batch shape:", yb.shape)

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