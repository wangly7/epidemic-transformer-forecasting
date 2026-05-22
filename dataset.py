import pandas as pd
import torch 

class Dataset:
    def __init__(self, path: str):
        self.path = path
        self.ilinet = pd.read_csv(path, header=1)

    def get_series_data(self):
        series = self.ilinet["%UNWEIGHTED ILI"]
        return series

    def sliding_window(self, series, hisotry=10, future=4):
        x, y = [], []
        for i in range(len(series) - hisotry - future):
            x.append(series[i : i + hisotry].values)
            y.append(series[i + hisotry : i + hisotry + future].values)
        return x, y
    
    def get_train_and_test_data(self, history=10, future=4, test_size=0.2):
        series = self.get_series_data()
        x, y = self.sliding_window(series, history, future)
        split_index = int(len(x) * (1 - test_size))
        x_train, y_train = x[:split_index], y[:split_index]
        x_test, y_test = x[split_index:], y[split_index:]
        return x_train, y_train, x_test, y_test
    
    def get_train_val_test_loader(
        self,
        history=10,
        future=4,
        test_size=0.2,
        val_size=0.1,
        batch_size=32
    ):
        x_train, y_train, x_test, y_test = self.get_train_and_test_data(
            history,
            future,
            test_size
        )
        # split train -> train + validation
        val_split = int(len(x_train) * (1 - val_size))
        x_val = x_train[val_split:]
        y_val = y_train[val_split:]

        x_train = x_train[:val_split]
        y_train = y_train[:val_split]
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_val, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y_val, dtype=torch.float32)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y_test, dtype=torch.float32)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    dataset = Dataset("./data/national/ILINET.csv")
    x_train, y_train, x_test, y_test = dataset.get_train_and_test_data(history=10, future=4, test_size=0.2)
    train_loader, test_loader = dataset.get_train_and_test_loader(history=10, future=4, test_size=0.2, batch_size=32)

