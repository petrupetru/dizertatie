import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from constants import SEED


class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(30, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 8)
        self.fc6 = nn.Linear(8, 1)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        return x
    

class BreastCancerDataset(Dataset):
    def __init__(self, df):
        self.X = torch.tensor(df.drop(columns=["ID", "Diagnosis"]).values.astype(float), dtype=torch.float32)
        self.y = torch.tensor(df["Diagnosis"].apply(lambda x: 1 if x == "M" else 0).values.astype(float), dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def train_val_test_split(df, test_size=0.15, val_size=0.18, random_state=SEED):
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=random_state)
    return train_df, val_df, test_df

def train_val_test_datasets(train_df, val_df, test_df):
    train_dataset = BreastCancerDataset(train_df)
    val_dataset = BreastCancerDataset(val_df)
    test_dataset = BreastCancerDataset(test_df)
    return train_dataset, val_dataset, test_dataset

def train_val_test_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device="cpu"):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        # --- Training phase ---
        model.train()
        train_loss, train_correct = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_correct += ((outputs >= 0.5) == y_batch.bool()).sum().item()

        # --- Validation phase ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_correct += ((outputs >= 0.5) == y_batch.bool()).sum().item()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(
            f"Epoch [{epoch:3d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}"
        )


def test(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            probs = model(X_batch).cpu()
            all_probs.append(probs)
            all_labels.append(y_batch)

    all_probs = torch.cat(all_probs).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    all_preds = (all_probs >= 0.5).astype(int)

    accuracy = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUROC:    {auroc:.4f}")
    print(f"Test AUPRC:    {auprc:.4f}")

    return {"accuracy": accuracy, "auroc": auroc, "auprc": auprc}



if __name__ == "__main__":
    dnn = DNN()
    print(dnn)