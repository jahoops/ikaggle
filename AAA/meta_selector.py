import pandas as pd
import torch
import torch.nn as nn

class MetaSelectorNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class MetaSelector:
    def __init__(self):
        self.model = None

    def fit(self, binind_matrix: pd.DataFrame, best_sub_ids: pd.Series, submissions, epochs=5):
        # Convert bin indicators into a tensor, sub_ids into binary classes per ID or a multi-class approach
        X_tensor = torch.tensor(binind_matrix.values, dtype=torch.float32)
        y_tensor = torch.tensor(best_sub_ids.values, dtype=torch.float32).unsqueeze(1)
        self.model = MetaSelectorNN(X_tensor.shape[1])
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = loss_fn(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, binind_matrix: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("MetaSelector must be trained first")
        
        X_tensor = torch.tensor(binind_matrix.values, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_tensor).squeeze().numpy()
        
        # Return predicted "preference score" for submission set 1, for example
        return pd.Series(probs, index=binind_matrix.index)


