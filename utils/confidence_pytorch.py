import torch
import numpy as np

def pytorch_dropout_predict(model, X, n_iter=100):
    model.train()  # Enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_iter):
            preds.append(model(X).numpy())
    preds = np.array(preds)
    pred_mean = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)
    print(f'Predicted Mean: {pred_mean}')
    print(f'Prediction Standard Deviation (Confidence): {pred_std}')
    return pred_mean, pred_std


