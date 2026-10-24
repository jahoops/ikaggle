import numpy as np
import tensorflow as tf

def tensorflow_dropout_predict(model, X, n_iter=100):
    # Define a function to perform prediction with dropout enabled
    f = tf.function(lambda x: model(x, training=True))
    preds = np.array([f(X) for _ in range(n_iter)])
    pred_mean = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)
    # Display prediction confidence
    print(f'Predicted Mean: {pred_mean}')
    print(f'Prediction Standard Deviation (Confidence): {pred_std}')
    return pred_mean, pred_std



