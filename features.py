import numpy as np

def extract_features(prediction_errors, bins=5):
    hist, _ = np.histogram(prediction_errors, bins=bins, range=(-50, 50), density=True)
    features = [
        np.mean(prediction_errors),
        np.std(prediction_errors),
        np.max(prediction_errors),
        np.min(prediction_errors),
        np.median(prediction_errors)
    ]
    features.extend(hist.tolist())
    return np.array(features)
extract_global_features = extract_features
