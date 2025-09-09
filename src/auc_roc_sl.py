import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import os

import matplotlib.pyplot as plt

y_pred = [0.55, 0.8, 0.3, 0.95, 0.7]
y_true = [0, 1, 1, 1, 0]
auc = roc_auc_score(y_true, y_pred)
print(f"AUC-ROC score: {auc:.4f}")



