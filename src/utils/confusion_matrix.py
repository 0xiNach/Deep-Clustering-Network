import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from parameters import Params


def plot_cm(label_matrix, predictions, class_names):
    """plot confusion matrix"""
    preds = predictions
    labels_ = label_matrix

    cm = confusion_matrix(labels_, preds, labels=np.arange(Params.num_classes))
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    indices = np.arange(Params.num_classes)
    plt.xticks(indices + 0.5, class_names, rotation=45)
    plt.yticks(indices + 0.5, class_names, rotation="horizontal")
    plt.title("Confusion matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"{Params.test_log_dir}/test_cm.png")
