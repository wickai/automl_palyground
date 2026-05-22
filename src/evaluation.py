"""评估指标模块"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)


def evaluate_classification(y_true, y_pred, y_prob=None):
    """评估分类模型
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（用于 AUC）
    
    Returns:
        dict: 包含准确率、精准率、召回率、F1、AUC 的字典
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }
    if y_prob is not None:
        try:
            if hasattr(y_prob, "to_numpy"):
                y_prob_array = y_prob.to_numpy()
            else:
                y_prob_array = np.asarray(y_prob)

            if y_prob_array.ndim > 1:
                if y_prob_array.shape[1] > 1:
                    y_prob_array = y_prob_array[:, 1]
                else:
                    y_prob_array = y_prob_array.reshape(-1)

            metrics["auc"] = roc_auc_score(y_true, y_prob_array)
        except:
            metrics["auc"] = None
    else:
        metrics["auc"] = None
    return metrics


def evaluate_regression(y_true, y_pred):
    """评估回归模型
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        dict: 包含 RMSE、MAE、R² 的字典
    """
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    return metrics


def print_metrics(task_type, metrics):
    """打印评估指标"""
    if task_type == "classification":
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision:{metrics['precision']:.4f}")
        print(f"  Recall:   {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        if metrics.get("auc") is not None:
            print(f"  AUC:      {metrics['auc']:.4f}")
    else:
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
