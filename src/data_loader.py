"""数据下载和加载模块"""
import os
import zipfile
import tarfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, fetch_california_housing


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_breast_cancer(self):
        """加载 sklearn 内置乳腺癌数据集（分类任务）"""
        print("Loading Breast Cancer dataset...")
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Breast Cancer: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
    
    def load_california_housing(self):
        """加载 California Housing 数据集（回归任务）"""
        print("Loading California Housing dataset...")
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="MedHouseVal")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"California Housing: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
    
    def download_kaggle_breast_cancer(self):
        """从 Kaggle 下载乳腺癌数据集（如果可用）"""
        # 使用 sklearn 内置数据集作为替代
        return self.load_breast_cancer()
    
    def download_kaggle_california_housing(self):
        """从 Kaggle 下载 California Housing 数据集"""
        # 使用 sklearn 内置数据集作为替代
        return self.load_california_housing()


if __name__ == "__main__":
    loader = DataLoader()
    # Test classification data
    X_train, X_test, y_train, y_test = loader.load_breast_cancer()
    print(f"Classification data loaded: {X_train.shape}")
    # Test regression data
    X_train, X_test, y_train, y_test = loader.load_california_housing()
    print(f"Regression data loaded: {X_train.shape}")
