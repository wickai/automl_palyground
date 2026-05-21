"""分类算法实现"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,
    GradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ClassificationModel:
    """分类模型基类"""
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X_train, y_train):
        raise NotImplementedError
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
    
    def transform(self, X):
        return self.scaler.transform(X)


class LogisticRegressionModel(ClassificationModel):
    """逻辑回归"""
    def __init__(self):
        super().__init__("Logistic Regression")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_train_scaled, y_train)


class NaiveBayesModel(ClassificationModel):
    """朴素贝叶斯"""
    def __init__(self):
        super().__init__("Naive Bayes")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = GaussianNB()
        self.model.fit(X_train_scaled, y_train)


class SVMModel(ClassificationModel):
    """支持向量机"""
    def __init__(self):
        super().__init__("SVM")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = SVC(probability=True, random_state=42)
        self.model.fit(X_train_scaled, y_train)


class RandomForestModel(ClassificationModel):
    """随机森林"""
    def __init__(self):
        super().__init__("Random Forest")
    
    def fit(self, X_train, y_train):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)


class KNNModel(ClassificationModel):
    """K近邻"""
    def __init__(self):
        super().__init__("KNN")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(X_train_scaled, y_train)


class LightGBMModel(ClassificationModel):
    """LightGBM"""
    def __init__(self):
        super().__init__("LightGBM")
        self.model = None
    
    def fit(self, X_train, y_train):
        try:
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            self.model.fit(X_train, y_train)
        except ImportError:
            print("LightGBM not installed, using GradientBoosting as fallback")
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
    
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)


class XGBoostModel(ClassificationModel):
    """XGBoost"""
    def __init__(self):
        super().__init__("XGBoost")
        self.model = None
    
    def fit(self, X_train, y_train):
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
            self.model.fit(X_train, y_train)
        except ImportError:
            print("XGBoost not installed, using GradientBoosting as fallback")
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)


class AdaBoostModel(ClassificationModel):
    """AdaBoost"""
    def __init__(self):
        super().__init__("AdaBoost")
    
    def fit(self, X_train, y_train):
        self.model = AdaBoostClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)


class BaggingModel(ClassificationModel):
    """Bagging"""
    def __init__(self):
        super().__init__("Bagging")
    
    def fit(self, X_train, y_train):
        self.model = BaggingClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)


class MLPModel(ClassificationModel):
    """多层感知机"""
    def __init__(self):
        super().__init__("MLP")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        self.model.fit(X_train_scaled, y_train)


class EnsembleModel(ClassificationModel):
    """集成学习（Voting + Stacking）"""
    def __init__(self):
        super().__init__("Ensemble")
        self.scaler = StandardScaler()
        self.base_models = None
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 定义基学习器
        lr = LogisticRegression(max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        
        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
            voting='soft'
        )
        
        # Stacking Classifier
        stacking_clf = StackingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3
        )
        
        # 组合
        self.base_models = {
            'voting': voting_clf,
            'stacking': stacking_clf
        }
        
        # 使用 Stacking 作为最终模型
        self.model = stacking_clf
        self.model.fit(X_train_scaled, y_train)
    
    def predict_proba(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)


# 算法注册表
CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegressionModel,
    "naive_bayes": NaiveBayesModel,
    "svm": SVMModel,
    "random_forest": RandomForestModel,
    "knn": KNNModel,
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "adaboost": AdaBoostModel,
    "bagging": BaggingModel,
    "mlp": MLPModel,
    "ensemble": EnsembleModel,
}

ALGO_DISPLAY_NAMES = {
    "logistic_regression": "1. Logistic Regression",
    "naive_bayes": "2. Naive Bayes",
    "svm": "3. SVM",
    "random_forest": "4. Random Forest",
    "knn": "5. KNN",
    "lightgbm": "6. LightGBM",
    "xgboost": "7. XGBoost",
    "adaboost": "8. AdaBoost",
    "bagging": "9. Bagging",
    "mlp": "10. MLP",
    "ensemble": "11. Ensemble",
}
