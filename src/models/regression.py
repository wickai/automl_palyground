"""回归算法实现"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, BaggingRegressor,
    GradientBoostingRegressor, VotingRegressor, StackingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class RegressionModel:
    """回归模型基类"""
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def fit(self, X_train, y_train):
        raise NotImplementedError
    
    def predict(self, X_test):
        return self.model.predict(X_test)


class LinearRegressionModel(RegressionModel):
    """线性回归"""
    def __init__(self):
        super().__init__("Linear Regression")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train_scaled)
    
    def predict(self, X_test):
        X_test_scaled = self.scaler_x.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()


class RidgeModel(RegressionModel):
    """岭回归"""
    def __init__(self):
        super().__init__("Ridge Regression")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_train_scaled, y_train_scaled)
    
    def predict(self, X_test):
        X_test_scaled = self.scaler_x.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()


class LassoModel(RegressionModel):
    """Lasso回归"""
    def __init__(self):
        super().__init__("Lasso Regression")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        self.model = Lasso(alpha=0.1)
        self.model.fit(X_train_scaled, y_train_scaled)
    
    def predict(self, X_test):
        X_test_scaled = self.scaler_x.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()


class SVMModel(RegressionModel):
    """支持向量回归"""
    def __init__(self):
        super().__init__("SVR")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        self.model = SVR(kernel='rbf')
        self.model.fit(X_train_scaled, y_train_scaled)
    
    def predict(self, X_test):
        X_test_scaled = self.scaler_x.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()


class RandomForestModel(RegressionModel):
    """随机森林回归"""
    def __init__(self):
        super().__init__("Random Forest")
    
    def fit(self, X_train, y_train):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)


class KNNModel(RegressionModel):
    """K近邻回归"""
    def __init__(self):
        super().__init__("KNN")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        self.model = KNeighborsRegressor(n_neighbors=5)
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, X_test):
        X_test_scaled = self.scaler_x.transform(X_test)
        return self.model.predict(X_test_scaled)


class LightGBMModel(RegressionModel):
    """LightGBM回归"""
    def __init__(self):
        super().__init__("LightGBM")
        self.model = None
    
    def fit(self, X_train, y_train):
        try:
            from lightgbm import LGBMRegressor
            self.model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            self.model.fit(X_train, y_train)
        except ImportError:
            print("LightGBM not installed, using GradientBoosting as fallback")
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)


class XGBoostModel(RegressionModel):
    """XGBoost回归"""
    def __init__(self):
        super().__init__("XGBoost")
        self.model = None
    
    def fit(self, X_train, y_train):
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
        except ImportError:
            print("XGBoost not installed, using GradientBoosting as fallback")
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)


class AdaBoostModel(RegressionModel):
    """AdaBoost回归"""
    def __init__(self):
        super().__init__("AdaBoost")
    
    def fit(self, X_train, y_train):
        self.model = AdaBoostRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)


class BaggingModel(RegressionModel):
    """Bagging回归"""
    def __init__(self):
        super().__init__("Bagging")
    
    def fit(self, X_train, y_train):
        self.model = BaggingRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)


class MLPModel(RegressionModel):
    """多层感知机回归"""
    def __init__(self):
        super().__init__("MLP")
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        self.model.fit(X_train_scaled, y_train_scaled)
    
    def predict(self, X_test):
        X_test_scaled = self.scaler_x.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()


class EnsembleModel(RegressionModel):
    """集成学习回归（Voting + Stacking）"""
    def __init__(self):
        super().__init__("Ensemble")
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler_x.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
        # 定义基学习器
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
        
        # Stacking Regressor
        self.model = StackingRegressor(
            estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
            final_estimator=Ridge(),
            cv=3
        )
        self.model.fit(X_train_scaled, y_train_scaled)
    
    def predict(self, X_test):
        X_test_scaled = self.scaler_x.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        return self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()


# 算法注册表
REGRESSION_MODELS = {
    "linear_regression": LinearRegressionModel,
    "ridge": RidgeModel,
    "lasso": LassoModel,
    "svr": SVMModel,
    "random_forest": RandomForestModel,
    "knn": KNNModel,
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "adaboost": AdaBoostModel,
    "bagging": BaggingModel,
    "mlp": MLPModel,
    "ensemble": EnsembleModel,
}

# 分类算法映射（回归使用类似的算法，但适配回归任务）
CLASSIFICATION_TO_REGRESSION = {
    "logistic_regression": "linear_regression",
    "naive_bayes": "ridge",  # 使用Ridge作为替代
    "svm": "svr",
    "random_forest": "random_forest",
    "knn": "knn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "adaboost": "adaboost",
    "bagging": "bagging",
    "mlp": "mlp",
    "ensemble": "ensemble",
}

ALGO_DISPLAY_NAMES = {
    "linear_regression": "1. Linear Regression",
    "ridge": "2. Ridge Regression",
    "lasso": "3. Lasso Regression",
    "svr": "4. SVR",
    "random_forest": "5. Random Forest",
    "knn": "6. KNN",
    "lightgbm": "7. LightGBM",
    "xgboost": "8. XGBoost",
    "adaboost": "9. AdaBoost",
    "bagging": "10. Bagging",
    "mlp": "11. MLP",
    "ensemble": "12. Ensemble",
}
