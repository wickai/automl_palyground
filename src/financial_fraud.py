"""财务舞弊数据读取、预处理与训练流水线。

该模块面向面板型财务数据，重点处理以下问题：
1. Excel 读取时跳过前两行说明文字；
2. 将 ``Stkcd`` 和 ``Accper`` 作为面板索引；
3. 使用公司内按时间正向填充，尽量只利用历史信息补齐缺失；
4. 对数值特征做分位数裁剪、缺失值填补和稳健缩放；
5. 对类别特征做众数填补和独热编码；
6. 使用 5 组分层三段切分，每组保持 train/validation/test=70%/10%/20%。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

from evaluation import evaluate_classification


DEFAULT_LABEL_COLUMNS = ["Vio", "V1", "V2", "V3", "V4", "V5"]
PRIMARY_LABEL = "Vio"
INDEX_COLUMNS = ["Stkcd", "Accper"]
META_COLUMNS = ["ShortName"]
FRAUD_ALGO_DISPLAY_NAMES = {
    "logistic_regression": "1. Logistic Regression",
    "random_forest": "2. Random Forest",
    "lightgbm": "3. LightGBM",
    "xgboost": "4. XGBoost",
    "adaboost": "5. AdaBoost",
}


class QuantileClipper(BaseEstimator, TransformerMixin):
    """按训练集分位数裁剪极端值，降低异常点对模型的影响。"""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    def fit(self, X, _y=None):
        array = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(array, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(array, self.upper, axis=0)
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float).copy()
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)


def _build_lightgbm_estimator():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )


def _build_xgboost_estimator():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
        )
    except ImportError:
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )


def _build_adaboost_estimator():
    return AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
    )


FRAUD_MODEL_BUILDERS: dict[str, Callable[[], BaseEstimator]] = {
    "logistic_regression": lambda: LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
    "lightgbm": _build_lightgbm_estimator,
    "xgboost": _build_xgboost_estimator,
    "adaboost": _build_adaboost_estimator,
}


@dataclass
class FinancialFraudSplit:
    """封装财务舞弊训练所需的 train/validation/test 切分结果。"""

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_valid: pd.DataFrame
    y_test: pd.DataFrame
    metadata_train: pd.DataFrame
    metadata_valid: pd.DataFrame
    metadata_test: pd.DataFrame
    fold_index: int
    n_splits: int
    train_ratio: float
    valid_ratio: float
    test_ratio: float
    label_columns: list[str]
    primary_label: str = PRIMARY_LABEL


class FraudPipelineModel:
    """财务舞弊模型包装器，同时保留主任务与多标签训练能力。"""

    def __init__(self, name: str, estimator_builder: Callable[[], BaseEstimator]):
        self.name = name
        self.estimator_builder = estimator_builder
        self.primary_pipeline: Pipeline | None = None
        self.multilabel_pipeline: Pipeline | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, primary_label: str = PRIMARY_LABEL):
        primary_target = y_train[primary_label].astype(int)
        sample_weight = compute_sample_weight(class_weight="balanced", y=primary_target)

        self.primary_pipeline = build_training_pipeline(X_train, self.estimator_builder())
        fit_kwargs = {}
        if _supports_sample_weight(self.primary_pipeline.named_steps["model"]):
            fit_kwargs["model__sample_weight"] = sample_weight
        self.primary_pipeline.fit(X_train, primary_target, **fit_kwargs)

        # 额外训练一套多标签头，保留对子类型标签的建模能力。
        multilabel_estimator = MultiOutputClassifier(self.estimator_builder())
        self.multilabel_pipeline = build_training_pipeline(X_train, multilabel_estimator)
        self.multilabel_pipeline.fit(X_train, y_train.astype(int))
        return self

    def predict(self, X_test: pd.DataFrame):
        return self.primary_pipeline.predict(X_test)

    def predict_proba(self, X_test: pd.DataFrame):
        return self.primary_pipeline.predict_proba(X_test)

    def predict_multilabel(self, X_test: pd.DataFrame):
        return self.multilabel_pipeline.predict(X_test)

    def predict_multilabel_proba(self, X_test: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.multilabel_pipeline is None:
            return {}
        probabilities = self.multilabel_pipeline.predict_proba(X_test)
        if not isinstance(probabilities, list):
            return {}

        result = {}
        for index, proba in enumerate(probabilities):
            if proba.ndim == 2 and proba.shape[1] > 1:
                result[index] = proba[:, 1]
            else:
                result[index] = np.asarray(proba).reshape(-1)
        return result


def _supports_sample_weight(estimator: BaseEstimator) -> bool:
    """简单检查估计器 fit 是否支持 sample_weight。"""
    fit_method = getattr(estimator, "fit", None)
    if fit_method is None or not hasattr(fit_method, "__code__"):
        return False
    return "sample_weight" in fit_method.__code__.co_varnames


def _coerce_label_frame(frame: pd.DataFrame, label_columns: list[str]) -> pd.DataFrame:
    """将标签转成稳定的 0/1 整数格式。"""
    labels = frame[label_columns].copy()
    labels = labels.apply(pd.to_numeric, errors="coerce").fillna(0)
    labels = (labels > 0).astype(int)
    return labels


def _validate_primary_label_split(name: str, y: pd.Series):
    """确保主标签在每个数据子集内都同时包含正负样本。"""
    if y.nunique() < 2:
        raise ValueError(f"`Vio` 在{name}只有单一类别，无法稳定评估 F1/AUC。")


def _extract_positive_class_probability(y_prob) -> np.ndarray:
    """统一抽取正类概率。"""
    prob_array = y_prob.to_numpy() if hasattr(y_prob, "to_numpy") else np.asarray(y_prob)
    if prob_array.ndim > 1:
        if prob_array.shape[1] > 1:
            return prob_array[:, 1]
        return prob_array.reshape(-1)
    return prob_array


def find_best_threshold(y_true: pd.Series, y_prob, step: float = 0.01) -> float:
    """在验证集上搜索能最大化 F1 的阈值。"""
    positive_prob = _extract_positive_class_probability(y_prob)
    thresholds = np.arange(step, 1.0, step)
    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (positive_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def load_financial_fraud_excel(
    file_path: str,
    skiprows: int = 2,
    label_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取财务舞弊 Excel，并构造多标签任务所需的特征与元数据。"""
    label_columns = label_columns or DEFAULT_LABEL_COLUMNS
    raw_df = pd.read_excel(file_path, skiprows=skiprows)

    required_columns = INDEX_COLUMNS + META_COLUMNS + label_columns
    missing_columns = [column for column in required_columns if column not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Excel 缺少必要列: {missing_columns}")

    raw_df["Accper"] = pd.to_datetime(raw_df["Accper"], errors="coerce")
    raw_df = raw_df.dropna(subset=["Stkcd", "Accper"]).copy()
    raw_df["Stkcd"] = raw_df["Stkcd"].astype(str)
    raw_df = raw_df.sort_values(["Stkcd", "Accper"]).drop_duplicates(["Stkcd", "Accper"], keep="last")

    features = raw_df.drop(columns=label_columns + META_COLUMNS).copy()
    features["FiscalYear"] = features["Accper"].dt.year.astype(int)
    features["FiscalMonth"] = features["Accper"].dt.month.astype(int)
    features = features.set_index(INDEX_COLUMNS)

    # 公司内只做正向填充，模拟真实预测时“只看到过去”的数据可得性。
    temporal_fill_columns = list(features.columns)
    features[temporal_fill_columns] = (
        features.groupby(level="Stkcd", group_keys=False)[temporal_fill_columns].ffill()
    )

    labels = _coerce_label_frame(raw_df, label_columns)
    labels.index = pd.MultiIndex.from_frame(raw_df[INDEX_COLUMNS], names=INDEX_COLUMNS)

    metadata = raw_df[INDEX_COLUMNS + META_COLUMNS].copy()
    metadata["FiscalYear"] = metadata["Accper"].dt.year.astype(int)
    metadata = metadata.set_index(INDEX_COLUMNS)

    return features, labels, metadata


def split_financial_fraud_data(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    primary_label: str = PRIMARY_LABEL,
    n_splits: int = 5,
    fold_index: int = 0,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> FinancialFraudSplit:
    """使用 5 组分层三段切分，保持 train/validation/test=70%/10%/20%。"""
    total_ratio = train_ratio + valid_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError("train/validation/test 比例之和必须为 1。")
    if not 0 <= fold_index < n_splits:
        raise ValueError(f"fold_index 必须在 [0, {n_splits - 1}] 范围内。")

    primary_target = labels[primary_label].astype(int)
    outer_splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_ratio,
        random_state=random_state,
    )
    split_indices = list(outer_splitter.split(features, primary_target))
    train_valid_idx, test_idx = split_indices[fold_index]

    inner_valid_ratio = valid_ratio / (train_ratio + valid_ratio)
    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=inner_valid_ratio,
        random_state=random_state + fold_index,
    )
    inner_primary_target = primary_target.iloc[train_valid_idx]
    relative_train_idx, relative_valid_idx = next(
        inner_splitter.split(features.iloc[train_valid_idx], inner_primary_target)
    )
    train_idx = train_valid_idx[relative_train_idx]
    valid_idx = train_valid_idx[relative_valid_idx]

    X_train = features.iloc[train_idx].copy()
    X_valid = features.iloc[valid_idx].copy()
    X_test = features.iloc[test_idx].copy()
    y_train = labels.iloc[train_idx].copy()
    y_valid = labels.iloc[valid_idx].copy()
    y_test = labels.iloc[test_idx].copy()
    metadata_train = metadata.iloc[train_idx].copy()
    metadata_valid = metadata.iloc[valid_idx].copy()
    metadata_test = metadata.iloc[test_idx].copy()

    _validate_primary_label_split("训练集", y_train[primary_label])
    _validate_primary_label_split("验证集", y_valid[primary_label])
    _validate_primary_label_split("测试集", y_test[primary_label])

    return FinancialFraudSplit(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        metadata_train=metadata_train,
        metadata_valid=metadata_valid,
        metadata_test=metadata_test,
        fold_index=fold_index,
        n_splits=n_splits,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        label_columns=list(labels.columns),
        primary_label=primary_label,
    )


def build_feature_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """根据训练集字段类型构造预处理器。"""
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("clipper", QuantileClipper(lower=0.01, upper=0.99)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_training_pipeline(X: pd.DataFrame, estimator: BaseEstimator) -> Pipeline:
    """拼装完整训练流水线。"""
    preprocessor = build_feature_preprocessor(X)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def create_fraud_model(algo_name: str) -> FraudPipelineModel:
    """根据算法名创建财务舞弊模型。"""
    if algo_name not in FRAUD_MODEL_BUILDERS:
        valid_algos = ", ".join(sorted(FRAUD_MODEL_BUILDERS.keys()))
        raise ValueError(f"Unsupported fraud algorithm: {algo_name}. Valid options: {valid_algos}")
    return FraudPipelineModel(
        name=algo_name,
        estimator_builder=FRAUD_MODEL_BUILDERS[algo_name],
    )


def evaluate_fraud_model(
    model: FraudPipelineModel,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    primary_label: str = PRIMARY_LABEL,
) -> dict:
    """先在验证集调阈值，再在测试集评估主标签 `Vio` 与多标签子任务。"""
    valid_target = y_valid[primary_label].astype(int)
    valid_prob = model.predict_proba(X_valid)
    decision_threshold = find_best_threshold(valid_target, valid_prob)

    valid_positive_prob = _extract_positive_class_probability(valid_prob)
    valid_pred = (valid_positive_prob >= decision_threshold).astype(int)
    validation_metrics = evaluate_classification(valid_target, valid_pred, valid_positive_prob)

    primary_target = y_test[primary_label].astype(int)
    primary_prob = model.predict_proba(X_test)
    primary_positive_prob = _extract_positive_class_probability(primary_prob)
    primary_pred = (primary_positive_prob >= decision_threshold).astype(int)

    metrics = evaluate_classification(primary_target, primary_pred, primary_positive_prob)
    metrics["validation_metrics"] = validation_metrics
    metrics["decision_threshold"] = decision_threshold

    multilabel_pred = model.predict_multilabel(X_test)
    multilabel_proba = model.multilabel_pipeline.predict_proba(X_test)
    per_label_metrics = {primary_label: metrics.copy()}
    for index, label_name in enumerate(y_test.columns):
        if label_name == primary_label:
            continue
        label_true = y_test[label_name].astype(int)
        label_pred = multilabel_pred[:, index]
        label_prob = multilabel_proba[index]
        per_label_metrics[label_name] = evaluate_classification(label_true, label_pred, label_prob)

    metrics["label_metrics"] = per_label_metrics
    return metrics
