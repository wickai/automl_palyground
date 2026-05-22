"""主运行脚本"""
import argparse
import time
import json
from datetime import datetime

import pandas as pd

from data_loader import DataLoader
from evaluation import evaluate_classification, evaluate_regression, print_metrics
from financial_fraud import (
    DEFAULT_LABEL_COLUMNS,
    FRAUD_ALGO_DISPLAY_NAMES,
    FRAUD_MODEL_BUILDERS,
    create_fraud_model,
    evaluate_fraud_model,
    find_best_threshold,
    load_financial_fraud_excel,
    split_financial_fraud_data,
)
from models.classification import CLASSIFICATION_MODELS, ALGO_DISPLAY_NAMES as CLF_ALGO_NAMES
from models.regression import REGRESSION_MODELS, CLASSIFICATION_TO_REGRESSION


AUTOGLUON_MODEL_ALIASES = {
    "xgboost": "XGB",
    "xgb": "XGB",
    "tabpfnv2": "REALTABPFN-V2",
    "realtabpfn-v2": "REALTABPFN-V2",
    "tabicl": "TABICL",
}
AUTOGLUON_UNSUPPORTED_MODELS = {
    "adaboost": "AutoGluon 1.5 的内置 Tabular 模型里不包含 AdaBoost，可改用自定义 fraud pipeline 的 `--algo adaboost`。",
}


def build_autogluon_hyperparameters(model_names=None):
    """将命令行模型名映射为 AutoGluon hyperparameters 配置。"""
    if not model_names:
        return None

    hyperparameters = {}
    for raw_name in model_names:
        model_name = raw_name.strip().lower()
        if not model_name:
            continue
        unsupported_reason = AUTOGLUON_UNSUPPORTED_MODELS.get(model_name)
        if unsupported_reason is not None:
            raise ValueError(f"Unsupported AutoGluon model: {raw_name}. {unsupported_reason}")
        ag_model_name = AUTOGLUON_MODEL_ALIASES.get(model_name)
        if ag_model_name is None:
            valid = ", ".join(sorted(AUTOGLUON_MODEL_ALIASES.keys()))
            raise ValueError(f"Unsupported AutoGluon model: {raw_name}. Valid options: {valid}")
        hyperparameters[ag_model_name] = [{}]

    return hyperparameters or None


def run_classification(algo=None, data_loader=None):
    """运行分类任务"""
    print("\n" + "="*60)
    print("分类任务 - Breast Cancer Dataset")
    print("="*60)
    
    X_train, X_test, y_train, y_test = data_loader.load_breast_cancer()
    
    results = {}
    algos = list(CLASSIFICATION_MODELS.keys()) if algo == "all" else [algo]
    
    for algo_key in algos:
        if algo_key not in CLASSIFICATION_MODELS:
            print(f"\nUnknown algorithm: {algo_key}")
            continue
        
        print(f"\n{'-'*40}")
        print(f"Training: {CLF_ALGO_NAMES.get(algo_key, algo_key)}")
        print(f"{'-'*40}")
        
        start_time = time.time()
        model = CLASSIFICATION_MODELS[algo_key]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        train_time = time.time() - start_time
        
        metrics = evaluate_classification(y_test, y_pred, y_prob)
        metrics["train_time"] = train_time
        
        print_metrics("classification", metrics)
        print(f"  Training Time: {train_time:.2f}s")
        
        results[algo_key] = metrics
    
    return results


def run_regression(algo=None, data_loader=None):
    """运行回归任务"""
    print("\n" + "="*60)
    print("回归任务 - California Housing Dataset")
    print("="*60)
    
    X_train, X_test, y_train, y_test = data_loader.load_california_housing()
    
    results = {}
    
    # 获取回归算法列表
    if algo == "all":
        # 映射到所有回归算法
        algos = list(REGRESSION_MODELS.keys())
    elif algo in CLASSIFICATION_TO_REGRESSION:
        algos = [CLASSIFICATION_TO_REGRESSION[algo]]
    else:
        algos = [algo]
    
    REGRESSION_ALGO_NAMES = {
        "linear_regression": "1. Linear Regression",
        "ridge": "2. Ridge",
        "lasso": "3. Lasso",
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
    
    for algo_key in algos:
        if algo_key not in REGRESSION_MODELS:
            print(f"\nUnknown algorithm: {algo_key}")
            continue
        
        print(f"\n{'-'*40}")
        print(f"Training: {REGRESSION_ALGO_NAMES.get(algo_key, algo_key)}")
        print(f"{'-'*40}")
        
        start_time = time.time()
        model = REGRESSION_MODELS[algo_key]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_time = time.time() - start_time
        
        metrics = evaluate_regression(y_test, y_pred)
        metrics["train_time"] = train_time
        
        print_metrics("regression", metrics)
        print(f"  Training Time: {train_time:.2f}s")
        
        results[algo_key] = metrics
    
    return results


def run_autogluon_baseline(task, data_loader, num_bag_folds=0, eval_metric=None, model_names=None):
    """运行 AutoGluon 基线"""
    print("\n" + "="*60)
    print(f"AutoGluon Baseline - {task.upper()}")
    print("="*60)
    
    try:
        from autogluon.tabular import TabularPredictor
        
        if task == "classification":
            X_train, X_test, y_train, y_test = data_loader.load_breast_cancer()
            
            # 合并数据
            train_df = X_train.copy()
            train_df['target'] = y_train.values
            
            start_time = time.time()
            if eval_metric is None:
                eval_metric = "roc_auc"
            hyperparameters = build_autogluon_hyperparameters(model_names)
            predictor = TabularPredictor(
                label='target',
                problem_type='binary',
                eval_metric=eval_metric
            )
            fit_kwargs = {"time_limit": 120}
            if num_bag_folds and num_bag_folds > 0:
                fit_kwargs["num_bag_folds"] = num_bag_folds
            if hyperparameters is not None:
                fit_kwargs["hyperparameters"] = hyperparameters
            predictor.fit(train_df, **fit_kwargs)  # 限制2分钟
            train_time = time.time() - start_time
            
            y_pred = predictor.predict(X_test)
            y_prob = predictor.predict_proba(X_test)
            
            metrics = evaluate_classification(y_test, y_pred, y_prob)
            metrics["train_time"] = train_time
            
            print(f"\nAutoGluon Baseline Results:")
            print_metrics("classification", metrics)
            print(f"  Training Time: {train_time:.2f}s")
            
        else:  # regression
            X_train, X_test, y_train, y_test = data_loader.load_california_housing()
            
            train_df = X_train.copy()
            train_df['target'] = y_train.values
            
            start_time = time.time()
            predictor = TabularPredictor(
                label='target',
                problem_type='regression',
                eval_metric='rmse'
            )
            predictor.fit(train_df, time_limit=120)
            train_time = time.time() - start_time
            
            y_pred = predictor.predict(X_test)
            
            metrics = evaluate_regression(y_test, y_pred)
            metrics["train_time"] = train_time
            
            print(f"\nAutoGluon Baseline Results:")
            print_metrics("regression", metrics)
            print(f"  Training Time: {train_time:.2f}s")
        
        return {"autogluon": metrics}
        
    except ImportError:
        print("AutoGluon not installed. Skipping baseline.")
        return {}
    except Exception as e:
        print(f"AutoGluon error: {e}")
        return {}


def run_financial_fraud(
    algo=None,
    data_path=None,
    skiprows=2,
    label_columns=None,
    primary_label="Vio",
    n_splits=5,
    fold_index=0,
    train_ratio=0.7,
    valid_ratio=0.1,
    test_ratio=0.2,
):
    """运行财务舞弊识别任务。"""
    print("\n" + "=" * 60)
    print("财务舞弊识别任务 - Financial Fraud Dataset")
    print("=" * 60)

    features, labels, metadata = load_financial_fraud_excel(
        file_path=data_path,
        skiprows=skiprows,
        label_columns=label_columns,
    )
    dataset_split = split_financial_fraud_data(
        features=features,
        labels=labels,
        metadata=metadata,
        primary_label=primary_label,
        n_splits=n_splits,
        fold_index=fold_index,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    results = {}
    algos = list(FRAUD_MODEL_BUILDERS.keys()) if algo == "all" else [algo]

    print(
        f"Fold: {dataset_split.fold_index + 1}/{dataset_split.n_splits}, "
        f"Split Ratio: train/validation/test="
        f"{dataset_split.train_ratio:.0%}/{dataset_split.valid_ratio:.0%}/{dataset_split.test_ratio:.0%}"
    )
    print(
        f"Train Samples: {dataset_split.X_train.shape[0]}, "
        f"Validation Samples: {dataset_split.X_valid.shape[0]}, "
        f"Test Samples: {dataset_split.X_test.shape[0]}"
    )
    print(
        f"Primary Label `{primary_label}` Positive Rate - "
        f"Train: {dataset_split.y_train[primary_label].mean():.4f}, "
        f"Validation: {dataset_split.y_valid[primary_label].mean():.4f}, "
        f"Test: {dataset_split.y_test[primary_label].mean():.4f}"
    )

    for algo_key in algos:
        if algo_key not in FRAUD_MODEL_BUILDERS:
            print(f"\nUnknown fraud algorithm: {algo_key}")
            continue

        print(f"\n{'-' * 40}")
        print(f"Training: {FRAUD_ALGO_DISPLAY_NAMES.get(algo_key, algo_key)}")
        print(f"{'-' * 40}")

        start_time = time.time()
        model = create_fraud_model(algo_key)
        model.fit(dataset_split.X_train, dataset_split.y_train, primary_label=primary_label)
        train_time = time.time() - start_time

        metrics = evaluate_fraud_model(
            model=model,
            X_valid=dataset_split.X_valid,
            y_valid=dataset_split.y_valid,
            X_test=dataset_split.X_test,
            y_test=dataset_split.y_test,
            primary_label=primary_label,
        )
        metrics["train_time"] = train_time
        metrics["fold_index"] = dataset_split.fold_index
        metrics["n_splits"] = dataset_split.n_splits
        metrics["train_positive_rate"] = float(dataset_split.y_train[primary_label].mean())
        metrics["validation_positive_rate"] = float(dataset_split.y_valid[primary_label].mean())
        metrics["test_positive_rate"] = float(dataset_split.y_test[primary_label].mean())

        print_metrics("classification", metrics)
        print(f"  Training Time: {train_time:.2f}s")
        print(
            f"  Validation F1/AUC: {metrics['validation_metrics']['f1']:.4f}/"
            f"{metrics['validation_metrics']['auc']:.4f}"
        )
        print(f"  Decision Threshold: {metrics['decision_threshold']:.2f}")
        print("  Per-label F1/AUC:")
        for label_name, label_metrics in metrics["label_metrics"].items():
            auc_value = (
                f"{label_metrics['auc']:.4f}"
                if label_metrics.get("auc") is not None
                else "N/A"
            )
            print(
                f"    {label_name}: F1={label_metrics['f1']:.4f}, "
                f"AUC={auc_value}"
            )

        results[algo_key] = metrics

    return results


def run_autogluon_financial_fraud(
    data_path,
    skiprows=2,
    label_columns=None,
    primary_label="Vio",
    n_splits=5,
    fold_index=0,
    train_ratio=0.7,
    valid_ratio=0.1,
    test_ratio=0.2,
    eval_metric="roc_auc",
    model_names=None,
    time_limit=120,
):
    """运行财务舞弊任务的 AutoGluon 训练，仅针对主标签 `Vio`。"""
    print("\n" + "=" * 60)
    print("财务舞弊识别任务 - AutoGluon")
    print("=" * 60)

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print("AutoGluon not installed. Skipping fraud training.")
        return {}

    features, labels, metadata = load_financial_fraud_excel(
        file_path=data_path,
        skiprows=skiprows,
        label_columns=label_columns,
    )
    dataset_split = split_financial_fraud_data(
        features=features,
        labels=labels,
        metadata=metadata,
        primary_label=primary_label,
        n_splits=n_splits,
        fold_index=fold_index,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    train_df = dataset_split.X_train.reset_index().copy()
    valid_df = dataset_split.X_valid.reset_index().copy()
    test_df = dataset_split.X_test.reset_index().copy()
    train_df["target"] = dataset_split.y_train[primary_label].astype(int).values
    valid_df["target"] = dataset_split.y_valid[primary_label].astype(int).values
    test_df["target"] = dataset_split.y_test[primary_label].astype(int).values

    hyperparameters = build_autogluon_hyperparameters(model_names)
    predictor = TabularPredictor(
        label="target",
        problem_type="binary",
        eval_metric=eval_metric,
    )

    start_time = time.time()
    predictor.fit(
        train_data=train_df,
        tuning_data=valid_df,
        hyperparameters=hyperparameters,
        time_limit=time_limit,
        num_bag_folds=0,
    )
    train_time = time.time() - start_time

    valid_features = valid_df.drop(columns=["target"])
    test_features = test_df.drop(columns=["target"])
    valid_prob = predictor.predict_proba(valid_features)
    test_prob = predictor.predict_proba(test_features)
    decision_threshold = find_best_threshold(valid_df["target"], valid_prob)

    valid_positive_prob = (
        valid_prob.iloc[:, 1].to_numpy() if isinstance(valid_prob, pd.DataFrame) else valid_prob
    )
    test_positive_prob = (
        test_prob.iloc[:, 1].to_numpy() if isinstance(test_prob, pd.DataFrame) else test_prob
    )
    valid_pred = (valid_positive_prob >= decision_threshold).astype(int)
    test_pred = (test_positive_prob >= decision_threshold).astype(int)

    validation_metrics = evaluate_classification(valid_df["target"], valid_pred, valid_positive_prob)
    metrics = evaluate_classification(test_df["target"], test_pred, test_positive_prob)
    metrics["validation_metrics"] = validation_metrics
    metrics["decision_threshold"] = decision_threshold
    metrics["train_time"] = train_time
    metrics["fold_index"] = dataset_split.fold_index
    metrics["n_splits"] = dataset_split.n_splits
    metrics["train_positive_rate"] = float(dataset_split.y_train[primary_label].mean())
    metrics["validation_positive_rate"] = float(dataset_split.y_valid[primary_label].mean())
    metrics["test_positive_rate"] = float(dataset_split.y_test[primary_label].mean())
    metrics["leaderboard"] = predictor.leaderboard(valid_df, silent=True).to_dict(orient="records")

    print(
        f"Fold: {dataset_split.fold_index + 1}/{dataset_split.n_splits}, "
        f"Split Ratio: train/validation/test="
        f"{dataset_split.train_ratio:.0%}/{dataset_split.valid_ratio:.0%}/{dataset_split.test_ratio:.0%}"
    )
    print_metrics("classification", metrics)
    print(f"  Training Time: {train_time:.2f}s")
    print(
        f"  Validation F1/AUC: {metrics['validation_metrics']['f1']:.4f}/"
        f"{metrics['validation_metrics']['auc']:.4f}"
    )
    print(f"  Decision Threshold: {metrics['decision_threshold']:.2f}")

    return {"autogluon": metrics}


def main():
    parser = argparse.ArgumentParser(description="AutoML Playground - 模型训练与评估")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression", "fraud", "both"],
        default="both",
        help="任务类型: classification, regression, fraud, 或 both"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        help="算法名称，如 random_forest, 或 'all' 运行所有算法"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="是否运行 AutoGluon 基线"
    )
    parser.add_argument(
        "--ag-num-bag-folds",
        type=int,
        default=5,
        help="AutoGluon bagging folds；建议>=5（0 表示不启用 bagging）"
    )
    parser.add_argument(
        "--ag-eval-metric",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "f1", "recall", "accuracy", "log_loss"],
        help="AutoGluon 分类任务的主评估指标（用于选模型）"
    )
    parser.add_argument(
        "--ag-models",
        type=str,
        default="",
        help="AutoGluon 模型，逗号分隔；例如 xgboost"
    )
    parser.add_argument(
        "--ag-time-limit",
        type=int,
        default=120,
        help="AutoGluon 训练时间上限（秒）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="结果输出目录"
    )
    parser.add_argument(
        "--fraud-data-path",
        type=str,
        default="/data/wk/code/github/automl_palyground/data/fruade_data/Summary Table-sent-v1（空值取最大值、最小值或者保留；其填充0，“内部控制信息披露指数”补充2024年，填充2025年；“内部控制指数”填充2025年）.xlsx",
        help="财务舞弊 Excel 数据路径"
    )
    parser.add_argument(
        "--fraud-skiprows",
        type=int,
        default=2,
        help="读取财务舞弊 Excel 时跳过的头部行数"
    )
    parser.add_argument(
        "--fraud-split-folds",
        type=int,
        default=5,
        help="财务舞弊任务的分层切分组数"
    )
    parser.add_argument(
        "--fraud-fold-index",
        type=int,
        default=0,
        help="使用第几个切分组，范围从 0 开始"
    )
    parser.add_argument(
        "--fraud-train-ratio",
        type=float,
        default=0.7,
        help="财务舞弊任务训练集比例"
    )
    parser.add_argument(
        "--fraud-valid-ratio",
        type=float,
        default=0.1,
        help="财务舞弊任务验证集比例"
    )
    parser.add_argument(
        "--fraud-test-ratio",
        type=float,
        default=0.2,
        help="财务舞弊任务测试集比例"
    )
    parser.add_argument(
        "--fraud-labels",
        type=str,
        default="Vio,V1,V2,V3,V4,V5",
        help="财务舞弊多标签列，逗号分隔"
    )
    parser.add_argument(
        "--fraud-primary-label",
        type=str,
        default="Vio",
        help="财务舞弊任务重点优化的主标签"
    )
    
    args = parser.parse_args()
    ag_model_names = [name.strip() for name in args.ag_models.split(",") if name.strip()]
    fraud_label_names = [name.strip() for name in args.fraud_labels.split(",") if name.strip()]
    if not fraud_label_names:
        fraud_label_names = DEFAULT_LABEL_COLUMNS
    
    import os
    os.makedirs(args.output, exist_ok=True)
    
    data_loader = DataLoader()
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "tasks": {}
    }
    
    if args.task in ["classification", "both"]:
        results = run_classification(algo=args.algo, data_loader=data_loader)
        if args.baseline:
            baseline = run_autogluon_baseline(
                "classification",
                data_loader,
                num_bag_folds=args.ag_num_bag_folds,
                eval_metric=args.ag_eval_metric,
                model_names=ag_model_names,
            )
            results.update(baseline)
        all_results["tasks"]["classification"] = results
    
    if args.task in ["regression", "both"]:
        results = run_regression(algo=args.algo, data_loader=data_loader)
        if args.baseline:
            baseline = run_autogluon_baseline("regression", data_loader)
            results.update(baseline)
        all_results["tasks"]["regression"] = results

    if args.task == "fraud":
        if args.baseline:
            results = run_autogluon_financial_fraud(
                data_path=args.fraud_data_path,
                skiprows=args.fraud_skiprows,
                label_columns=fraud_label_names,
                primary_label=args.fraud_primary_label,
                n_splits=args.fraud_split_folds,
                fold_index=args.fraud_fold_index,
                train_ratio=args.fraud_train_ratio,
                valid_ratio=args.fraud_valid_ratio,
                test_ratio=args.fraud_test_ratio,
                eval_metric=args.ag_eval_metric,
                model_names=ag_model_names,
                time_limit=args.ag_time_limit,
            )
        else:
            results = run_financial_fraud(
                algo=args.algo,
                data_path=args.fraud_data_path,
                skiprows=args.fraud_skiprows,
                label_columns=fraud_label_names,
                primary_label=args.fraud_primary_label,
                n_splits=args.fraud_split_folds,
                fold_index=args.fraud_fold_index,
                train_ratio=args.fraud_train_ratio,
                valid_ratio=args.fraud_valid_ratio,
                test_ratio=args.fraud_test_ratio,
            )
        all_results["tasks"]["fraud"] = results
    
    # 保存结果
    output_file = os.path.join(args.output, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
