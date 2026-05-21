"""主运行脚本"""
import argparse
import time
import json
from datetime import datetime

from data_loader import DataLoader
from evaluation import evaluate_classification, evaluate_regression, print_metrics
from models.classification import CLASSIFICATION_MODELS, ALGO_DISPLAY_NAMES as CLF_ALGO_NAMES
from models.regression import REGRESSION_MODELS, CLASSIFICATION_TO_REGRESSION


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


def run_autogluon_baseline(task, data_loader, num_bag_folds=0, eval_metric=None):
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
            predictor = TabularPredictor(
                label='target',
                problem_type='binary',
                eval_metric=eval_metric
            )
            fit_kwargs = {"time_limit": 120}
            if num_bag_folds and num_bag_folds > 0:
                fit_kwargs["num_bag_folds"] = num_bag_folds
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


def main():
    parser = argparse.ArgumentParser(description="AutoML Playground - 模型训练与评估")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression", "both"],
        default="both",
        help="任务类型: classification, regression, 或 both"
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
        "--output",
        type=str,
        default="./results",
        help="结果输出目录"
    )
    
    args = parser.parse_args()
    
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
            )
            results.update(baseline)
        all_results["tasks"]["classification"] = results
    
    if args.task in ["regression", "both"]:
        results = run_regression(algo=args.algo, data_loader=data_loader)
        if args.baseline:
            baseline = run_autogluon_baseline("regression", data_loader)
            results.update(baseline)
        all_results["tasks"]["regression"] = results
    
    # 保存结果
    output_file = os.path.join(args.output, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
