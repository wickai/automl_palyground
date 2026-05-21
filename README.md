# AutoML Playground

使用 AutoGluon 对常见机器学习算法进行分类和回归任务的 Demo 集。

## 项目结构

```
.
├── data/                    # 数据集下载目录
├── src/
│   ├── data_loader.py       # 数据下载和加载
│   ├── evaluation.py        # 评估指标
│   ├── models/              # 各算法实现
│   │   ├── classification.py
│   │   └── regression.py
│   └── runner.py            # 主运行脚本
├── notebooks/               # Jupyter notebooks
└── results/                # 实验结果
```

## 数据集

- 分类任务: Kaggle 肿瘤二分类数据集
- 回归任务: Kaggle California House Prices 数据集

## 算法列表

| # | 算法 | 类型 |
|---|------|------|
| 1 | 逻辑回归 | 分类 |
| 2 | 朴素贝叶斯 | 分类 |
| 3 | 支持向量机 | 分类 |
| 4 | 随机森林 | 分类 |
| 5 | KNN | 分类 |
| 6 | LightGBM | 分类 |
| 7 | XGBoost | 分类 |
| 8 | AdaBoost | 分类 |
| 9 | Bagging | 分类 |
| 10 | MLP | 分类 |
| 11 | 集成学习 | 分类 |

## 评估指标

- 分类: 准确率 (Accuracy), 召回率 (Recall), F1 Score, AUC
- 回归: RMSE, MAE, R²

## 使用方法

```bash
# 安装依赖
pip install autogluon lightgbm xgboost catboost scikit-learn pandas numpy

# 下载数据并运行
python src/runner.py --task classification --algo random_forest
python src/runner.py --task regression --algo lightgbm
python src/runner.py --task classification --algo all
```
## tabpfnv2， tabicl
新算法评估
```bash

uv run python src/runner.py \
  --task classification \
  --algo random_forest \
  --baseline \
  --ag-num-bag-folds 5 \
  --ag-eval-metric roc_auc \
  --ag-models tabpfnv2,tabicl \
  --output ./results
```


## Docker
Dockerfile 已创建在 `/Users/user/code/github/automl_palyground/Dockerfile`。

    主要内容：
    - 基于 `python:3.13-slim`
    - 安装 autogluon 1.5.0

    **使用方式：**

    ```bash
    cd /Users/user/code/github/automl_palyground

    # 构建镜像
    docker build -t automl_playground .

    # 运行容器
    docker run --rm automl_playground

    # 或交互式运行
    docker run --rm -it automl_playground bash

    # 自定义运行命令
    docker run --rm automl_playground python src/runner.py --task classification --algo random_forest

    # 挂载本地代码（开发时实时修改）
    docker run --rm -v $(pwd):/workspace automl_playground python src/runner.py --task both --baseline
    ```