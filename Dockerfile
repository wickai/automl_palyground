FROM python:3.13-slim

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt autogluon==1.5.0

# 复制代码
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认运行命令
CMD ["python", "src/runner.py", "--task", "both", "--baseline"]
