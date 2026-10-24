# 基础镜像：Python 3.10 轻量级版本
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（处理PDF/Word/PPT/Excel需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有代码
COPY . .

# 暴露Streamlit默认端口
EXPOSE 8501

# 容器启动命令
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
