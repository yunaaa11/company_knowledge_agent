# 1. 使用轻量级的 Python 3.12 作为基础镜像
FROM python:3.12-slim

# 2. 设置容器内的工作目录
WORKDIR /app

# 3. 安装必要的系统依赖（添加 unzip 和 curl，后面下载模型需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 4. 先复制依赖文件并安装（利用 Docker 缓存机制加速构建）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 下载 en_core_web_sm 模型（用于英文文本处理）
RUN python -m spacy download en_core_web_sm

# 5. 预下载 flashrank 模型（解决运行时解压失败问题）
# 设置模型缓存目录
ENV FLASHRANK_CACHE_DIR=/root/.cache/flashrank
RUN mkdir -p $FLASHRANK_CACHE_DIR && \
    curl -L https://hf-mirror.com/prithivida/flashrank/resolve/main/ms-marco-MultiBERT-L-12.zip -o /tmp/model.zip && \
    unzip /tmp/model.zip -d $FLASHRANK_CACHE_DIR/ms-marco-MultiBERT-L-12 && \
    rm /tmp/model.zip


# 6. 复制项目所有代码到容器中
COPY . .

# 7. 暴露你的应用端口（假设你之后会跑一个 API，如果是纯脚本测试可忽略）
EXPOSE 8000

#8. 运行你的交互测试脚本
CMD ["python", "test/test_cache.py"]