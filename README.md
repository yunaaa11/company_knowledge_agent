# 企业行政知识库问答系统 (Company Knowledge Agent)
基于 RAG (Retrieval-Augmented Generation) 的企业内部制度问答助手。支持混合检索（BM25 + 向量）、查询改写、重排序、Redis 缓存及 RAGAS 评估。使用 Docker Compose 一键部署。

## 功能特点
* **📄 支持多种文档格式**：PDF、Word、Markdown、TXT

* **🔍 混合检索**：BM25 关键词 + 向量语义，RRF 融合

* **✍️ 查询改写**：基于对话历史的多轮重写

* **🎯 重排序**：FlashRank 模型精排检索结果

* **💬 交互式问答**：支持连续对话，自动缓存高频问题（Redis）

* **📊 评估工具**：基于 RAGAS 指标的自动化评测

* **🐳 容器化部署**：Docker Compose 一键启动

## 技术栈
* **后端框架**：LangGraph + LangChain

* **向量库**：Chroma（本地持久化）

* **混合检索**：BM25 (rank_bm25) + 向量 (BAAI/bge-small-zh-v1.5)

* **重排序**：FlashRank (ms-marco-MultiBERT-L-12)

* **大语言模型**：通义千问 qwen-plus (兼容 OpenAI API)

* **缓存**：Redis

* **容器化**：Docker, Docker Compose

## 前置要求
* Docker Desktop 20.10+（Windows 用户需启用 WSL2）

* 至少 4GB 可用内存（推荐 8GB）

* 一个有效的 OpenAI API 兼容的 API Key（本示例使用阿里云通义千问，需自行申请）

* （可选）Hugging Face Token（用于下载模型，国内可使用镜像 hf-mirror.com）

## 快速开始
1. 克隆仓
```Bash
git clone https://github.com/your-username/company_knowledge_agent.git
cd company_knowledge_agent
```
2. 配置环境变量
复制示例文件并填写真实值：
```Bash
cp .env.example .env
```
编辑 .env，至少修改以下字段：
```ini
# 必填：你的 API Key（通义千问或其他兼容 OpenAI 的服务）
OPENAI_API_KEY=sk-xxxxxxxxxxxxxx
# 可选：API 基础 URL（默认 https://api.openai.com/v1）
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# 可选：模型名称（默认 qwen-plus）
LLM_MODEL=qwen-plus

# Redis 配置（无需修改，Docker Compose 会使用内部服务名）
REDIS_HOST=redis-server
REDIS_PORT=6379

# Hugging Face 镜像（国内加速）
HF_ENDPOINT=https://hf-mirror.com
# 如果你的模型需要登录，可配置 HF_TOKEN（可选）
HF_TOKEN=
```
注意：REDIS_PASSWORD 和 ENABLE_CACHE 等已在 config.py 中有默认值，一般无需在 .env 中设置。

3. 启动服务
```bash
docker-compose up -d
```
首次启动会构建镜像，自动下载：

* Python 基础镜像

* Python 依赖包（requirements.txt）

* spaCy 英文模型 en_core_web_sm

* FlashRank 重排序模型（约 98MB，已预下载到镜像中）

构建完成后，容器会自动运行：

* rag-app：问答应用（启动后执行 test/test_cache.py 交互脚本）

* redis-server：缓存服务

4. 运行交互式问答
进入应用容器：

```bash
docker exec -it bussiness-rag-app-1 bash
```
在容器内运行：

```bash
python test/test_cache.py
```
你会看到：

```text
--- 正在初始化系统（加载向量库与混合检索） ---
✅ BM25 成功加载，索引内含文档数: 12
✅ 系统就绪！你可以开始提问了（输入 'exit' 退出）
--------------------------------------------------
提问:
```
输入问题（例如“年假的享受天数与工龄有什么关系？”），系统会自动检索文档并生成回答。

第二次问完全相同的问题时会命中 Redis 缓存，返回速度极快。

5. 其他测试脚本
* 单独测试检索 + 重排序：python test/test_retrieval.py

* 测试完整 Agent 流程（单次）：python test/test_agent.py

* 批量评估 RAGAS 指标：python test/run_eval.py（需要先准备好测试集）

* 重建索引（增量更新）：python test/run_indexing.py

## 项目结构
```text
.
├── src/                    # 核心源码
│   ├── agent/              # LangGraph 工作流节点
│   ├── cache/              # Redis 缓存客户端
│   ├── document/           # 文档解析与去重
│   ├── evaluation/         # RAGAS 评估模块
│   └── retrieval/          # 检索、重排、查询改写
├── data/raw/               # 待索引的文档（可自行添加）
├── vector_db/              # Chroma 持久化目录（自动生成）
├── tests/                  # 测试脚本（实际为 test/ 目录）
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example            # 环境变量模板
└── README.md
```

## 如何添加自己的文档
1.将 PDF / Word / Markdown / TXT 文件放入 data/raw/（支持子目录）。

2.进入容器运行索引脚本：
```bash
docker exec -it bussiness-rag-app-1 python test/run_indexing.py
```
3.系统会自动切分文档、构建向量索引和 BM25 索引。

## 许可证
MIT License