# 企业行政知识库问答系统

一个面向企业制度场景的 Agentic RAG 项目，聚焦“员工能否快速、准确地从公司制度中得到答案”这一真实业务问题。系统支持混合检索、查询改写、重排序、检索质量反思、Redis 缓存，以及基于 RAGAS 的批量评估，适合作为校招 / 社招 AI 工程项目展示。

## 项目亮点

- 基于 `LangGraph` 搭建 `改写 -> 检索 -> 反思 -> 生成` 的工作流，而不是单次检索后直接回答
- 采用 `BM25 + 向量检索 + RRF 融合`，兼顾关键词匹配和语义召回
- 引入 `FlashRank` 重排序，并结合分数阈值、梯度截断、去重策略提升上下文质量
- 支持多轮对话中的查询改写，能处理追问、口语化表达和多子问题场景
- 接入 `Redis` 做高频问题缓存，降低重复问答成本并提升响应速度
- 使用 `RAGAS + 自定义检索指标` 做离线评估，形成“优化 -> 跑分 -> 分析”的闭环

## 技术栈

- 后端编排：`LangGraph`、`LangChain`
- 大模型：`qwen-plus`（OpenAI 兼容接口）
- 向量模型：`BAAI/bge-small-zh-v1.5`
- 混合检索：`rank_bm25` + `Chroma`
- 重排序：`FlashRank`
- API 服务：`FastAPI`
- 缓存：`Redis`
- 评估：`RAGAS`、`Pandas`
- 部署：`Docker`、`Docker Compose`

## 核心流程

```text
用户问题
  -> Query Rewrite（结合 chat_history 改写问题）
  -> Hybrid Retrieval（BM25 + Vector + RRF）
  -> Rerank & Filter（FlashRank + 分数裁剪 + 去重）
  -> Reflection（判断检索质量是否足够，不够则重试）
  -> Answer Generation（严格基于制度文档生成答案）
```

其中，项目不是简单调用 RAG 模板，而是重点对以下环节做了工程化增强：

- **查询改写**：支持追问场景，把“那审批完以后多久能打款？”补全为可检索的完整问题
- **重排序裁剪**：根据 `relevance_score` 做阈值过滤、梯度截断和保底补全，减少噪音文档
- **检索反思**：如果 Top 文档相关性过低，会自动回到改写节点重试
- **生成约束**：系统提示词要求答案注明来源、拒绝幻觉、逐项覆盖多子问题

## 支持能力

- 文档格式：`PDF`、`Word`、`Markdown`、`TXT`
- 问题类型：事实问答、流程题、多条件题、跨文档组合题、口语化提问、多轮追问、无答案拒答
- 输出方式：命令行交互问答、FastAPI 流式接口

## 前置要求

- `Docker Desktop 20.10+`（Windows 建议启用 WSL2）
- 至少 `4GB` 可用内存，推荐 `8GB`
- 一个可用的 OpenAI 兼容 API Key
- 可选：`Hugging Face Token`（拉取模型时使用）

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd bussiness
```

### 2. 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

至少需要配置以下内容：

```ini
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

REDIS_HOST=redis-server
REDIS_PORT=6379

HF_ENDPOINT=https://hf-mirror.com
HF_TOKEN=
```

### 3. 启动服务

```bash
docker-compose up -d
```

容器启动后包含两个核心服务：

- `rag-app`：RAG 应用容器
- `redis-server`：缓存服务

### 4. 运行交互式问答

进入容器：

```bash
docker exec -it bussiness-rag-app-1 bash
```

运行交互脚本：

```bash
python test/test_cache.py
```

示例问题：

- `年假的享受天数与工龄有什么关系？`
- `病假需要提供什么材料？病假工资怎么发？`
- `那审批完以后多久能打款？`

第二次提问相同问题时，会优先命中 Redis 缓存。

## API 接口

项目提供流式问答接口：

- `POST /api/v1/chat`
- 健康检查：`GET /health`

请求体示例：

```json
{
  "query": "北京出差住宿标准是多少？",
  "chat_history": []
}
```

## 评估设计

为了避免只展示“能跑通”，项目增加了一个更贴近真实业务的离线评估流程。

### 评估集覆盖的场景

`test/run_eval.py` 中手工构造了 12 条测试样本，覆盖：

- 单文档事实问答
- 流程步骤题
- 跨制度组合题
- 多条件限制题
- 口语化提问
- 多轮追问改写
- 无答案拒答

### 评估指标

#### RAGAS 四项指标

- `faithfulness`：答案是否忠于检索上下文
- `answer_relevancy`：答案是否回答了问题
- `context_precision`：召回内容是否精准
- `context_recall`：召回内容是否覆盖标准答案关键点

#### 自定义检索与综合指标

除 RAGAS 外，项目还补充了更适合工程分析的指标：

- `retrieval_count`：最终保留文档数
- `unique_source_count`：去重后的来源文档数
- `retrieved_sources`：最终使用了哪些文档
- `top_relevance_score`：Top1 重排分
- `avg_relevance_score`：平均重排分
- `keyword_hit_ratio`：答案对标准答案关键词的覆盖率
- `strict_score`：融合 RAGAS 四项指标与检索质量分的综合分数

这样做的好处是：当回答效果不好时，可以更容易定位问题到底出在“没检索到”、“检索不准”还是“生成没覆盖全”。

### 运行评估

```bash
python test/run_eval.py
```

评估脚本会：

- 批量运行 Agent 获取回答
- 对每条样本计算 RAGAS 和自定义指标
- 输出整体均值和按类别分组统计
- 生成 `reports/eval_report2.csv` 和 `reports/eval_summary2.csv`

## 本项目做过的关键优化

### 1. 检索后质量控制

在 `src/retrieval/reranker.py` 中，对重排结果加入：

- 文档去重
- 最低分阈值过滤
- 分数梯度截断
- 保底补全文档

目标是减少低质量上下文进入生成阶段。

### 2. 检索反思重试

在 `src/agent/reflection.py` 中，如果检索为空，或者前两条文档相关度过低，则触发重试，避免一次低质量召回直接生成答案。

### 3. 面向业务的测试集设计

不是只用自动生成问题，而是增加了手工业务测试集，覆盖追问、跨制度、拒答等真实企业问答场景，更适合面试展示“你是如何定义效果好坏的”。

## 面试时可以怎么讲

你可以围绕下面这个结构介绍项目：

1. **业务问题**：企业制度分散在多个文档里，员工提问口语化、多轮化，直接关键词检索体验差
2. **系统方案**：我做了一个 Agentic RAG，流程是改写、混合检索、重排、反思、生成
3. **核心优化**：重点优化了查询改写、检索后裁剪、检索反思，以及离线评估体系
4. **评估方式**：不是只看主观回答效果，而是用 RAGAS + 自定义检索指标做批量评估
5. **结果复盘**：能分析不同问题类型下的效果差异，并据此继续优化提示词、检索阈值和测试集

## 项目结构

```text
.
├── src/
│   ├── agent/              # LangGraph 工作流、状态与反思逻辑
│   ├── api/                # FastAPI 接口
│   ├── cache/              # Redis 缓存客户端
│   ├── document/           # 文档解析、切分、去重
│   ├── evaluation/         # RAGAS 评估与测试集生成
│   └── retrieval/          # 混合检索、查询改写、重排序
├── data/raw/               # 原始制度文档
├── test/                   # 测试与评估脚本
├── vector_db/              # Chroma 持久化目录（运行后生成）
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## 如何添加自己的文档

1. 将 `PDF / Word / Markdown / TXT` 文件放入 `data/raw/`
2. 执行索引构建脚本：

```bash
docker exec -it bussiness-rag-app-1 python test/run_indexing.py
```

3. 系统会自动完成文档切分、向量索引构建和 BM25 索引更新

## 后续可继续优化的方向

- 增加“优化前 vs 优化后”的对比实验，量化每个改动带来的收益
- 将拒答类样本单独统计，观察是否存在“明明无答案却硬答”的情况
- 引入来源引用展示，让前端直接看到答案对应的制度文档
- 记录平均响应时间、缓存命中率等线上工程指标
- 扩充评估集规模，并增加人工复核，避免只依赖自动指标

## 许可证

MIT License