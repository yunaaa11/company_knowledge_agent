# 企业行政知识库问答系统

一个面向企业制度文档的 Agentic RAG 项目。项目目标不是简单做“文档问答 Demo”，而是模拟企业员工在请假、报销、IT 权限、采购审批、预算付款、信息安全等制度场景下的真实提问，并用可评估、可复盘的工程链路提升回答准确性。

当前系统支持多格式制度文档解析、混合检索、查询改写、意图识别、重排序、检索质量控制、Redis 分层缓存、FastAPI 接口、Streamlit 页面，以及 RAGAS + 自定义指标的离线评估。

## 项目亮点

- 基于 `LangGraph` 搭建 `改写 -> 检索 -> 反思 -> 生成` 的 Agentic RAG 工作流。
- 使用 `BM25 + Chroma 向量检索 + RRF 融合`，兼顾关键词命中和语义召回。
- 加入查询意图识别与制度类型过滤，提升复杂制度场景下的检索稳定性。
- 使用 `FlashRank` 做重排序，并加入分数阈值、断崖截断、去重、来源规则加权等质量控制。
- 支持 Redis 分层缓存，包括检索缓存、回答缓存和版本化缓存键。
- 支持 `PDF / Word / Excel / Markdown / TXT` 等制度文档扩展，适合模拟真实企业知识库。
- 提供复杂评估集、RAGAS 指标、自定义检索指标、分类统计和错误样本分析。

## 技术栈

- 工作流编排：`LangGraph`、`LangChain`
- 大模型：`qwen-plus`，通过 OpenAI 兼容接口调用
- 向量模型：`BAAI/bge-small-zh-v1.5`
- 向量库：`Chroma`
- 关键词检索：`rank_bm25`
- 重排序：`FlashRank`
- API 服务：`FastAPI`
- 前端演示：`Streamlit`
- 缓存：`Redis`
- 评估：`RAGAS`、`Pandas`
- 部署：`Docker`、`Docker Compose`

## 核心流程

```text
用户问题
  -> Query Rewrite：结合历史对话改写口语化问题
  -> Intent Detection：识别请假、报销、IT、采购、预算等制度意图
  -> Hybrid Retrieval：BM25 + 向量检索 + 多查询融合
  -> Rerank & Filter：FlashRank 重排、阈值过滤、去重、来源规则加权
  -> Reflection：判断检索质量是否足够，不足时重试
  -> Answer Generation：严格基于制度文档生成结构化答案
```

## 本次改动内容

本轮改动主要把项目从“能跑通的 RAG”推进到“可解释、可评估、可展示的 AI Agent 应用原型”。

- `src/agent/agent_nodes.py`：改写节点并行生成 rewrite 与 HyDE 查询；检索节点记录子查询统计；生成提示词加强来源约束、拒绝幻觉和多子问题覆盖。
- `src/retrieval/intent.py`：增强意图识别，支持单意图、多意图、上一轮意图继承，适配多轮追问和跨制度问题。
- `src/retrieval/query_rewrite.py`：优化查询改写策略，让口语化追问更容易变成可检索问题。
- `src/retrieval/hybrid_search.py`：增强 BM25 加载失败时的降级逻辑，可从原始文档临时重建轻量 BM25。
- `src/retrieval/reranker.py`：加入多意图过滤、来源规则加权、分数截断、低分过滤、去重和日志安全输出。
- `src/cache/redis_client.py`：增强 Redis 缓存键、缓存 TTL 和 JSON 缓存能力，降低重复问答成本。
- `src/api/routes.py`：增强接口返回信息，方便前端展示来源、缓存命中和检索统计。
- `streamlit_app.py`：增强演示页面，适合面试或答辩时展示完整链路。
- `test/run_eval_complex.py`：新增复杂评估流程，输出总表、分类统计、错误样本和 RAGAS 结果。
- `reports/v2_complex_eval/`：保存复杂评估输出，用于复盘不同策略的效果。

## 项目结构

```text
.
├── src/
│   ├── agent/              # LangGraph 工作流、节点和状态
│   ├── api/                # FastAPI 接口
│   ├── cache/              # Redis 缓存
│   ├── document/           # 文档解析、切分、元数据
│   ├── evaluation/         # 评估指标与数据处理
│   └── retrieval/          # 混合检索、意图识别、查询改写、重排序
├── data/raw/               # 原始制度文档
├── data/raw/enhanced/      # 增强版制度语料
├── test/                   # 索引、问答、评估、消融实验脚本
├── reports/                # 评估报告输出
├── vector_db/              # Chroma 向量库，运行后生成，建议不要上传
├── chroma_db/              # 旧版或本地向量库目录，建议不要上传
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```


## 前置要求

- `Docker Desktop 20.10+`
- 推荐至少 `8GB` 可用内存
- 一个可用的 OpenAI 兼容 API Key
- 可选：`Hugging Face Token`

## 快速开始

### 1. 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

至少需要配置：

```ini
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

REDIS_HOST=redis-server
REDIS_PORT=6379

INDEX_VERSION=v1
PROMPT_VERSION=v1
CACHE_KEY_PREFIX=rag_cache
```

### 2. 启动服务

```bash
docker-compose up -d
```

### 3. 构建或重建索引

当你新增、删除、替换制度文档后，需要重新构建索引：

```bash
docker exec -it bussiness-rag-app-1 python test/run_indexing.py
```

如果文档内容变化较大，建议同时修改 `.env` 中的 `INDEX_VERSION`，例如从 `v1` 改成 `v2`，避免旧缓存继续命中旧答案。

### 4. 运行问答

命令行测试：

```bash
docker exec -it bussiness-rag-app-1 python test/test_cache.py
```

API 服务：

```text
GET  /health
POST /api/v1/chat
```

请求体示例：

```json
{
  "query": "北京出差住宿标准是多少？",
  "chat_history": []
}
```

## 如何替换数据并重新运行

推荐流程：

1. 删除或移走旧的样例文档，只保留你想测试的新文档。
2. 将新的 `PDF / Word / Excel / Markdown / TXT` 文件放入 `data/raw/` 或 `data/raw/enhanced/`。
3. 确认文档中没有真实姓名、手机号、身份证、客户名称、合同金额等敏感信息。
4. 重建索引。
5. 递增 `INDEX_VERSION`。
6. 如提示词有调整，也递增 `PROMPT_VERSION`。
7. 重新运行问答或评估脚本。

如果只是想快速清理本地运行产物，通常需要清理：

- `vector_db/`
- `chroma_db/`
- `data/doc_cache.json`
- `data/doc_store/`
- Redis 缓存
- 旧的 `reports/` 运行结果

清理前请确认这些目录里没有你要保留的实验结果。

## 多格式文档建议

当前 `data/raw/enhanced/` 里主要是 `.md`，这对开发和调试很方便，但真实企业知识库通常会混合 Word、Excel、PDF。

更推荐的做法不是把所有 Markdown 都机械转换成不同格式，而是保留一部分 Markdown 作为可读样例，再选择几份代表性制度转换成不同格式：

- 请假、报销制度：适合 Word，模拟正式制度文档。
- 差旅标准、预算标准、设备领用清单：适合 Excel，模拟表格型规则。
- 信息安全、客户资料保护：适合 PDF，模拟发布版制度。
- README 或说明类材料：继续使用 Markdown。

## 运行评估

普通评估：

```bash
python test/run_eval.py
```

复杂评估：

```bash
python test/run_eval_complex.py
```

消融实验：

```bash
python test/run_ablation.py
python test/run_ablation_complex.py
```

复杂评估报告通常输出到：

```text
reports/v2_complex_eval/
```

常见文件含义：

- `eval_report*.csv`：逐题明细，包括问题、答案、召回来源和指标。
- `eval_summary*.csv`：整体平均分。
- `category_summary*.csv`：按问题类型统计。
- `error_cases*.csv`：低分样本，适合用于复盘。
- `testset_complex*.csv`：复杂测试集。
- `*_checkpoint.csv`、`*_part*.csv`：中间结果。


## 后续改进方向

- 引入更完整的文档权限模型，不同角色只能检索自己有权限的制度。
- 增加来源引用可视化，前端展示答案对应的原文片段。
- 扩大评估集规模，加入人工复核标签，避免只依赖自动指标。
- 记录线上指标，例如响应时间、缓存命中率、拒答率、用户追问率。
- 增加文档版本管理，支持制度更新后的增量索引和缓存自动失效。
- 增加多租户隔离，适配不同部门或不同企业的知识库。
- 对 Excel 表格类制度做专门解析，避免表头、合并单元格、跨行规则丢失。
- 增加灰度发布和回归评估，让每次提示词或检索策略改动都有可量化对比。

## 许可证

MIT License
