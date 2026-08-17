# AI Agent 文件管家

一个基于 **FastAPI + LangGraph + DeepSeek + RAG** 的全栈 AI Agent，支持文件操作、互联网搜索、天气查询、知识库检索，并带有人机协同（HITL）安全机制。

## ✨ 核心功能

- **ReAct 推理循环**：自主思考、调用工具、观察结果、继续推理
- **8 个工具**：文件读写、目录浏览、互联网搜索、天气查询、知识库存取、PDF/Word 文档读取
- **三级记忆系统**：
  - 短期记忆（对话历史）
  - 长期记忆（JSON 持久化）
  - 知识库（ChromaDB 向量数据库 + 语义检索）
- **RAG 进阶**：文档智能分段、重排序
- **流式输出**：SSE 打字机效果
- **人机协同（HITL）**：关键操作前暂停确认
- **安全防护**：Prompt Injection 双层防御
- **云部署**：Docker 容器化 + 腾讯云服务器

## 🛠 技术栈

| 层级 | 技术 |
|---|---|
| 前端 | HTML5 / CSS3 / JavaScript / SSE |
| 后端 | Python / FastAPI / Uvicorn |
| AI 框架 | LangChain / LangGraph |
| 大模型 | DeepSeek API |
| 向量数据库 | ChromaDB |
| 文档解析 | pypdf / python-docx |
| 部署 | Docker / Railway / 腾讯云 |

## 🚀 快速开始

### 本地运行

```bash
git clone https://github.com/16638123226/ai-agent-file-manager.git
cd ai-agent-file-manager
pip install -r requirements.txt
```

创建 `.env` 文件：

```
DEEPSEEK_API_KEY=你的Key
HF_ENDPOINT=https://hf-mirror.com
```

启动：

```bash
python server.py
```

浏览器打开 `http://localhost:8000`

### Docker 运行

```bash
docker build -t agent .
docker run -p 8000:8000 --env DEEPSEEK_API_KEY=你的Key --env HF_ENDPOINT=https://hf-mirror.com agent
```

## 🌐 在线演示

- http://124.222.59.215
- http://shibaobao.chat （备案中）

## 📝 使用示例

- "看看目录里有什么文件"
- "读一下 a.txt"
- "帮我创建 b.txt，内容是：你好"
- "搜索：北京天气"
- "帮我记住：我喜欢美式咖啡。来源：用户偏好"
- "我之前说过我喜欢喝什么？"
- "读一下红警运行教程V6.pdf"
- "红警卡菜单怎么解决？"

## 🔒 安全机制

- Prompt Injection 防御：System Prompt 规则 + 输入过滤
- 人机协同：写文件等操作需用户确认
- 危险词拦截：删除类操作直接拒绝
- API Key 管理：环境变量，不硬编码

## 📈 待优化

- [ ] 多会话持久化
- [ ] 多 Agent 协作集成
- [ ] 支持更多文档格式
- [ ] 用户认证

## 👤 作者

- 石饱饱
- 前端转 AI Agent 开发
- GitHub: https://github.com/16638123226