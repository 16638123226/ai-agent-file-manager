# AI Agent 文件管家

一个基于 **FastAPI + DeepSeek + RAG** 的全栈 AI Agent，能自主调用工具、操作文件、搜索互联网、查询天气，并拥有三级记忆系统。

## ✨ 功能特性

- **ReAct 循环**：自主推理 + 工具调用，不是写死的 if-else
- **6 个工具**：文件读写、目录浏览、互联网搜索、天气查询、知识库存取
- **三级记忆系统**：
  - 短期记忆（对话历史）
  - 长期记忆（JSON 持久化，跨会话记忆）
  - 知识库（ChromaDB 向量数据库，语义检索）
- **流式输出**：SSE 实现打字机效果
- **浏览器自动化**：Playwright 驱动，真实搜索互联网
- **专业 UI**：侧边栏、折叠思考过程、记忆面板、移动端适配
- **云部署**：Railway 一键部署，公网可访问

## 🛠 技术栈

| 层级 | 技术 |
|---|---|
| 前端 | HTML5 / CSS3 / JavaScript / SSE |
| 后端 | Python / FastAPI / Uvicorn |
| AI | DeepSeek API / Function Calling / ReAct |
| 记忆 | JSON / ChromaDB / Sentence Transformers |
| 浏览器 | Playwright |
| 部署 | Railway / Git |

## 🏗 架构

用户输入 → FastAPI 后端 → Agent 核心循环（ReAct） → 工具系统 → 流式输出 → 前端 UI

## 🚀 快速开始

### 1. 克隆仓库

`git clone https://github.com/16638123226/ai-agent-file-manager.git`

### 2. 安装依赖

`pip install -r requirements.txt`

`playwright install chromium`

### 3. 配置环境变量

创建 `.env` 文件，内容为：

`DEEPSEEK_API_KEY=你的Key`

`HF_ENDPOINT=https://hf-mirror.com`

### 4. 启动

`python server.py`

浏览器打开 `http://localhost:8000`

## 🌐 在线演示

https://selfless-nurturing-production-68fa.up.railway.app

## 📝 使用示例

- "帮我看看目录里有什么文件"
- "读一下 a.txt"
- "帮我创建一个 b.txt，内容是：你好"
- "用搜索工具查一下：北京今天天气"
- "帮我记住：我喜欢喝美式咖啡。来源：用户偏好"
- "我之前说过我喜欢喝什么？"

## 🔒 安全说明

- API Key 通过环境变量管理，不硬编码
- 文件操作限制在当前目录
- 敏感文件通过 `.gitignore` 排除

## 📈 待优化

- [ ] 支持 PDF/Word 文档解析
- [ ] 多会话持久化
- [ ] 多 Agent 协作
- [ ] 用户认证

## 👤 作者

- 石饱饱
- 前端转 AI Agent 开发
- GitHub: https://github.com/16638123226