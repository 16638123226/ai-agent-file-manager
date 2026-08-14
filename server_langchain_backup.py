from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from dotenv import load_dotenv
import os
import json
import asyncio
import time

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 初始化 LLM ==========
api_key = os.environ.get("DEEPSEEK_API_KEY", "")
print(f"DEBUG: API Key 长度 = {len(api_key)}")  # 打印长度，不打印 Key 本身
if not api_key:
    # 如果环境变量没有，尝试从 .env 读
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    print(f"DEBUG: 从 .env 读取后，API Key 长度 = {len(api_key)}")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
)

# ========== 记忆管理 ==========
MEMORY_FILE = "memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []

def save_memory(memory_list):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory_list, f, ensure_ascii=False, indent=2)

def add_memory(user_message, agent_response):
    memories = load_memory()
    memories.append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_message,
        "agent_summary": agent_response[:200]
    })
    if len(memories) > 20:
        memories = memories[-20:]
    save_memory(memories)

# ========== RAG 知识库 ==========
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = chroma_client.get_collection("agent_knowledge")
except:
    collection = chroma_client.create_collection("agent_knowledge")

embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return embedding_model

# ========== 工具定义 ==========

@tool
def read_file(filename: str) -> str:
    """读取工作目录下的文件内容。当你需要查看某个文本文件的内容时使用。"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"文件 {filename} 的内容是：\n{content}"
    except FileNotFoundError:
        return f"错误：文件 {filename} 不存在"

@tool
def list_files() -> str:
    """列出当前工作目录下的所有文件和文件夹名称。当你需要知道目录里有什么文件时使用。"""
    files = os.listdir(".")
    return "当前目录文件列表：\n" + "\n".join(files)

@tool
def write_file(filename: str, content: str) -> str:
    """创建一个新文件或在已有文件中写入内容。当你需要创建文件、写入文字时使用。"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"文件 {filename} 写入成功，共 {len(content)} 字"
    except Exception as e:
        return f"写入失败：{str(e)}"

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。当用户询问天气、温度时使用。"""
    import urllib.request
    import urllib.parse
    try:
        encoded_city = urllib.parse.quote(city)
        url = f"https://wttr.in/{encoded_city}?format=j1&lang=zh"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
        current = data.get("current_condition", [{}])[0]
        desc = current.get("lang_zh", ["未知"])[0]
        temp = current.get("temp_C", "未知")
        return f"{city}天气：{desc}，温度{temp}°C"
    except Exception as e:
        return f"查询失败：{str(e)}"

@tool
def add_document(content: str, source: str = "未标注") -> str:
    """把重要文字保存到知识库。当用户说'记住'、'保存'时使用。"""
    try:
        embedding = get_embedding_model().encode(content).tolist()
        doc_id = f"doc_{int(time.time() * 1000)}"
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"source": source}]
        )
        return f"已保存到知识库（来源：{source}）"
    except Exception as e:
        return f"保存失败：{str(e)}"

@tool
def query_knowledge(query: str) -> str:
    """从知识库检索与问题最相关的内容。当用户问'之前存的资料'、'文档里写了什么'时，必须先调这个工具，不要重新读文件。"""
    try:
        # 第一步：粗检索，返回 5 个候选片段
        q_emb = get_embedding_model().encode(query).tolist()
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=5  # 先取 5 个候选
        )
        
        if not results or not results["documents"] or not results["documents"][0]:
            return "知识库中没有相关的内容。"
        
        candidates = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            candidates.append({
                "doc": doc,
                "source": meta.get("source", "未知")
            })
        
        # 第二步：用 LLM 重排序，选出最相关的 2 段
        rerank_prompt = f"""用户的问题是："{query}"

以下是知识库检索到的 {len(candidates)} 个候选片段：

"""
        for i, c in enumerate(candidates, 1):
            rerank_prompt += f"片段{i}（来源：{c['source']}）：\n{c['doc'][:500]}\n\n"
        
        rerank_prompt += f"""请从以上片段中，选出最能回答用户问题的 2 个片段。
只返回片段编号，格式如：1,3 或 2,5。不要返回其他内容。"""
        
        rerank_llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0  # 重排序要确定性
        )
        
        rerank_response = rerank_llm.invoke(rerank_prompt)
        chosen_indices = rerank_response.content.strip()
        
        # 解析返回的编号
        import re
        chosen = []
        for match in re.findall(r'\d+', chosen_indices):
            idx = int(match)
            if 1 <= idx <= len(candidates) and idx not in chosen:
                chosen.append(idx)
        
        # 如果解析失败，就用第一个候选
        if not chosen:
            chosen = [1]
        
        # 第三步：返回重排序后选中的片段
        out = "知识库检索结果（重排序后）：\n"
        for idx in chosen[:2]:
            c = candidates[idx - 1]
            out += f"--- 来源：{c['source']} ---\n{c['doc']}\n"
        
        return out
        
    except Exception as e:
        return f"检索失败：{str(e)}"

@tool
def read_document_file(filename: str) -> str:
    """读取PDF或Word文档，自动分段存入知识库。支持.pdf和.docx。读取后内容可供后续 query_knowledge 检索。"""
    try:
        text = ""
        if filename.lower().endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(filename)
            for page in reader.pages[:20]:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        elif filename.lower().endswith(".docx"):
            from docx import Document
            doc = Document(filename)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            return "不支持的文件格式，请用 .pdf 或 .docx"

        if not text.strip():
            return "文档没有可提取的文字"

        # ========== 新的分段逻辑 ==========
        def split_text_smart(text, max_chunk=800, min_chunk=200):
            """按段落和句子智能分段，每个块尽量完整"""
            # 第一步：按换行符拆成段落
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                # 如果当前块加上这个段落还没超过限制，就继续累积
                if len(current_chunk) + len(para) <= max_chunk:
                    current_chunk += para + "\n"
                else:
                    # 当前块已经够长了，先保存
                    if len(current_chunk) >= min_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = para + "\n"
                    else:
                        # 当前块太短，但加上这个段落又超了，需要拆分段落
                        # 把当前块和这个段落合并，然后按句号拆分
                        combined = current_chunk + para
                        sentences = combined.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n")
                        temp = ""
                        for sent in sentences:
                            if len(temp) + len(sent) <= max_chunk:
                                temp += sent
                            else:
                                if temp.strip():
                                    chunks.append(temp.strip())
                                temp = sent
                        if temp.strip():
                            chunks.append(temp.strip())
                        current_chunk = ""
            
            # 处理最后剩余的块
            if current_chunk.strip():
                if len(current_chunk) >= min_chunk:
                    chunks.append(current_chunk.strip())
                else:
                    # 如果最后一块太短，合并到前一块
                    if chunks:
                        chunks[-1] = chunks[-1] + "\n" + current_chunk.strip()
                    else:
                        chunks.append(current_chunk.strip())
            
            # 过滤太短的块
            return [c for c in chunks if len(c) >= 50]

        chunks = split_text_smart(text)
        
        if not chunks:
            # 如果智能分段没分出来，回退到固定字数
            chunks = [text[i:i+800] for i in range(0, len(text), 800)]

        # 每段分别存进知识库
        try:
            embeddings = get_embedding_model().encode(chunks).tolist()
            doc_ids = [f"doc_{int(time.time() * 1000)}_{i}" for i in range(len(chunks))]
            collection.add(
                ids=doc_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"source": filename} for _ in chunks]
            )
            return f"已读取 {filename}，智能分段为 {len(chunks)} 段存入知识库。你现在可以问我关于这个文档的任何问题。"
        except Exception as e:
            return f"读取成功但存入知识库失败：{str(e)}"

    except FileNotFoundError:
        return f"文件 {filename} 不存在"
    except Exception as e:
        return f"读取文档失败：{str(e)}"

tools = [read_file, list_files, write_file, get_weather, add_document, query_knowledge, read_document_file]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

@app.get("/memories")
async def get_memories():
    return load_memory()

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_message = body.get("message", "")

    async def generate():
        memories = load_memory()
        messages = []
        rules = """你是文件管家Agent。严格遵守以下规则：
                1. 当 query_knowledge 工具返回了内容，直接把这些内容展示给用户，绝不要调用 write_file，绝不要说"帮你保存成备忘录"。
                2. 你只需要回答问题，不需要主动保存任何文件，除非用户明确说"保存"、"写文件"。
                3. 不要承诺之后再做，现在就做。"""

        if memories:
            mem_text = "【长期记忆】以下是历史对话摘要：\n"
            for m in memories:
                mem_text += f"- 用户：{m['user']}\n- Agent：{m['agent_summary']}\n"
            rules += "\n\n" + mem_text + "\n当用户问'之前'、'上次'等问题时，优先引用以上记忆。"

        messages.append(SystemMessage(content=rules))
        messages.append(HumanMessage(content=user_message))

        total_tokens = 0
        agent_response = ""

        for step in range(10):
            response = llm_with_tools.invoke(messages)

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                total_tokens += response.usage_metadata.get("total_tokens", 0)

            if response.tool_calls:
                messages.append(response)
                for tc in response.tool_calls:
                    name = tc["name"]
                    args = tc["args"]
                    yield f"data: {json.dumps({'type': 'tool_call', 'name': name, 'args': args}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.1)
                    try:
                        result = tool_map[name].invoke(args)
                    except Exception as e:
                        result = f"工具执行错误：{str(e)}"
                    yield f"data: {json.dumps({'type': 'tool_result', 'content': result}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.1)
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            else:
                final = response.content if isinstance(response.content, str) else str(response.content)
                messages.append(AIMessage(content=final))
                stream = llm.stream(messages)
                for chunk in stream:
                    if chunk.content:
                        yield f"data: {json.dumps({'type': 'stream', 'content': chunk.content}, ensure_ascii=False)}\n\n"
                        agent_response += chunk.content
                        await asyncio.sleep(0.02)
                yield f"data: {json.dumps({'type': 'done', 'steps': step+1, 'tokens': total_tokens}, ensure_ascii=False)}\n\n"
                add_memory(user_message, agent_response)
                return

        yield f"data: {json.dumps({'type': 'error', 'content': '达到最大步数'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)