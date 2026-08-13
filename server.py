# ========== RAG 知识库 ==========
import chromadb
from chromadb.config import Settings

# 创建或连接本地向量数据库
chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = chroma_client.get_collection("agent_knowledge")
except:
    collection = chroma_client.create_collection("agent_knowledge")



from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import openai
import json
import os
import asyncio
# 使用开源模型把文本变成向量
from sentence_transformers import SentenceTransformer
# 使用国内镜像下载模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 延迟加载，避免启动时重复加载
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return embedding_model
app = FastAPI()

# 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 OpenAI 客户端（用你的 DeepSeek Key）
client = openai.OpenAI(
    api_key="sk-839bdb30f68f42508aac170f3ce709e7",  # ← 改成你的 Key
    base_url="https://api.deepseek.com/v1"
)

# ========== 记忆管理 ==========
import time
MEMORY_FILE = "memory.json"

def load_memory():
    """从文件加载记忆，返回记忆列表"""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_memory(memory_list):
    """将记忆列表存入文件"""
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory_list, f, ensure_ascii=False, indent=2)

def add_memory(user_message, agent_response):
    """把一轮对话的关键信息存入记忆"""
    memories = load_memory()
    # 用清晰、对模型友好的格式存储
    memories.append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_message,
        "agent_summary": agent_response[:200]
    })
    # 只保留最近 20 条记忆
    if len(memories) > 20:
        memories = memories[-20:]
    save_memory(memories)
    return memories

# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作目录下的文件内容。当你需要查看某个文件的内容时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名，只填文件名不要包含路径。例如：a.txt"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出当前工作目录下的所有文件和文件夹名称。当你需要知道目录里有什么文件时使用。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "在文件中写入",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "创建一个新文件或在已有文件中写入内容。当你需要创建文件、写入文字内容时使用这个工具。",
                    },
                    "content":{
                         "type": "string",
                         "description": "要写入文件的文字内容"
                    }
                },
                "required": ["filename","content"]
            }
        }
    },
     {
        "type": "function",
        "function": {
            "name": "search_internet",
            "description": "在百度搜索关键词，返回第一条搜索结果的标题。当用户想查找实时信息、新闻、或任何你不知道的内容时使用",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "要在百度搜索的关键词。",
                    },
                },
                "required": ["query"]
            }
        }
    },
     {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息。当用户询问天气、温度、是否下雨等问题时，必须使用这个工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州"
                    }
                },
                "required": ["city"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "add_document",
            "description": "把一段重要的文字内容保存到知识库中，以便将来检索。当用户说'记住'、'保存'、'记录'某段信息时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要保存到知识库的文字内容"
                    },
                    "source": {
                        "type": "string",
                        "description": "这段内容的来源或标题，例如：合同条款、会议纪要、用户偏好"
                    }
                },
                "required": ["content", "source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge",
            "description": "从知识库中检索与问题最相关的内容。当用户询问'我之前存的资料'、'合同里写了什么'、'我的偏好是什么'等问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要检索的问题或关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

async def execute_tool(tool_name, arguments):
    if tool_name == "read_file":
        filename = arguments.get("filename")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"文件 {filename} 的内容是：\n{content}"
        except FileNotFoundError:
            return f"错误：文件 {filename} 不存在"
    elif tool_name == "list_files":
        files = os.listdir(".")
        return "当前目录文件列表：\n" + "\n".join(files)
    elif tool_name == "write_file":
        filename = arguments.get("filename")
        content = arguments.get("content")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"文件 {filename} 写入成功，内容共 {len(content)} 个字符"
        except Exception as e:
            return f"错误：写入文件 {filename} 失败，原因：{str(e)}" 
    elif tool_name == "search_internet":
        from playwright.async_api import async_playwright
        query = arguments.get("query")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # 第一步：搜索
            search_query = query
            if "天气" in query:
                search_query = query + " 实时 预报"
            search_url = f"https://cn.bing.com/search?q={search_query}"
            await page.goto(search_url, wait_until="domcontentloaded")
            
            # 第二步：抓前三条结果的标题和链接
            results = []
            try:
                await page.wait_for_selector("#b_results", timeout=10000)
                items = await page.query_selector_all("#b_results .b_algo h2 a")
                for item in items[:3]:  # 只取前三条
                    title = await item.inner_text()
                    href = await item.get_attribute("href")
                    if title and href:
                        results.append({"title": title, "href": href})
            except:
                pass
            
            if not results:
                await browser.close()
                return f"搜索 '{query}' 没有找到结果"
            
            # 第三步：点进第一条结果，抓取正文
            content_text = ""
            try:
                first_link = results[0]["href"]
                await page.goto(first_link, wait_until="domcontentloaded", timeout=15000)
                # 抓取页面中所有 <p> 标签的文字，拼接起来
                paragraphs = await page.query_selector_all("p")
                texts = []
                for p in paragraphs[:10]:  # 只取前10个段落
                    text = await p.inner_text()
                    if len(text.strip()) > 20:  # 过滤太短的
                        texts.append(text.strip())
                content_text = "\n".join(texts)[:500]  # 最多500字
            except Exception as e:
                content_text = f"(无法抓取详细内容: {str(e)})"
            
            await browser.close()
            
            # 返回摘要
            summary = f"搜索 '{query}' 的结果：\n"
            summary += f"第一条标题：{results[0]['title']}\n"
            summary += f"页面内容摘要：\n{content_text}\n"
            if len(results) > 1:
                summary += f"\n其他相关结果：\n"
                for r in results[1:]:
                    summary += f"- {r['title']}\n"
            
            return summary
    elif tool_name == "get_weather":
        import urllib.request
        import urllib.parse
        city = arguments.get("city")
        try:
            # 使用 wttr.in 免费天气 API，不需要注册和 Key
            encoded_city = urllib.parse.quote(city)
            url = f"https://wttr.in/{encoded_city}?format=j1&lang=zh"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            current = data.get("current_condition", [{}])[0]
            weather_desc = current.get("lang_zh", [current.get("weatherDesc", [{}])[0].get("value", "未知")])[0]
            temp_c = current.get("temp_C", "未知")
            humidity = current.get("humidity", "未知")
            wind_speed = current.get("windspeedKmph", "未知")
            
            result = f"{city}当前天气：\n"
            result += f"- 天气：{weather_desc}\n"
            result += f"- 温度：{temp_c}°C\n"
            result += f"- 湿度：{humidity}%\n"
            result += f"- 风速：{wind_speed}km/h"
            
            return result
        except Exception as e:
            return f"查询 {city} 天气失败：{str(e)}"
    elif tool_name == "add_document":
        content = arguments.get("content")
        source = arguments.get("source", "未标注来源")
        try:
            # 生成向量
            embedding = get_embedding_model().encode(content).tolist()
            # 存入 ChromaDB，用当前时间作为唯一 ID
            import time
            doc_id = f"doc_{int(time.time() * 1000)}"
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{"source": source}]
            )
            return f"已保存到知识库。来源：{source}，内容长度：{len(content)} 字"
        except Exception as e:
            return f"保存失败：{str(e)}"
    
    elif tool_name == "query_knowledge":
        query = arguments.get("query")
        try:
            # 把问题变成向量，搜索最相关的片段
            query_embedding = get_embedding_model().encode(query).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            
            if results and results["documents"] and results["documents"][0]:
                output = "从知识库中找到以下相关内容：\n"
                for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                    source = metadata.get("source", "未知来源")
                    output += f"\n--- 片段 {i}（来源：{source}）---\n{doc}\n"
                return output
            else:
                return "知识库中没有找到相关的内容。"
        except Exception as e:
            return f"检索失败：{str(e)}"
    return "未知工具"
@app.get("/memories")
async def get_memories():
    """返回 Agent 的长期记忆列表，用于前端侧边栏展示"""
    memories = load_memory()
    return memories
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_message = body.get("message", "")
    
    async def generate():
        # 加载记忆
        memories = load_memory()
        
        messages = []
        if memories:
            memory_text = "【系统提示：你的长期记忆】\n"
            memory_text += "你和这个用户之前有过以下对话。请务必记住这些信息，并在回答相关问题时优先引用：\n\n"
            for i, mem in enumerate(memories, 1):
                memory_text += f"记忆{i}：\n"
                memory_text += f"- 用户问：{mem['user']}\n"
                memory_text += f"- 你的回答/操作：{mem['agent_summary']}\n\n"
            memory_text += "【重要】当用户问你“之前”、“上次”、“以前”等涉及历史的问题时，请先搜索你的长期记忆。"
            memory_text += "如果记忆中有相关信息，直接引用记忆来回答。如果记忆中没有，诚实地告诉用户你没有相关记忆。"
            memory_text += "不要编造记忆中没有的信息。"
            messages.append({"role": "system", "content": memory_text})
        
        # 加上当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        total_tokens = 0
        step_count = 0
        agent_response = ""  # 记录 Agent 的最终回答，用于存入记忆          # ← 新增：步数计数
        
        for step in range(10):
            step_count = step + 1  # ← 新增：记录当前步数
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools
            )
            
            # ← 新增：累加 Token 消耗
            usage = response.usage
            total_tokens += usage.total_tokens
            
            msg = response.choices[0].message
            
            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # 推送：Agent 正在调用工具
                yield f"data: {json.dumps({'type': 'tool_call', 'name': tool_name, 'args': arguments}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.1)
                
                # 把 assistant 的工具调用请求加入历史
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                # 执行工具
                result = await execute_tool(tool_name, arguments)
                
                # 推送：工具执行结果
                yield f"data: {json.dumps({'type': 'tool_result', 'content': result}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.1)
                
                # 把工具结果加入历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                
            else:
                # 模型给了最终回答，但我们需要用流式再调一次，逐字产出
                # 先把 messages 里加上模型的最终回答（没有 tool_calls 的 assistant 消息）
                messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
                
                # 用 stream=True 再调一次，让模型逐 Token 产出
                stream = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True
                )
                
                # 逐块推送给前端
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield f"data: {json.dumps({'type': 'stream', 'content': delta.content}, ensure_ascii=False)}\n\n"
                        agent_response += delta.content  # ← 收集完整回答
                        await asyncio.sleep(0.02)  # 小延迟，让打字机效果更明显
                
                # 流式结束后，推送最终统计信息
                yield f"data: {json.dumps({'type': 'done', 'steps': step_count, 'tokens': total_tokens}, ensure_ascii=False)}\n\n"
                 # 保存记忆
                add_memory(user_message, agent_response)
                return
        
        # 超过最大步数
        yield f"data: {json.dumps({'type': 'error', 'content': 'Agent 达到最大步数限制'}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# 托管前端页面
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)