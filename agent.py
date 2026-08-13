import openai
import json
import os  # 加在 import json 下面

# 1. 设置你的 API Key (强烈建议用环境变量，Demo 先这样写)
client = openai.OpenAI(
    api_key="sk-839bdb30f68f42508aac170f3ce709e7",
    base_url="https://api.deepseek.com/v1"  # 或者你用的其他兼容接口
)

# 2. 定义 Agent 可以用的工具 (就像定义前端的 Interface)
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取工作目录下的文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "文件名，例如 a.txt"}
                },
                "required": ["filename"]
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出当前工作目录下的所有文件",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# 3. 工具的实际执行逻辑
def execute_tool(tool_name, arguments):
    if tool_name == "read_file":
        # 安全限制：只允许读当前目录的文件
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
    return "未知工具"

# 4. Agent 核心循环
def run_agent(user_prompt):
    messages = [{"role": "user", "content": user_prompt}]
    
    while True:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )
        
        msg = response.choices[0].message
        
            # 情况A：模型想调用工具了
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print(f"🔧 Agent 决定使用工具: {tool_name}, 参数: {arguments}")
            
            # ⚠️ 关键修复：先把模型“我要调用工具”这个请求本身加入历史
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            })
            
            # 执行工具，拿到结果
            result = execute_tool(tool_name, arguments)
            
            # 再把工具的执行结果加入历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
            
        # 情况B：模型给出了最终回答
        else:
            print(f"🤖 Agent 最终回答: {msg.content}")
            break

# 5. 运行！
if __name__ == "__main__":
    # 先在当前目录创建一个测试文件 a.txt
    with open("a.txt", "w", encoding="utf-8") as f:
        f.write("你好，世界！这是 AI Agent 的第一个测试文件。")
    
    run_agent("先看看当前目录有什么文件，然后把第一个文件的内容读给我")