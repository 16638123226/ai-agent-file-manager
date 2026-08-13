from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# 定义工具：写一个函数，加个 @tool 装饰器就行了
@tool
def add(a: int, b: int) -> int:
    """两个数字相加。当你需要计算加法时使用。"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """两个数字相乘。当你需要计算乘法时使用。"""
    return a * b

# 把工具列表告诉模型
tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)

# 测试：问一个需要调工具的问题
messages = [
    SystemMessage(content="你是一个数学助手，会使用工具计算。"),
    HumanMessage(content="帮我算一下 23 乘以 45 等于多少")
]

response = llm_with_tools.invoke(messages)

print("模型想调用的工具：", response.tool_calls)

# 如果模型要调工具，手动执行
if response.tool_calls:
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        if tool_name == "add":
            result = add.invoke(args)
        elif tool_name == "multiply":
            result = multiply.invoke(args)
        print(f"工具执行结果：{result}")