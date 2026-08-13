from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# 定义工具
@tool
def add(a: int, b: int) -> int:
    """两个数字相加。当需要计算加法时使用。"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """两个数字相乘。当需要计算乘法时使用。"""
    return a * b

# 把工具名映射到函数，执行时用
tool_map = {
    "add": add,
    "multiply": multiply,
}

tools = [add, multiply]
llm_with_tools = llm.bind_tools(tools)

def run_agent(user_input):
    # 初始化对话
    messages = [
        SystemMessage(content="你是一个数学助手。遇到计算题必须使用工具，不要自己算。"),
        HumanMessage(content=user_input)
    ]
    
    for step in range(10):
        print(f"\n=== 第 {step+1} 轮循环 ===")
        
        response = llm_with_tools.invoke(messages)
        
        # 情况1：模型要调工具
        if response.tool_calls:
            print(f"模型想调工具：{response.tool_calls}")
            
            # 把模型的工具调用请求加入历史
            messages.append(response)
            
            # 执行每个工具调用
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                
                # 真正执行工具
                result = tool_map[tool_name].invoke(args)
                print(f"工具执行结果：{result}")
                
                # 把结果作为 ToolMessage 加入历史
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))
                
        # 情况2：模型给出最终回答
        else:
            print(f"最终回答：{response.content}")
            return response.content
    
    print("达到最大步数限制")

if __name__ == "__main__":
    run_agent("帮我算一下：(23 + 17) × 3 等于多少")