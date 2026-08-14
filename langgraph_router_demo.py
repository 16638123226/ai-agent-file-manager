from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
)

# 定义两类工具
@tool
def add(a: int, b: int) -> int:
    """两个数字相加"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """两个数字相乘"""
    return a * b

@tool
def get_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [add, multiply, get_time]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# 节点1：调用模型
def call_model(state):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 节点2：执行工具
def call_tools(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_results = []
    for tc in last_message.tool_calls:
        name = tc["name"]
        args = tc["args"]
        result = tool_map[name].invoke(args)
        tool_results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    
    return {"messages": tool_results}

# 路由函数：判断模型想用什么工具，走不同分支
def route_by_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return END
    
    tool_name = last_message.tool_calls[0]["name"]
    
    # 根据工具名返回不同路径
    if tool_name in ["add", "multiply"]:
        return "math_tools"
    elif tool_name == "get_time":
        return "time_tool"
    return "math_tools"

# 数学工具节点
def call_math_tools(state):
    messages = state["messages"]
    last_message = messages[-1]
    results = []
    for tc in last_message.tool_calls:
        name = tc["name"]
        args = tc["args"]
        result = tool_map[name].invoke(args)
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

# 时间工具节点
def call_time_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    results = []
    for tc in last_message.tool_calls:
        result = get_time.invoke({})
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

# 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("math_tools", call_math_tools)
graph.add_node("time_tool", call_time_tool)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    route_by_tool,
    {
        "math_tools": "math_tools",
        "time_tool": "time_tool",
        END: END
    }
)

# 所有工具节点执行完回到 agent
graph.add_edge("math_tools", "agent")
graph.add_edge("time_tool", "agent")

app = graph.compile()

if __name__ == "__main__":
    # 测试1：数学
    print("=== 测试1：数学 ===")
    result = app.invoke({
        "messages": [
            SystemMessage(content="你是助手，有工具就用工具。"),
            HumanMessage(content="帮我算 15 + 27")
        ]
    })
    print("回答：", result["messages"][-1].content)
    
    print("\n=== 测试2：时间 ===")
    result = app.invoke({
        "messages": [
            SystemMessage(content="你是助手，有工具就用工具。"),
            HumanMessage(content="现在几点了？")
        ]
    })
    print("回答：", result["messages"][-1].content)