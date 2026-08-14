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

# 定义工具
@tool
def add(a: int, b: int) -> int:
    """两个数字相加"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """两个数字相乘"""
    return a * b

tools = [add, multiply]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# 定义状态（Agent 携带的数据）
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 消息列表，用 add 合并

# 节点1：调用模型
def call_model(state):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}  # 把模型回复加进消息

# 节点2：执行工具
def call_tools(state):
    messages = state["messages"]
    last_message = messages[-1]  # 最后一条消息（带 tool_calls）
    
    tool_results = []
    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        result = tool_map[name].invoke(args)
        tool_results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": tool_results}

# 条件函数：判断下一步去哪
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果模型想调工具 → 去 tools 节点
    if last_message.tool_calls:
        return "tools"
    # 否则结束
    return END

# 构建图
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)

# 设置入口
graph.set_entry_point("agent")

# 添加条件边：agent 之后判断去 tools 还是 END
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)

# tools 执行完回到 agent
graph.add_edge("tools", "agent")

# 编译
app = graph.compile()

# 运行测试
if __name__ == "__main__":
    user_input = "帮我算：(23 + 17) × 3"
    result = app.invoke({
        "messages": [
            SystemMessage(content="你是数学助手，必须用工具计算。"),
            HumanMessage(content=user_input)
        ]
    })
    
    # 打印最终回答
    final_message = result["messages"][-1]
    print("最终回答：", final_message.content)