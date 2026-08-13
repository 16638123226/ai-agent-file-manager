from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# 构造对话消息
messages = [
    SystemMessage(content="你是一个友好的AI助手。"),
    HumanMessage(content="你好，介绍一下你自己")
]

# 调用模型
response = llm.invoke(messages)

print("模型回答：", response.content)