from crewai import Agent, Task, Crew, Process
from crewai import LLM
from dotenv import load_dotenv
import os

load_dotenv()

llm = LLM(
    model="deepseek/deepseek-chat",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
)

# 创建三个不同角色的 Agent
planner = Agent(
    role="任务规划师",
    goal="把用户的复杂需求拆解成清晰的小步骤",
    backstory="你擅长结构化思维，能把大问题拆成可执行的小任务。",
    llm=llm,
    verbose=True
)

executor = Agent(
    role="执行者",
    goal="按照规划师给的步骤，逐步执行并给出具体内容",
    backstory="你擅长把计划变成实际的成果，执行能力强。",
    llm=llm,
    verbose=True
)

reviewer = Agent(
    role="审查员",
    goal="检查执行者的成果是否完整、准确、符合用户需求",
    backstory="你注重细节，善于发现问题并提出改进建议。",
    llm=llm,
    verbose=True
)

# 定义任务
plan_task = Task(
    description="用户想学做一道红烧肉。请把学习步骤拆解成 3-5 个小步骤。",
    expected_output="一个清晰的步骤列表",
    agent=planner
)

execute_task = Task(
    description="根据规划师的步骤列表，写出每个步骤的具体做法，包括食材和烹饪细节。",
    expected_output="一份完整的红烧肉做法教程",
    agent=executor
)

review_task = Task(
    description="检查执行者写的教程，确认食材、步骤、火候都合理，如有问题请指出。",
    expected_output="审核意见，如果通过就说'通过'",
    agent=reviewer
)

# 创建团队，串行执行
crew = Crew(
    agents=[planner, executor, reviewer],
    tasks=[plan_task, execute_task, review_task],
    process=Process.sequential  # 串行：按顺序一个接一个
)

# 启动
result = crew.kickoff()
print("\n=== 最终结果 ===")
print(result)