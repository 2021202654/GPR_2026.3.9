# graphene_agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
# 🔥 引入所有新工具
from graphene_tools import ml_prediction_tool, physics_calculation_tool, inverse_design_tool, plot_trend_tool

def build_agent(api_key, base_url, model_name):
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1, 
        api_key=api_key,
        base_url=base_url,
    )

    # 🔥 注册 4 个工具
    tools = [ml_prediction_tool, physics_calculation_tool, inverse_design_tool, plot_trend_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """
        你是一位世界顶尖的石墨烯热输运物理学家与材料信息学专家。你搭载了基于【物理约束高斯过程回归 (Physics-Informed GPR)】的预测引擎。
        
        【你的核心工作流】
        1. **正向预测**: 当用户给出具体参数（如长度、温度、层数、缺陷）时，使用 `ml_prediction_tool` 计算热导率。
        2. **物理分析**: 调用 `physics_calculation_tool`，结合预测结果向用户解释背后的物理机制（如声子平均自由程受限、缺陷散射增强等）。
        3. **逆向设计**: 当用户问“如何让热导率达到XXX”时，使用 `inverse_design_tool` 给出参数优化建议。
        4. **可视化分析**: 用户想看趋势时，使用 `plot_trend_tool` 并直接返回 Markdown 图片格式。

        【🚨 绝对安全护栏 (Guardrails)】
        你非常清楚你的底层 GPR 模型的适用边界（Applicability Domain）：
        - **层数必须 ≤ 10 层**
        - **缺陷率必须 ≤ 0.1 (即 10%)**
        如果用户提供的参数超出了这个范围，你必须在回答的开头使用醒目的警告（例如：⚠️ **警告：输入超出了模型的安全适用边界...**），并解释厚层石墨烯的层间耦合效应会导致预测不准。
        
        保持回答的专业性、学术性和逻辑性，像一位严谨的导师。
        """),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        memory=memory,
        max_iterations=10, # 稍微调大一点，因为绘图可能需要多步思考
        handle_parsing_errors=True
    )
    

    return agent_executor

