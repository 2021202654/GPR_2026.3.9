import streamlit as st
import os
from graphene_agent import build_agent 

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="石墨烯热导率预测助手", 
    page_icon="🧪", 
    layout="wide"
)

st.title("🧪 石墨烯科研助手 (Graphene Agent)")
st.caption("基于物理信息驱动高斯过程回归 (Physics-Informed GPR) 与大型语言模型的智能专家系统")

# --- 2. 关键修复：带缓存的 Agent 获取函数 ---
# 必须加上缓存，否则每次对话都会重置 Agent，导致丢失记忆
@st.cache_resource(show_spinner=False)
def get_agent_executor(api_key, base_url, model_name):
    """
    使用 st.cache_resource 缓存 Agent 对象。
    这样 Agent 实例（以及它内部的 Memory）就会一直存在内存中。
    """
    return build_agent(api_key, base_url, model_name)

# --- 3. 侧边栏配置 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    api_key = st.text_input("输入 API Key", type="password", help="请输入你的豆包/OpenAI API Key")
    base_url = st.text_input("Base URL", value="https://ark.cn-beijing.volces.com/api/v3")
    model_name = st.text_input("模型名称", value="deepseek-v3-2-251201") 
    
    st.divider()
    
    # --- 关键修复：清空历史逻辑 ---
    if st.button("🗑️ 清空对话历史"):
        # 1. 清空 UI 显示的历史
        st.session_state.messages = []
        
        # 2. 🔥 核心修复：显式清空 Agent 脑子里的记忆
        # 即使 Agent 是缓存的，我们也可以调用它的方法来重置状态
        if api_key: # 只有 Key 存在时才能获取 Agent
            try:
                executor = get_agent_executor(api_key, base_url, model_name)
                executor.memory.clear() # <--- 这一行让 Agent 忘记过去
            except:
                pass # 如果 Agent 还没初始化成功，就忽略
            
        st.rerun()

# --- 4. 初始化 Session State (对话历史) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是石墨烯科研助手。我可以帮你预测材料热导率。\n试试问我：预测一下 300K 温度下，缺陷为 0.5% 的石墨烯热导率。"}
    ]

# --- 5. 渲染历史消息 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        # 如果历史消息里存了图片，就把它画出来
        if "image" in msg:
            st.image(msg["image"])

# --- 6. 处理用户输入 ---
if prompt_input := st.chat_input("请输入你的科研问题..."):
    # 6.1 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    st.chat_message("user").write(prompt_input)

    # 6.2 检查 Key 是否存在
    if not api_key:
        st.warning("⚠️ 请先在左侧侧边栏输入 API Key！")
        st.stop()

    # 6.3 Agent 回复
    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent 正在思考并调用工具..."):
                executor = get_agent_executor(api_key, base_url, model_name)
                response = executor.invoke({"input": prompt_input})
                output_text = response["output"]
                
                # 先渲染大模型输出的文字
                st.markdown(output_text, unsafe_allow_html=True)
                
            # 🚀 拦截器：检查工具是否在后台生成了图片
            img_bytes = None
            if os.path.exists("trend_plot.png"):
                st.image("trend_plot.png")  # 立即在界面上显示图表
                
                # 把图片读成二进制存起来，确保刷新页面不丢失
                with open("trend_plot.png", "rb") as f:
                    img_bytes = f.read()
                os.remove("trend_plot.png") # 阅后即焚，保持服务器干净
                
            # 保存到系统记忆中
            msg_data = {"role": "assistant", "content": output_text}
            if img_bytes:
                msg_data["image"] = img_bytes # 把图片也塞进记忆里
                
            st.session_state.messages.append(msg_data)
            
        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            st.cache_resource.clear()
