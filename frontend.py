import streamlit as st
from ai_researcher2 import INITIAL_PROMPT, graph, config
import logging
from langchain_core.messages import AIMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Research AI Agent", page_icon="📄")
st.title("📄 Intellexa: An Autonomous AI Agent for Real-Time Research Paper Generation and PDF Synthesis")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("What research topic would you like to explore?")

if user_input:
    logger.info(f"User_input: {user_input}")
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    chat_input = {
        "messages": [{"role": "system", "content": INITIAL_PROMPT}]
        + st.session_state.chat_history
    }

    assistant_box = st.chat_message("assistant")
    placeholder = assistant_box.empty()
    full_response = ""

    try:
        for s in graph.stream(chat_input, config, stream_mode="values"):
            message = s["messages"][-1]

            # Log tool calls only
            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    logger.info(f"Tool call: {tool_call['name']}")

            # ✅ Gemini / LangChain output handling
            if isinstance(message, AIMessage) and message.content:

                if isinstance(message.content, list):
                    text_chunks = []
                    for block in message.content:
                        if isinstance(block, dict) and "text" in block:
                            text_chunks.append(block["text"])
                    text = " ".join(text_chunks)
                else:
                    text = str(message.content)

                full_response += text + " "
                placeholder.write(full_response)

    except ValueError as e:
        if "arXiv" in str(e) or "429" in str(e):
            st.warning("⚠️ arXiv rate limit reached. Please try again after a few minutes.")
        else:
            st.error(f"❌ Error: {e}")

    # ✅ append assistant response ONLY ONCE
    if full_response:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response}
        )
