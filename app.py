import os

import streamlit as st

# Streamlit Cloud: copy secret into environment before importing the pipeline
try:
    for key in ("GROQ_API_KEY", "GROQ_MODEL"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

from rag.pipeline import BOT_NAME, DuoBot  # noqa: E402

st.set_page_config(page_title=BOT_NAME, page_icon="🤖")


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_bot():
    return DuoBot()


st.title(f"🤖 {BOT_NAME}")
st.caption("Ask me about Syeda Nazish Fatima and Ayesha Khizar (UMT, Software Engineering).")

with st.sidebar:
    st.header("About")
    st.write(
        f"**{BOT_NAME}** is a RAG chatbot. It retrieves facts from personal documents "
        "and answers with an LLM. If the answer isn't in its data, it says so."
    )
    st.markdown("**People**\n- Syeda Nazish Fatima\n- Ayesha Khizar")
    show_sources = st.checkbox("Show retrieved sources", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("**Try asking**\n- What are Nazish's skills?\n- Tell me about her internship\n"
                "- What is Ayesha's hobby?\n- What is Nazish's blood group?")

try:
    bot = load_bot()
except Exception as e:
    st.error(f"Could not start the bot: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: role, content, sources

if not st.session_state.messages:
    st.chat_message("assistant").write(
        f"Hi! I'm {BOT_NAME}. Ask me anything about Nazish or Ayesha."
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if show_sources and m.get("sources"):
            with st.expander("Sources"):
                st.caption(m["sources"])

if prompt := st.chat_input("Ask a question..."):
    st.chat_message("user").write(prompt)
    history = [(m["role"], m["content"]) for m in st.session_state.messages]
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, docs, standalone, route = bot.ask(prompt, history)
            except Exception as e:
                answer, docs, standalone, route = f"Sorry, something went wrong: {e}", [], prompt, "-"
        st.write(answer)
        sources = f"Router: {route} | Search query: {standalone}\n\n" + "\n\n".join(
            f"- ({d.metadata['source']}) {d.page_content[:200]}..." for d in docs
        )
        if show_sources and docs:
            with st.expander("Sources"):
                st.caption(sources)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
