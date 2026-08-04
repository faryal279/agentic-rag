import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from sample_db import build_sample_database
from tools import make_document_search_tool, search_web, make_sql_tool

st.set_page_config(page_title="Agentic RAG Research Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Agentic RAG Research Assistant")
st.caption(
    "An agent that decides for itself whether to search your documents, "
    "search the live web, or query a database — and shows its work."
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Sidebar Setup
with st.sidebar:
    st.header("Setup")

    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    st.markdown("---")
    st.subheader("Optional: add your own documents")
    st.caption(
        "The agent already has a sample sales/papers database and live web "
        "search available. Uploading documents adds a third source it can "
        "draw on."
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    process_btn = st.button("Process documents", use_container_width=True)

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.chat_log = []
        st.session_state.agent = None
        st.rerun()

    st.markdown("---")
    with st.expander("Try asking..."):
        st.markdown("""\
- *"What were total units sold for Wireless Mouse across all months?"* → SQL tool
- *"Who is the current president of France?"* → Web search tool
- *"Summarize the key finding in the uploaded document"* → Document tool (after upload)
- *"Which paper in the sample database has the most citations?"* → SQL tool
""")

# Session State Initialization
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "db_path" not in st.session_state:
    st.session_state.db_path = build_sample_database()


def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def build_retriever(file_paths, google_api_key):
    all_documents = []
    for path in file_paths:
        all_documents.extend(load_document(path))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_documents)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 4}), len(chunks)


def build_agent(google_api_key, retriever=None):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        temperature=0.2,
        convert_system_message_to_human=True,
    )

    tools = [search_web, make_sql_tool(st.session_state.db_path)]
    doc_tool = make_document_search_tool(retriever)
    if doc_tool is not None:
        tools.append(doc_tool)

    system_prompt = (
        "You are a research assistant with access to tools: web search, a SQL "
        "database (sales and papers tables), and (if available) a document "
        "search tool over user-uploaded files. Always choose the most "
        "appropriate tool for the question rather than answering from memory "
        "alone when a tool could give a more accurate or current answer. "
        "Keep answers concise and cite which source you used."
    )

    return create_react_agent(llm, tools, prompt=system_prompt)


# Document processing logic
if process_btn:
    if not api_key:
        st.sidebar.error("Please enter your Google API key.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one file.")
    else:
        with st.spinner("Building document retriever..."):
            temp_paths = []
            try:
                for f in uploaded_files:
                    suffix = os.path.splitext(f.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.read())
                        temp_paths.append(tmp.name)

                retriever, num_chunks = build_retriever(temp_paths, api_key)
                st.session_state.retriever = retriever
                st.session_state.agent = build_agent(api_key, retriever)
                st.sidebar.success(f"Ready! Indexed {num_chunks} chunks from {len(uploaded_files)} file(s).")
            except Exception as e:
                st.sidebar.error(f"Failed to process documents: {e}")
            finally:
                for p in temp_paths:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

# Chat interface rendering
for role, text, tool_names in st.session_state.chat_log:
    with st.chat_message(role):
        st.markdown(text)
        if tool_names:
            st.caption(f"🔧 Tool(s) used: {', '.join(tool_names)}")

user_question = st.chat_input("Ask a question...")

if user_question:
    if not api_key:
        st.warning("Please enter your Google API key in the sidebar first.")
    else:
        if st.session_state.agent is None:
            st.session_state.agent = build_agent(api_key, st.session_state.retriever)

        st.session_state.chat_log.append(("user", user_question, []))
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Deciding which tool(s) to use..."):
                try:
                    result = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=user_question)]}
                    )
                    messages = result["messages"]

                    tools_used = []
                    for msg in messages:
                        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                            for call in msg.tool_calls:
                                tools_used.append(call["name"])

                    final_answer = messages[-1].content
                    st.markdown(final_answer)
                    if tools_used:
                        st.caption(f"🔧 Tool(s) used: {', '.join(tools_used)}")

                    st.session_state.chat_log.append(("assistant", final_answer, tools_used))
                except Exception as e:
                    error_msg = f"Something went wrong: {e}"
                    st.error(error_msg)
                    st.session_state.chat_log.append(("assistant", error_msg, []))
