import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from sample_db import build_sample_database
from tools import make_document_search_tool, search_web, make_sql_tool

st.set_page_config(page_title="Agentic RAG Research Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Agentic RAG Research Assistant")
st.caption(
    "An agent that decides for itself whether to search your documents, "
    "search the live web, or query a database — powered by Groq & Gemini Embeddings."
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Sidebar Setup
with st.sidebar:
    st.header("Setup Keys")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get a key at https://console.groq.com/keys",
    )

    google_api_key = st.text_input(
        "Google API Key (for Embeddings)",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Get a key at https://aistudio.google.com/app/apikey",
    )

    st.markdown("---")
    st.subheader("Add your documents")
    st.caption("Upload both your paper and review documents together.")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    process_btn = st.button("Process documents", use_container_width=True)

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.chat_log = []
        st.session_state.agent = None
        st.session_state.retriever = None
        st.rerun()

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


def build_retriever(file_paths, google_key):
    all_documents = []
    for path in file_paths:
        all_documents.extend(load_document(path))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_key,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 8}), len(chunks)


def build_agent(groq_key, retriever=None):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_key,
        temperature=0.2,
    )

    tools = [search_web, make_sql_tool(st.session_state.db_path)]
    if retriever is not None:
        doc_tool = make_document_search_tool(retriever)
        if doc_tool is not None:
            tools.append(doc_tool)

    system_prompt = (
        "You are an active research assistant. When the user asks about uploaded documents, "
        "you MUST call the `search_documents` tool first to inspect the content before answering. "
        "NEVER tell the user to use the search tool themselves—YOU are the agent responsible for calling the tool. "
        "Query the document search tool, analyze the returned text from both files, and output "
        "a detailed list of specific changes, feedback points, and recommended edits for the paper."
    )

    return create_react_agent(llm, tools, prompt=system_prompt)


# Document processing logic
if process_btn:
    if not google_api_key:
        st.sidebar.error("Please enter your Google API key for embeddings.")
    elif not groq_api_key:
        st.sidebar.error("Please enter your Groq API key.")
    elif not uploaded_files:
        st.sidebar.error("Please upload your documents.")
    else:
        with st.spinner("Processing both documents..."):
            temp_paths = []
            try:
                for f in uploaded_files:
                    suffix = os.path.splitext(f.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.read())
                        temp_paths.append(tmp.name)

                retriever, num_chunks = build_retriever(temp_paths, google_api_key)
                st.session_state.retriever = retriever
                st.session_state.agent = build_agent(groq_api_key, retriever)
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
    if not groq_api_key or not google_api_key:
        st.warning("Please enter both API keys in the sidebar.")
    else:
        st.session_state.agent = build_agent(groq_api_key, st.session_state.retriever)

        st.session_state.chat_log.append(("user", user_question, []))
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
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

                    raw_answer = messages[-1].content
                    if isinstance(raw_answer, list):
                        clean_parts = []
                        for part in raw_answer:
                            if isinstance(part, dict) and "text" in part:
                                clean_parts.append(part["text"])
                            elif isinstance(part, str):
                                clean_parts.append(part)
                        final_answer = "\n".join(clean_parts)
                    else:
                        final_answer = str(raw_answer)

                    st.markdown(final_answer)
                    if tools_used:
                        st.caption(f"🔧 Tool(s) used: {', '.join(tools_used)}")

                    st.session_state.chat_log.append(("assistant", final_answer, tools_used))
                except Exception as e:
                    error_msg = f"Something went wrong: {e}"
                    st.error(error_msg)
                    st.session_state.chat_log.append(("assistant", error_msg, []))
