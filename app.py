import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils import load_dataset
from graph import build_graph
from pdf_export import generate_pdf

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("🧠 AI Data Analyst")

# ── Session state defaults ────────────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "df" not in st.session_state:
    st.session_state.df = None
if "history" not in st.session_state:
    st.session_state.history = []
if "llm" not in st.session_state:
    st.session_state.llm = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── AI Provider config ────────────────────────────────────────────────────
    st.header("🔑 AI Provider")

    provider = st.selectbox(
        "Select provider",
        ["Google Gemini", "OpenAI (ChatGPT)"],
        key="provider_select",
    )

    if provider == "Google Gemini":
        model_options = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        model = st.selectbox("Model", model_options, key="gemini_model")
        api_key = st.text_input("Gemini API Key", type="password",
                                placeholder="AIza...",
                                help="Get your key at https://aistudio.google.com/app/apikey")
    else:
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        model = st.selectbox("Model", model_options, key="openai_model")
        api_key = st.text_input("OpenAI API Key", type="password",
                                placeholder="sk-...",
                                help="Get your key at https://platform.openai.com/api-keys")

    if st.button("✅ Apply API Key", use_container_width=True):
        if not api_key.strip():
            st.error("Please enter an API key.")
        else:
            try:
                if provider == "Google Gemini":
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(model=model, temperature=0,
                                                google_api_key=api_key.strip())
                else:
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(model=model, temperature=0,
                                    api_key=api_key.strip())
                st.session_state.llm = llm
                st.session_state.history = []   # reset chat on key change
                st.session_state.pdf_bytes = None
                st.success(f"Connected to {provider}!")
            except Exception as e:
                st.error(f"Failed to initialise LLM: {e}")

    if st.session_state.llm:
        st.caption(f"✓ Active: {provider} / {model}")

    st.divider()

    # ── Dataset upload ────────────────────────────────────────────────────────
    st.header("Dataset")
    uploaded = st.file_uploader(
        "Upload CSV / Excel / JSON / SQLite (.db)",
        type=["csv", "xlsx", "xls", "json", "db"],
    )
    if uploaded:
        st.session_state.df = load_dataset(uploaded)
        st.success(f"Loaded: {uploaded.name}")

    if st.session_state.df is not None:
        st.write(f"Rows: {st.session_state.df.shape[0]}, "
                 f"Cols: {st.session_state.df.shape[1]}")

# ── Main area ─────────────────────────────────────────────────────────────────
if st.session_state.llm is None:
    st.info("👈 Enter your API key in the sidebar to get started.")
elif st.session_state.df is None:
    st.info("👈 Upload a dataset in the sidebar to get started.")
else:
    st.subheader("Preview")
    st.dataframe(st.session_state.df.head())

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["query"])
        with st.chat_message("assistant"):
            st.write(entry["insight"])
            if entry.get("result") is not None:
                st.write(entry["result"])
            if entry.get("fig") is not None:
                st.plotly_chart(entry["fig"], use_container_width=True)

    if st.session_state.history:
        st.divider()
        dataset_name = uploaded.name if uploaded else "Dataset"
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Building PDF..."):
                    pdf_bytes = generate_pdf(st.session_state.history, dataset_name)
                st.session_state.pdf_bytes = pdf_bytes
        with col2:
            if st.session_state.get("pdf_bytes"):
                st.download_button(
                    label="⬇️ Download PDF",
                    data=st.session_state.pdf_bytes,
                    file_name="analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    query = st.chat_input("Ask a question about your data...")
    if query:
        with st.spinner("Analyzing..."):
            result_state = st.session_state.graph.invoke({
                "df": st.session_state.df,
                "user_query": query,
                "llm": st.session_state.llm,
            })
        st.session_state.history.append({
            "query": query,
            "insight": result_state["insight"],
            "result": result_state["result"],
            "fig": result_state["fig"],
            "code": result_state["code"],
        })
        st.rerun()
