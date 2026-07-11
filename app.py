import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils import load_dataset
from graph import build_graph
from pdf_export import generate_pdf

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("🧠 AI Data Analyst")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "df" not in st.session_state:
    st.session_state.df = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader(
        "Upload CSV / Excel / JSON / SQLite (.db)",
        type=["csv", "xlsx", "xls", "json", "db"]
    )
    if uploaded:
        st.session_state.df = load_dataset(uploaded)
        st.success(f"Loaded: {uploaded.name}")

    if st.session_state.df is not None:
        st.write(f"Rows: {st.session_state.df.shape[0]}, Cols: {st.session_state.df.shape[1]}")


if st.session_state.df is None:
    st.info("Upload a dataset to get started.")
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
            with st.expander("Generated code"):
                st.code(entry["code"], language="python")

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
                "user_query": query
            })
        st.session_state.history.append({
            "query": query,
            "insight": result_state["insight"],
            "result": result_state["result"],
            "fig": result_state["fig"],
            "code": result_state["code"]
        })
        st.rerun()