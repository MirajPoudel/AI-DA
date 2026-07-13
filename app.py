import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils import load_dataset, QuotaExhaustedError


def _is_quota_error(exc: Exception) -> bool:
    """Walk the exception chain — LangGraph wraps node errors, so
    QuotaExhaustedError may not be the top-level exception type."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, QuotaExhaustedError):
            return True
        # Also catch by message in case it's been stringified through a wrapper
        if "free-tier daily quota is exhausted" in str(exc) or \
           "insufficient_quota" in str(exc):
            return True
        exc = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    return False
from graph import build_graph
from pdf_export import generate_pdf
from sessions import (
    list_sessions, save_session, load_session,
    delete_session, new_session_id, make_title,
)

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("🧠 AI Data Analyst")

# ── Constants ─────────────────────────────────────────────────────────────────
GEMINI_FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

# ── Session state defaults ────────────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "df" not in st.session_state:
    st.session_state.df = None
if "history" not in st.session_state:
    st.session_state.history = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()
if "session_title" not in st.session_state:
    st.session_state.session_title = None
# Store key/provider so fallback can rebuild the LLM with a different model
if "active_provider" not in st.session_state:
    st.session_state.active_provider = None
if "active_api_key" not in st.session_state:
    st.session_state.active_api_key = None
if "active_model" not in st.session_state:
    st.session_state.active_model = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── AI Provider ───────────────────────────────────────────────────────────
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
        key_val = api_key.strip()
        if not key_val:
            st.error("Please enter an API key.")
        else:
            # ── Auto-detect provider from key format ──────────────────
            detected = None
            if key_val.startswith("AIza"):
                detected = "Google Gemini"
            elif key_val.startswith("sk-"):
                detected = "OpenAI (ChatGPT)"

            if detected and detected != provider:
                st.session_state["provider_select"] = detected
                provider = detected
                # Pick a sensible default model for the detected provider
                model = ("gemini-2.0-flash" if detected == "Google Gemini"
                         else "gpt-4o-mini")
                st.info(f"🔍 Auto-detected **{detected}** from your key format — switched automatically.")

            try:
                if provider == "Google Gemini":
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(model=model, temperature=0,
                                                google_api_key=key_val)
                else:
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(model=model, temperature=0,
                                    api_key=key_val)
                st.session_state.llm = llm
                st.session_state.active_provider = provider
                st.session_state.active_api_key = key_val
                st.session_state.active_model = model
                st.session_state.history = []
                st.session_state.session_id = new_session_id()
                st.session_state.session_title = None
                st.session_state.pdf_bytes = None
                st.success(f"Connected to {provider} / {model}!")
            except Exception as e:
                st.error(f"Failed to initialise LLM: {e}")

    if st.session_state.llm:
        st.caption(f"✓ Active: {provider} / {model}")

    st.divider()

    # ── Dataset upload ────────────────────────────────────────────────────────
    st.header("📂 Dataset")
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

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────────
    st.header("💬 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.session_id = new_session_id()
        st.session_state.session_title = None
        st.session_state.pdf_bytes = None
        st.rerun()

    sessions = list_sessions()
    if sessions:
        for s in sessions:
            col_btn, col_del = st.columns([5, 1])
            is_active = s["id"] == st.session_state.session_id
            label = ("▶ " if is_active else "") + s["title"]
            with col_btn:
                if st.button(label, key=f"sess_{s['id']}", use_container_width=True):
                    st.session_state.history = load_session(s["id"])
                    st.session_state.session_id = s["id"]
                    st.session_state.session_title = s["title"]
                    st.session_state.pdf_bytes = None
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{s['id']}"):
                    delete_session(s["id"])
                    if s["id"] == st.session_state.session_id:
                        st.session_state.history = []
                        st.session_state.session_id = new_session_id()
                        st.session_state.session_title = None
                    st.rerun()
    else:
        st.caption("No saved chats yet.")

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
            result_state = None
            last_error = None

            # Build the list of LLMs to try: current one first, then Gemini
            # fallbacks (only when the active provider is Gemini).
            llms_to_try = [st.session_state.llm]

            if st.session_state.active_provider == "Google Gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                current_model = st.session_state.active_model
                tried = {current_model}
                for fallback_model in GEMINI_FALLBACK_MODELS:
                    if fallback_model not in tried:
                        llms_to_try.append(
                            ChatGoogleGenerativeAI(
                                model=fallback_model,
                                temperature=0,
                                google_api_key=st.session_state.active_api_key,
                            )
                        )
                        tried.add(fallback_model)

            original_model = st.session_state.active_model

            for i, llm_attempt in enumerate(llms_to_try):
                try:
                    result_state = st.session_state.graph.invoke({
                        "df": st.session_state.df,
                        "user_query": query,
                        "llm": llm_attempt,
                    })
                    # Success — if we used a fallback, update active LLM & show notice
                    if i > 0:
                        new_model = llm_attempt.model
                        st.session_state.llm = llm_attempt
                        st.session_state.active_model = new_model
                        st.warning(
                            f"⚠️ **{original_model}** quota exhausted — "
                            f"automatically switched to **{new_model}** and completed your request."
                        )
                    last_error = None
                    break  # done
                except Exception as e:
                    if _is_quota_error(e):
                        last_error = e
                        continue  # try next model
                    st.error(f"**Error during analysis:** {e}")
                    st.stop()

            if result_state is None:
                # All models exhausted
                st.error(
                    "**All Gemini free-tier models are quota-exhausted for today** ⚠️\n\n"
                    "**Quick fixes:**\n"
                    "- 🔄 Try again tomorrow (quota resets at midnight Pacific)\n"
                    "- 💳 [Add billing to your Gemini key](https://aistudio.google.com) for unlimited requests\n"
                    "- 🔀 Switch to **OpenAI** in the sidebar (needs billing credits at "
                    "https://platform.openai.com/settings/billing)"
                )
                st.stop()

        entry = {
            "query":   query,
            "insight": result_state["insight"],
            "result":  result_state["result"],
            "fig":     result_state["fig"],
            "code":    result_state["code"],
        }
        st.session_state.history.append(entry)

        # Auto-title from first question, then save
        if st.session_state.session_title is None:
            st.session_state.session_title = make_title(query)

        save_session(
            st.session_state.session_id,
            st.session_state.session_title,
            st.session_state.history,
        )
        st.rerun()
