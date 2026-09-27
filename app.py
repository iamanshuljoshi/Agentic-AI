import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


# ============================================================
# ORIGINAL LOGIC BELOW — UNCHANGED
# ============================================================

# Step 1 - Building the RAG retrievers

@st.cache_resource(show_spinner="Loading embeddings model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()


def build_retriver(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                               chunk_overlap=100)

    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner="Indexing college documents...")
def get_retrievers():
    acedemic_retriever = build_retriver("academics_handbook.pdf")
    fee_retriever = build_retriver("fee_structure.pdf")
    return acedemic_retriever, fee_retriever


acedemic_retriever, fee_retriever = get_retrievers()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)


# step2 - State

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# Step 3 - Nodes generation

def classifier_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = acedemic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


# step 4 - router function

def route_query(state: State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


# step 5 - Building the graph

@st.cache_resource(show_spinner=False)
def get_app():
    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    # edges

    graph.add_edge(START, "classifier")

    graph.add_conditional_edges(
        "classifier", route_query
    )

    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    return graph.compile()


app = get_app()

# ============================================================
# ORIGINAL LOGIC ABOVE — UNCHANGED
# (step 6, the CLI run loop, is replaced below with a Streamlit UI
#  that drives the same `app.invoke(...)` call)
# ============================================================


# ---------------- Streamlit UI ----------------

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    /* ---- Base app background: deep ocean gradient ---- */
    .stApp {
        max-width: 900px;
        margin: 0 auto;
        background: radial-gradient(circle at 20% 0%, #16324f 0%, #0b1e33 45%, #071624 100%);
    }

    /* ---- Sidebar: brushed silver panel ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #e8edf2 0%, #c7d2dc 45%, #aab8c4 100%);
        border-right: 1px solid rgba(255,255,255,0.4);
        box-shadow: 4px 0 18px rgba(0,0,0,0.25);
    }
    section[data-testid="stSidebar"] * { color: #14283b !important; }
    section[data-testid="stSidebar"] .stButton button {
        background: linear-gradient(145deg, #2f6f9e, #14405e);
        color: #f3f8fc !important;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(20,64,94,0.5), inset 0 1px 1px rgba(255,255,255,0.25);
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(20,64,94,0.6);
    }

    /* ---- Header banner: glossy silver-to-ocean-blue 3D bar ---- */
    .programme-banner {
        background: linear-gradient(135deg, #cfd9e3 0%, #8fa9bf 25%, #1e5f8c 65%, #0b3554 100%);
        color: #f5fafd;
        padding: 0.85rem 1.2rem;
        border-radius: 16px;
        margin-bottom: 1.4rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-align: center;
        box-shadow:
            0 10px 22px rgba(6, 30, 51, 0.55),
            inset 0 1px 1px rgba(255,255,255,0.45),
            inset 0 -2px 4px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.25);
        text-shadow: 0 1px 2px rgba(0,0,0,0.35);
    }

    /* ---- Chat bubbles: frosted glass over deep ocean ---- */
    .stChatMessage {
        border-radius: 16px;
        padding: 0.3rem 0.4rem;
        margin-bottom: 0.4rem;
        background: linear-gradient(160deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
        backdrop-filter: blur(6px);
        border: 1px solid rgba(163, 196, 224, 0.25);
        box-shadow: 0 6px 16px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08);
    }
    div[data-testid="stChatMessageContent"] p {
        font-size: 0.98rem;
        color: #e7f1f8;
    }

    /* User bubble: ocean blue metallic */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(160deg, #1c5478, #0c2c44);
        border: 1px solid rgba(140, 190, 220, 0.35);
    }

    /* Assistant bubble: brushed silver-blue */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background: linear-gradient(160deg, rgba(203,217,229,0.14), rgba(120,150,170,0.06));
        border: 1px solid rgba(203, 217, 229, 0.3);
    }

    /* ---- Chat input: glossy silver capsule ---- */
    div[data-testid="stChatInput"] {
        background: linear-gradient(145deg, #dfe7ee, #aebccb);
        border-radius: 999px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.4), inset 0 1px 1px rgba(255,255,255,0.6);
        border: 1px solid rgba(255,255,255,0.35);
    }
    div[data-testid="stChatInput"] textarea {
        color: #10263b !important;
    }

    /* ---- Radio buttons in sidebar: subtle depth ---- */
    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.35);
        border-radius: 10px;
        padding: 0.3rem 0.6rem;
        margin-bottom: 0.3rem;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.08);
        transition: background 0.15s ease;
    }
    div[role="radiogroup"] label:hover { background: rgba(255,255,255,0.6); }

    /* ---- Headings ---- */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #dfeaf3;
        text-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "student_programme" not in st.session_state:
    st.session_state.student_programme = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {"role": "human"/"ai", "content": str}

with st.sidebar:
    st.title("🎓 College Assistant")
    st.caption("Ask about academics, fees, or just say hi.")

    programme_map = {
        "BCA": "BCA",
        "BBA": "BBA",
        "B.Com (H)": "B.Com (H)",
    }

    selected = st.radio(
        "Which programme are you in?",
        list(programme_map.keys()),
        index=list(programme_map.keys()).index(st.session_state.student_programme)
        if st.session_state.student_programme in programme_map
        else 0,
    )
    st.session_state.student_programme = programme_map[selected]

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by LangGraph + Groq (gpt-oss-120b)")

st.markdown(
    f'<div class="programme-banner">You\'re set as a {st.session_state.student_programme} student</div>',
    unsafe_allow_html=True,
)

# Render chat history
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "human" else "assistant"
    avatar = "🧑‍🎓" if role == "user" else "🎓"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask about attendance, fees, exams, or anything else...")

if user_query:
    st.session_state.messages.append({"role": "human", "content": user_query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            # Build the message history in the (role, content) tuple form
            # the original graph expects, then invoke the same compiled app.
            history = [
                (m["role"], m["content"]) for m in st.session_state.messages
            ]
            result = app.invoke({
                "programme": st.session_state.student_programme,
                "messages": history,
            })
            answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state.messages.append({"role": "ai", "content": answer})