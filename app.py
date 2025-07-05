import streamlit as st
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import time

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="DocumentMind AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check for required packages
try:
    import pypdf
except ImportError:
    st.error("""
        **Required package missing**: `pypdf` is not installed.
        Please install it by running:
        ```
        pip install pypdf
        ```
        Then restart the app.
    """)
    st.stop()

if "show_app" not in st.session_state:
    st.session_state.show_app = False
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_input_value" not in st.session_state:  # New session state for input clearing
    st.session_state.query_input_value = ""

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        min-height: 100vh;
        margin: 0;
        padding: 0;
    }
    
    .main-title {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    
    .subtitle {
        font-size: 1.4rem;
        text-align: center;
        color: #6b7280;
        margin-bottom: 3rem;
        font-weight: 400;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* LAUNCH BUTTON - SPECIFIC STYLING */
    div[data-testid="stButton"] button[kind="primary"] {
        padding: 1.5rem 4rem !important;
        background: linear-gradient(135deg, #7E57C2 0%, #5E35B1 100%) !important;
        color: white !important;
        border-radius: 50px !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        border: none !important;
        cursor: pointer;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 25px rgba(126, 87, 194, 0.4) !important;
        position: relative;
        overflow: hidden;
        letter-spacing: 1px !important;
        width: fit-content !important;
        margin: 3rem auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-5px) scale(1.03) !important;
        box-shadow: 0 10px 30px rgba(126, 87, 194, 0.6) !important;
        background: linear-gradient(135deg, #8E67D2 0%, #6E45C1 100%) !important;
    }

    /* BACK BUTTON - SPECIFIC STYLING */
    div[data-testid="stButton"] button[kind="secondary"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        background-color: white !important;
        color: #4f46e5 !important;
        border: 1px solid #4f46e5 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        margin: 0 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        min-width: 80px !important;
        height: auto !important;
        line-height: normal !important;
    }

    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #f3f4f6 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    
    .feature-title {
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
        color: #1f2937;
    }
    
    .feature-desc {
        color: #6b7280;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
        text-align: center;
        margin: 0 !important;
        padding: 15px 0 !important;
        color: #6b7280;
        font-size: 0.9rem;
        border-top: 1px solid rgba(0,0,0,0.05);
        z-index: 100;
    }
    
    .footer a {
        margin: 0 12px;
        color: #4f46e5;
        text-decoration: none;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .footer a:hover {
        color: #6366f1;
    }
    
    .footer strong {
        color: #4f46e5;
    }
    
    .upload-area {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border: 2px dashed rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    .chat-container {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin-top: 2rem;
    }
    
    .user-query {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .ai-response {
        background: #f9fafb;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        max-width: 80%;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .query-input {
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease forwards;
    }
    
    .delayed-1 {
        animation-delay: 0.2s;
    }
    
    .delayed-2 {
        animation-delay: 0.4s;
    }
    
    .delayed-3 {
        animation-delay: 0.6s;
    }
    
    @media (max-width: 768px) {
        .main-title {
            font-size: 3rem;
        }
        
        .subtitle {
            font-size: 1.1rem;
        }
        
        div[data-testid="stButton"] button[kind="primary"] {
            padding: 1.2rem 2.5rem !important;
            font-size: 1.5rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- HOME PAGE --------------------
if not st.session_state.show_app:
    st.markdown('<div class="main-title fade-in">DocumentMind AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle fade-in delayed-1">Transform documents into intelligent conversations with our <br> AI-powered analysis platform</div>', unsafe_allow_html=True)
    
    # Features grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="feature-card fade-in delayed-1">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Multi-Format Support</div>
                <div class="feature-desc">Upload PDFs, Word docs, and text files. Our system extracts and understands content from any format.</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card fade-in delayed-2">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Lightning Fast</div>
                <div class="feature-desc">Powered by Groq's ultra-fast LLM inference, get answers in seconds, not minutes.</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card fade-in delayed-3">
                <div class="feature-icon">🔒</div>
                <div class="feature-title">Secure Processing</div>
                <div class="feature-desc">Your documents are processed locally when possible and never stored permanently.</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Launch button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Launch DocumentMind AI",
                            key="launch_btn",
                            use_container_width=True,
                            type="primary"):
            st.session_state.show_app = True
            st.rerun()

# -------------------- MAIN APP --------------------
else:
    # App header with back button
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("← Back",
                            key="back_btn",
                            type="secondary"):
            st.session_state.show_app = False
            st.session_state.file_processed = False
            st.session_state.chat_history = []
            st.session_state.query_input_value = ""  # Clear input when going back
            st.rerun()
    with col2:
       st.markdown('<div class="main-title fade-in">DocumentMind AI</div>', unsafe_allow_html=True)
    
    # Upload section
    uploaded_file = st.file_uploader("Drag and drop or click to browse files",
                                       type=["pdf", "docx", "txt"],
                                       key="file_uploader")
    
    if uploaded_file and not st.session_state.file_processed:
        try:
            # Step 1: Save file
            with st.spinner("📁 Saving uploaded file..."):
                TEMP_DIR = Path("/tmp")  # or Path("temp") for local
                TEMP_DIR.mkdir(exist_ok=True)
                file_path = TEMP_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Step 2: Load document
            with st.spinner("📄 Loading document..."):
                ext = uploaded_file.name.split(".")[-1].lower()
                if ext == "pdf":
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load_and_split()
                elif ext == "docx":
                    loader = Docx2txtLoader(str(file_path))
                    docs = loader.load()
                elif ext == "txt":
                    loader = TextLoader(str(file_path))
                    docs = loader.load()
                else:
                    st.error("Unsupported file format")
                    st.stop()

            # Step 3: Validate content
            if not docs or not any(doc.page_content.strip() for doc in docs):
                st.error("No readable text found in the document")
                st.stop()

            # Step 4: Split into chunks
            with st.spinner("✂️ Splitting document into chunks..."):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=100,
                    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
                )
                chunks = splitter.split_documents(docs)

            if not chunks:
                st.error("Failed to create valid text chunks from document")
                st.stop()

            # Step 5: Generate embeddings
            with st.spinner("🧠 Generating embeddings..."):
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                    model_kwargs={'device': 'cpu'}
                )
                test_embed = embeddings.embed_query("test")

            # Step 6: Build vector store
            with st.spinner("📚 Building document index..."):
                vectorstore = FAISS.from_documents(chunks, embeddings)

            # Step 7: Initialize LLM
            with st.spinner("⚙️ Loading LLM (Groq)..."):
                llm = ChatGroq(
                    api_key=st.secrets["GROQ_API_KEY"],
                    model="llama3-8b-8192",
                    temperature=0.1
                )

            # Step 8: Create RAG QA chain
            with st.spinner("🧩 Creating document Q&A pipeline..."):
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    return_source_documents=True,
                    chain_type="stuff"
                )

            st.session_state.qa_chain = qa_chain
            st.session_state.file_processed = True
            st.session_state.file_name = uploaded_file.name

            st.success(f"✅ {uploaded_file.name} is ready for questions!")

        except Exception as e:
            st.error(f"💥 Error processing document: {str(e)}")
        finally:
            try:
                file_path.unlink()
            except:
                pass

    
    if st.session_state.get("file_processed", False):
        with st.container():
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                        <div class="user-query">
                            {message["content"]}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="ai-response">
                            {message["content"]}
                        </div>
                    """, unsafe_allow_html=True)
            
            # Query input with form
            with st.form(key="query_form"):
                # Create a unique key for the input that changes after submission
                input_key = f"query_input_{len(st.session_state.chat_history)}"
                query = st.text_input(
                    "Ask a question about your document",
                    key=input_key,
                    placeholder="Type your question here...",
                    label_visibility="collapsed"
                )
                submit_button = st.form_submit_button("Ask")
            
            if submit_button and query:
                with st.spinner("Analyzing document..."):
                    try:
                        result = st.session_state.qa_chain.invoke({"query": query})
                        st.session_state.chat_history.append({"role": "user", "content": query})
                        st.session_state.chat_history.append({"role": "ai", "content": result["result"]})
                        st.session_state.last_query = query
                        st.rerun()  # This will clear the input
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")


# -------------------- FOOTER --------------------
st.markdown("""
    <div class="footer">
        © 2025 DocumentMind AI. All rights reserved.
        <br>
        Created by <strong>Manga Sree Rapelli</strong>
        <br>
        <a href="mailto:mangasreerapelli@gmail.com">Email</a>
        <a href="https://github.com/Mangasree" target="_blank">GitHub</a>
        <a href="https://www.linkedin.com/in/mangasree-rapelli/" target="_blank">LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
