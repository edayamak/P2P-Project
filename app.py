import streamlit as st
from PyPDF2 import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os

# --- 1. FONKSİYONLAR ---

def get_text_from_files(docs):
    text = ""
    for doc in docs:
        try:
            if doc.name.endswith('.pdf'):
                pdf_reader = PdfReader(doc)
                for page in pdf_reader.pages:
                    content = page.extract_text()
                    if content: text += content
            elif doc.name.endswith('.docx'):
                doc_file = docx.Document(doc)
                for para in doc_file.paragraphs:
                    text += para.text + "\n"
            elif doc.name.endswith('.txt'):
                text += doc.read().decode("utf-8") + "\n"
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")
    return text

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def generate_answer(retrieved_docs, question, api_key):
    # API Key'i doğrudan model tanımlarken veriyoruz (Daha güvenli)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"Bağlam: {context}\n\nSoru: {question}\n\nCevap:"
    model = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.1)
    return model.invoke(prompt).content

def generate_summary_and_questions(text, api_key):
    # API Key hatasını engellemek için doğrudan buraya paslıyoruz
    model = ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.5)
    prompt = f"Aşağıdaki metni özetle ve 3 adet örnek soru çıkar:\n\n{text[:4000]}"
    return model.invoke(prompt).content

# --- 2. ARAYÜZ TASARIMI ---

st.set_page_config(page_title="Doküman Asistanı", page_icon="🌸", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #FFF5F7; } 
    h1, h2, h3 { color: #D87093 !important; } 
    p, span, label, div { color: #4A4A4A !important; } 
    .stButton>button { background-color: #D87093; color: white; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"words": 0, "chunks": 0}

# --- 3. SIDEBAR ---

with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    # API Key'i temizle (trim) özelliğini ekledik
    groq_api_key = st.text_input("API Key:", type="password").strip()
    uploaded_files = st.file_uploader("Dosyaları Yükle", accept_multiple_files=True)
    
    c1, c2 = st.columns(2)
    with c1: train_btn = st.button("Eğit 🧠")
    with c2: clear_btn = st.button("Sil 🗑️")

    if clear_btn:
        st.session_state.messages = []
        st.session_state.stats = {"words": 0, "chunks": 0}
        st.rerun()

    if train_btn:
        if not groq_api_key: st.warning("API Key eksik!")
        elif uploaded_files:
            with st.spinner("İşleniyor... ✨"):
                raw_text = get_text_from_files(uploaded_files)
                st.session_state.raw_text_content = raw_text 
                st.session_state.stats["words"] = len(raw_text.split())
                chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_text(raw_text)
                st.session_state.stats["chunks"] = len(chunks)
                st.session_state.vector_store = get_vector_store(chunks)
                st.success("Sistem Hazır!")
        else: st.error("Dosya seçilmedi!")

    if st.session_state.stats["chunks"] > 0:
        st.divider()
        if st.button("Dökümanı Özetle ✨"):
            if not groq_api_key:
                st.error("API Key girmeden özetleyemem!")
            else:
                with st.spinner("Özetleniyor..."):
                    try:
                        summary = generate_summary_and_questions(st.session_state.raw_text_content, groq_api_key)
                        st.session_state.messages.append({"role": "assistant", "content": f"📝 **Özet:**\n\n{summary}"})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Hata oluştu: {e}")
        
        st.subheader("📊 Analiz")
        st.metric("Kelime", st.session_state.stats["words"])
        st.metric("Parça", st.session_state.stats["chunks"])

# --- 4. ANA EKRAN ---

st.title("🌸 Doküman Asistanı")
st.write("PDF, DOCX veya TXT dosyalarını yükle ve sorular sor!")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if user_question := st.chat_input("Sorunu yaz..."):
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    if "vector_store" not in st.session_state:
        st.error("Önce dökümanları eğitmelisin!")
    else:
        with st.spinner("Düşünüyorum..."):
            try:
                docs = st.session_state.vector_store.similarity_search(user_question, k=3)
                answer = generate_answer(docs, user_question, groq_api_key)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("🔍 Kaynaklar"):
                        for d in docs: st.info(d.page_content)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")