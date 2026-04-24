import streamlit as st
from PyPDF2 import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq  # Google yerine Groq kullanıyoruz!
import os

# --- ARAYÜZ TASARIMI 🌸 ---
st.set_page_config(page_title="Doküman Asistanı", page_icon="🌸")
st.markdown("""
<style>
    .stApp { background-color: #FFF5F7; } 
    h1, h2, h3 { color: #D87093 !important; } 
    p, span, label, div { color: #4A4A4A !important; } 
</style>
""", unsafe_allow_html=True)

st.title("🌸 Kendi Dokümanların ile Sohbet Et")
st.write("PDF, DOCX veya TXT dosyalarını yükle ve yapay zekaya sorular sor!")

# --- 1. METİN ÇIKARMA ---
def get_text_from_files(docs):
    text = ""
    for doc in docs:
        if doc.name.endswith('.pdf'):
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()
        elif doc.name.endswith('.docx'):
            doc_file = docx.Document(doc)
            for para in doc_file.paragraphs:
                text += para.text + "\n"
        elif doc.name.endswith('.txt'):
            text += doc.read().decode("utf-8") + "\n"
    return text

# --- 2. METNİ PARÇALAMA (CHUNKING) ---
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# --- 3. VEKTÖR VERİTABANI (LOKAL HUGGINGFACE) ---
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# --- 4. YAPAY ZEKA İLE CEVAP ÜRETME (GROQ LLM 🚀) ---
def generate_answer(docs, question, api_key):
    os.environ["GROQ_API_KEY"] = api_key
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    Aşağıdaki bağlamı (context) kullanarak soruyu olabildiğince detaylı ve doğru cevapla. 
    Eğer sorunun cevabı bağlamda yoksa "Üzgünüm, yüklenen dokümanlarda bu bilgi bulunmuyor" de, uydurma.
    
    Context:
    {context}
    
    Soru: 
    {question}
    
    Cevap:
    """
    
    # Dünyanın en hızlı modellerinden biri olan Llama-3'ü Groq üzerinden çağırıyoruz
    model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)
    response = model.invoke(prompt)
    return response.content

# --- ANA AKIŞ ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    # Kullanıcıdan artık Groq API Key istiyoruz
    groq_api_key = st.text_input("Groq API Key'ini girin:", type="password")
    st.markdown("[Groq API Key Al](https://console.groq.com/keys)")
    st.divider()
    
    st.header("📂 Doküman Yükle")
    uploaded_files = st.file_uploader("Dosyaları seçin", accept_multiple_files=True)
    
    if st.button("İşle & Yükle 🚀"):
        if not groq_api_key:
            st.error("Önce API Key girilmeli!")
        elif not uploaded_files:
            st.warning("Dosya yüklenmedi!")
        else:
            with st.spinner("Dokümanlar analiz ediliyor... ✨"):
                raw_text = get_text_from_files(uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.success("Sistem hazır! Soru sorabilirsin.")

user_question = st.text_input("Dokümanlarına ne sormak istersin? 💬")

if user_question:
    if "vector_store" not in st.session_state:
        st.warning("Önce sol menüden doküman yüklemelisin.")
    elif not groq_api_key:
        st.error("API Key eksik!")
    else:
        with st.spinner("Cevap üretiliyor..."):
            docs = st.session_state.vector_store.similarity_search(user_question)
            answer = generate_answer(docs, user_question, groq_api_key)
            
            st.markdown("### 🤖 Cevap:")
            st.write(answer)