# P2P-Project
🌸 Kendi Dokümanların ile Sohbet Et (RAG Tabanlı AI Asistanı)

Bu proje, kullanıcıların farklı formatlardaki (PDF, DOCX, TXT) dokümanları yükleyerek içlerindeki verilerle yapay zeka üzerinden etkileşim kurmasını sağlayan **Retrieval-Augmented Generation (RAG)** tabanlı bir web uygulamasıdır. 

"No Code / Low Code + AI Challenge" kapsamında, uçtan uca bir yapay zeka ürününün hızlı, minimal ve çalışır bir versiyonu (MVP) olarak geliştirilmiştir.

## ✨ Özellikler

- **Çoklu Format Desteği:** PDF, DOCX ve TXT dosyalarını aynı anda okuyup analiz edebilir.
- **Lokal & Hızlı Vektörleştirme (Embedding):** Metinleri anlamlandırmak için API bağımsız, yerel çalışan HuggingFace modeli (`all-MiniLM-L6-v2`) kullanılmıştır.
- **Yüksek Hızlı LLM Entegrasyonu:** Cevap üretimi için dünyanın en hızlı yapay zeka altyapılarından biri olan **Groq API** (Llama-3.1 modeli) tercih edilmiştir.
- **Kullanıcı Dostu Arayüz:** Streamlit ile geliştirilmiş, estetik ve kullanımı son derece basit bir "Sohbet (Chat)" deneyimi sunar.

## 🛠️ Kullanılan Teknolojiler

- **Backend & Arayüz:** Python, Streamlit
- **Doküman İşleme:** PyPDF2, python-docx
- **RAG Mimarisi:** LangChain
- **Vektör Veritabanı:** FAISS (Facebook AI Similarity Search)
- **Embedding:** HuggingFace (`sentence-transformers`)
- **LLM:** Groq API (`llama-3.1-8b-instant`)
