import streamlit as st
import pickle
import faiss
import numpy as np
import pandas as pd
import re
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
from dotenv import load_dotenv
import subprocess
import sys
import spacy
import nltk 

# Auto-download spaCy model on cloud
load_dotenv()

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def download_models():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    return "models ready"

download_models()  # call AFTER set_page_config

# ── Load models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    classifier= pickle.load(open('models/best_classifier.pkl',  'rb'))
    embedder= SentenceTransformer('all-MiniLM-L6-v2')
    index= faiss.read_index('models/faiss_index.index')
    index_df= pd.read_csv('models/index_papers.csv')
    return classifier, embedder, index, index_df

classifier, embedder, faiss_index, index_df = load_models()

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ── Helper functions ─────────────────────────────────────────
def classify_paper(text):
    return classifier.predict([text])[0]

def extractive_summary(text, num_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s.split()) > 5]
    if len(sentences) <= num_sentences:
        return text
    try:
        tfidf= TfidfVectorizer(stop_words='english')
        matrix       = tfidf.fit_transform(sentences)
        scores       = np.array(matrix.sum(axis=1)).flatten()
        top_indices  = sorted(np.argsort(scores)[-num_sentences:].tolist())
        return ' '.join([sentences[i] for i in top_indices])
    except:
        return ' '.join(sentences[:num_sentences])

def semantic_search(query, k=3):
    query_vec           = embedder.encode([query]).astype('float32')
    distances, indices  = faiss_index.search(query_vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        paper = index_df.iloc[idx]
        results.append({
            'title'    : paper['title'],
            'category' : paper['main_category'],
            'abstract' : paper['abstract'][:200] + '...',
            'score'    : round(float(1 / (1 + dist)), 4)
        })
    return results

def deep_answer(question, context):
    """Use Groq to answer deep questions about the paper"""
    prompt = f"""You are an expert research paper analyst.
Use ONLY the context below to answer the question precisely.
If the answer is not in the context, say: 'This information is not available in the paper.'

Context:
{context[:3000]}

Question: {question}

Answer:"""
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 500
    )
    return response.choices[0].message.content

def detect_intent(question):
    """Detect what the user wants"""
    q = question.lower()
    if any(w in q for w in ['summarize','summary','about','what is']):
        return 'summarize'
    elif any(w in q for w in ['author','who wrote','researcher']):
        return 'authors'
    elif any(w in q for w in ['find','similar','related','search']):
        return 'search'
    elif any(w in q for w in ['domain','category','field','topic']):
        return 'classify'
    else:
        return 'deep'   # deep question → Groq

# ── UI Layout ────────────────────────────────────────────────
st.title("📄 SmartDoc AI")
st.caption("Intelligent Research Paper Analysis Chatbot")

# Sidebar
with st.sidebar:
    st.header("📁 Upload Paper")
    uploaded = st.file_uploader(
        "Upload a research paper (.txt or .pdf)",
        type=['txt', 'pdf']
    )

    if uploaded:
        if uploaded.type == "text/plain":
            paper_text = uploaded.read().decode('utf-8')
        else:
            import fitz  # PyMuPDF
            doc        = fitz.open(stream=uploaded.read(),
                                   filetype="pdf")
            paper_text = "\n".join([p.get_text() for p in doc])

        st.success(f"✅ Paper loaded — {len(paper_text.split())} words")
        st.session_state['paper_text'] = paper_text

        # Auto-classify
        category = classify_paper(paper_text[:1000])
        st.info(f"🏷️ Domain detected : **{category}**")

    st.divider()
    st.header("💡 Example questions")
    examples = [
        "Summarize this paper",
        "Who are the authors?",
        "What method did they propose?",
        "What are the main results?",
        "What are the limitations?",
        "Find similar papers about deep learning",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state['example'] = ex

# Main chat area
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role"   : "assistant",
        "content": "👋 Hello! I'm SmartDoc AI.\n\nUpload a research paper on the left and ask me anything about it — I can summarize it, extract key information, find similar papers, or answer deep questions!"
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Handle example button click
if 'example' in st.session_state:
    prompt = st.session_state.pop('example')
else:
    prompt = st.chat_input("Ask anything about the paper...")

if prompt:
    # Add user message
    st.session_state.messages.append({
        "role": "user", "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing..."):

            paper_text = st.session_state.get('paper_text', '')
            intent     = detect_intent(prompt)
            response   = ""

            # ── Route to correct module ──
            if intent == 'summarize':
                if paper_text:
                    summary  = extractive_summary(paper_text)
                    response = f"📝 **Summary :**\n\n{summary}"
                else:
                    response = "⚠️ Please upload a paper first!"

            elif intent == 'authors':
                if paper_text:
                    response = deep_answer(
                        "Who are the authors of this paper? List them.",
                        paper_text
                    )
                else:
                    response = "⚠️ Please upload a paper first!"

            elif intent == 'classify':
                if paper_text:
                    cat      = classify_paper(paper_text[:1000])
                    response = f"🏷️ **Scientific domain :** `{cat}`"
                else:
                    response = "⚠️ Please upload a paper first!"

            elif intent == 'search':
                results  = semantic_search(prompt, k=3)
                response = "🔍 **Most relevant papers :**\n\n"
                for r in results:
                    response += f"**{r['rank'] if 'rank' in r else ''}. {r['title']}**\n"
                    response += f"- Category : `{r['category']}`\n"
                    response += f"- Score : `{r['score']}`\n"
                    response += f"- {r['abstract']}\n\n"

            elif intent == 'deep':
                if paper_text:
                    response = deep_answer(prompt, paper_text)
                else:
                    response = "⚠️ Please upload a paper first to ask deep questions!"

            st.markdown(response)

    # Save response
    st.session_state.messages.append({
        "role": "assistant", "content": response
    })