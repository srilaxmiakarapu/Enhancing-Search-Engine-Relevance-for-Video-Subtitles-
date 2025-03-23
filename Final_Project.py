import streamlit as st
import speech_recognition as sr
import sqlite3
import zlib
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Set up ChromaDB storage
vector_db = Chroma("subtitle_search_db", embedding_function=embeddings)

def load_subtitles(database_path):
    """Extract subtitles from the SQLite database."""
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.execute("SELECT num, content FROM zipfiles")
    extracted_data = []
    
    for num, content in cursor.fetchall():
        try:
            text = zlib.decompress(content).decode('latin-1')
            extracted_data.append((num, text))
        except:
            continue
    
    connection.close()
    return extracted_data

# Define database path and extract subtitles
db_file = "Your/Subtitiles/Data_set/Here"  # Update this path
subtitle_data = load_subtitles(db_file)

# Process subtitles into chunks and store embeddings
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for id_num, subtitle_text in subtitle_data:
    text_chunks = splitter.split_text(subtitle_text)
    chunk_embeddings = embeddings.embed_documents(text_chunks)
    
    for text_chunk, emb in zip(text_chunks, chunk_embeddings):
        vector_db.add_texts([text_chunk], metadatas=[{"num": id_num}])

# Streamlit User Interface
st.title("Video Subtitle Search Engine")

# Audio file upload feature
uploaded_file = st.file_uploader("Upload a 2-minute audio clip", type=["wav", "mp3"])

if uploaded_file:
    speech_recognizer = sr.Recognizer()
    transcribed_query = None
    
    with sr.AudioFile(uploaded_file) as source:
        audio_content = speech_recognizer.record(source)
        try:
            transcribed_query = speech_recognizer.recognize_google(audio_content)
        except sr.UnknownValueError:
            st.write("Could not understand the audio.")
        except sr.RequestError:
            st.write("Error with speech recognition service.")
    
    if transcribed_query:
        # Generate query embedding
        query_embedding = embeddings.embed_query(transcribed_query)
        
        # Retrieve relevant subtitles
        search_results = vector_db.similarity_search_by_vector(query_embedding, k=5)
        
        # Display results
        st.write("## Top Matching Subtitle Segments")
        for index, result in enumerate(search_results):
            st.write(f"**Result {index + 1}:** {result.page_content}")
        
        # Show transcribed query
        st.write(f"## Transcribed Query: {transcribed_query}")
    else:
        st.write("No valid transcription available for searching.")
