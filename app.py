"""
Greenpeace.ch Archive Assistant
This Streamlit app allows Greenpeace researchers to ask questions about the content of the greenpeace.ch website
"""

import streamlit as st
import os
import psycopg2
from urllib.parse import urlparse
from google.genai import types
from gemini_client import get_genai_client

# 1. Config optimized for your research
MAX_SOURCES = 100         
SIMILARITY_CUTOFF = 0.85 

def _escape_markdown(text):
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("#", "\\#")
        .replace("+", "\\+")
        .replace("-", "\\-")
        .replace(".", "\\.")
        .replace("!", "\\!")
    )


def _safe_source_url(url):
    parsed = urlparse(str(url))
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    return str(url)

def _load_database_url():
    return (
        os.getenv("DATABASE_URL")
        or st.secrets.get("DATABASE_URL")
        or st.secrets.get("database", {}).get("DATABASE_URL")
    )

# Cache the database connection so it doesn't reconnect on every click
@st.cache_resource
def get_db_connection():
    database_url = _load_database_url()
    if not database_url:
        raise RuntimeError(
            "Missing DATABASE_URL. Set it as an environment variable or in .streamlit/secrets.toml."
        )
    return psycopg2.connect(database_url)

# Main function to handle the RAG process
def ask_hybrid_rag(user_query, target_language):
    client = get_genai_client()
    conn = get_db_connection()
    cur = conn.cursor()
    
    # A. Direct Vector Search
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=user_query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
    )
    query_vector = result.embeddings[0].values

    search_sql = """
        SELECT post_title, post_url, content_chunk, (embedding <=> %s::vector) as distance
        FROM post_chunks
        WHERE (embedding <=> %s::vector) < %s
        ORDER BY distance ASC
        LIMIT %s;
    """
    
    cur.execute(search_sql, (query_vector, query_vector, SIMILARITY_CUTOFF, MAX_SOURCES))
    db_results = cur.fetchall()
    cur.close()

    if not db_results:
        return "Keine relevanten Informationen im Archiv gefunden.", []

    # B. Package context
    context_chunks = []
    source_list = []
    for title, url, chunk, distance in db_results:
        context_chunks.append(f"Source: {title} ({url})\nDistance Score: {distance:.3f}\nContent: {chunk}\n---")
        safe_url = _safe_source_url(url)
        safe_title = _escape_markdown(title)
        if safe_url:
            source_list.append((safe_title, safe_url))
        else:
            source_list.append((safe_title, None))

    context_text = "\n".join(context_chunks)
    
    # C. System Instructions
    system_instruction = f"""
    You are a meticulous Greenpeace historian and archivist. 
    You are provided with a context consisting of unstructured text passages.
    Your task is to answer the user's question in detail based on this context.
    
    IMPORTANT LANGUAGE RULES:
    - You MUST write the answer entirely in the following language: {target_language}.
    - If the source texts or the question are in another language, translate the facts accurately directly into this target language ({target_language}).
    
    IMPORTANT FOR YOUR RESEARCH:
    - Pay special attention to specific proper nouns, cities (e.g., Liestal), company names, or dates.
    - Sometimes the connection you're looking for is subtle (e.g., just a casual mention in a sentence). Don't ignore these details!
    - If the context contains irrelevant noise, filter it out on your own.
    - If the query does not specify a time period, prioritize more recent content over older content.
    - If the facts you are looking for appear in the text, answer the question precisely.
    - Always cite the URLs of the sources you use for your statements. Include them both at the end of the answer for general sources and directly within the answer to substantiate each individual statement.
    - If the answer cannot be found in the context, respond that the question cannot be answered due to the archive.
    """
    
    main_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {user_query}"
    
    # D. Generation
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite", 
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1 
        ),
        contents=main_prompt
    )
    
    return response.text, list(set(source_list))

# --- STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="Greenpeace.ch Archive Assistant", page_icon="🕵️", layout="centered")

# --- STEP 1: GOOGLE AUTHENTICATION & DOMAIN LOCK ---
if not st.user.is_logged_in:
    st.title("🔒 Greenpeace Archive Assistant")
    st.write("Please sign in with your Greenpeace account to proceed.")
    if st.button("Log in with Google"):
        st.login("google")
    st.stop()

# Strict domain lock check
user_email = st.user.email
if not user_email.endswith("@greenpeace.org"):
    st.error(f"Access denied. The account '{user_email}' does not have the needed permissions.")
    if st.button("Log out"):
        st.logout()
    st.stop()

# UI element in the sidebar to show logged in user and allow logout
st.sidebar.write(f"Logged in as: **{st.user.name}**")
if st.sidebar.button("Log out"):
    st.logout()


# Sidebar Layout for Settings & Language Selection
st.sidebar.title("⚙️ Setup")
lang_options = {
    "English": "English",
    "Deutsch": "German",
    "Français": "French",
    "Italiano": "Italian"
}
selected_lang_ui = st.sidebar.selectbox("Output Language:", list(lang_options.keys()))
target_language = lang_options[selected_lang_ui]

st.title("🌱 Greenpeace.ch Archive Assistant")
st.write("Use AI to search through more than 8,000 articles and press releases from greenpeace.ch")

# User Input Entry
user_query = st.text_input("Your question to the AI archive assistant:", placeholder="e.g. Tell me about recent trips of the Rainbow Warrior.")

if user_query:
    with st.spinner("Searching through archive vectors and generating a response..."):
        try:
            # MODIFIED: Passing the selected target language to the function
            answer, source_urls = ask_hybrid_rag(user_query, target_language)
            
            # Render the response nicely in Markdown
            st.markdown(f"### 🤖 The archivist bot's answer ({selected_lang_ui}):")
            st.markdown(answer)
            
            st.write("---")
            
            # Collapsible menu for the research links
            with st.expander(f"📚 Show used sources ({len(source_urls)})"):
                for title, url in source_urls:
                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")
                    
        except Exception as e:
            st.error("Oops, an error occurred while processing your request.")