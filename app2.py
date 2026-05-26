import streamlit as st
import psycopg2
from google.genai import types
from gemini_client import get_genai_client

# 1. Config
MAX_SOURCES = 30         
SIMILARITY_CUTOFF = 0.85 

client = get_genai_client()

@st.cache_resource
def get_db_connection():
    return psycopg2.connect("host=localhost dbname=postgres user=postgres password=mysecretpassword")

# MODIFIED: Added target_language parameter
def ask_hybrid_rag(user_query, target_language):
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
        return f"Keine relevanten Informationen gefunden. / No relevant information found.", []

    # B. Package context
    context_chunks = []
    source_list = []
    for title, url, chunk, distance in db_results:
        context_chunks.append(f"Source: {title} ({url})\nDistance Score: {distance:.3f}\nContent: {chunk}\n---")
        source_list.append(f"- [{title}]({url})")

    context_text = "\n".join(context_chunks)
    
    # C. System Instructions (MODIFIED for language enforcement)
    system_instruction = f"""
    Du bist ein präziser Greenpeace-Historiker und Archivar. 
    Dir wird ein Kontext aus unstrukturierten Textabschnitten zur Verfügung gestellt.
    Deine Aufgabe ist es, die Frage des Benutzers basierend auf diesem Kontext detailgetreu zu beantworten.
    
    WICHTIGE REGELN:
    - SPRACHE: Du musst die Antwort vollständig in der folgenden Sprache verfassen: {target_language}.
    - Falls der Quelltext oder die Frage in einer anderen Sprache sind, übersetze die Fakten akkurat in die Zielsprache ({target_language}).
    - Achte besonders auf spezifische Eigennamen, Städte (z.B. Liestal), Firmennamen oder Jahreszahlen.
    - Wenn der Kontext irrelevantes Rauschen enthält, filtere es eigenständig heraus.
    - Nenne alle relevanten Quellen-URLs am Ende deiner Antwort.
    """
    
    main_prompt = f"KONTEXT:\n{context_text}\n\nFRAGE: {user_query}"
    
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
st.set_page_config(page_title="Greenpeace Archiv Brain", page_icon="🔍", layout="wide")

# NEW: Sidebar configuration for Language
st.sidebar.title("⚙️ Einstellungen / Settings")
lang_options = {
    "Deutsch": "German",
    "English": "English",
    "Français": "French",
    "Italiano": "Italian"
}
selected_lang_ui = st.sidebar.selectbox("Antwortsprache / Output Language:", list(lang_options.keys()))
target_language = lang_options[selected_lang_ui]

st.title("🌱 Greenpeace Archiv Assistent")
st.write("Durchsuche 8.000 Blog-Artikel mithilfe von KI nach historischen Verbindungen und Entitäten.")

# User Input Entry
user_query = st.text_input("Deine Frage an das Archiv:", placeholder="z.B. Welche Verbindungen gibt es zur Stadt Liestal?")

if user_query:
    with st.spinner("Durchsuche Archiv-Vektoren und synthetisiere Antwort..."):
        try:
            # MODIFIED: Passing target_language here
            answer, source_urls = ask_hybrid_rag(user_query, target_language)
            
            st.markdown(f"### 🤖 Antwort des Archivars ({selected_lang_ui}):")
            st.markdown(answer)
            
            st.write("---")
            
            with st.expander(f"📚 Verwendete Quellen analysiert ({len(source_urls)})"):
                for source in source_urls:
                    st.markdown(source)
                    
        except Exception as e:
            st.error(f"Ein Fehler ist aufgetreten: {e}")