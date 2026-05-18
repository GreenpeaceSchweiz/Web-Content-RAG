import streamlit as st
import psycopg2
from google import genai
from google.genai import types

# 1. Config optimized for your research
MAX_SOURCES = 100         
SIMILARITY_CUTOFF = 0.85 

# Add your Gemini API key here
client = genai.Client(api_key="")

# Cache the database connection so it doesn't reconnect on every click
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
        return "Keine relevanten Informationen im Archiv gefunden.", []

    # B. Package context
    context_chunks = []
    source_list = []
    for title, url, chunk, distance in db_results:
        context_chunks.append(f"Source: {title} ({url})\nDistance Score: {distance:.3f}\nContent: {chunk}\n---")
        # Format as Markdown clickable links for the UI
        source_list.append(f"- [{title}]({url})")

    context_text = "\n".join(context_chunks)
    
    # C. System Instructions (MODIFIED: Injected language rule seamlessly)
    system_instruction = f"""
    Du bist ein präziser Greenpeace-Historiker und Archivar. 
    Dir wird ein Kontext aus unstrukturierten Textabschnitten zur Verfügung gestellt.
    Deine Aufgabe ist es, die Frage des Benutzers basierend auf diesem Kontext detailgetreu zu beantworten.
    
    WICHTIGE REGEL FÜR DIE SPRACHE:
    - Du MUSST die Antwort vollständig in folgender Sprache verfassen: {target_language}.
    - Falls die Quelltexte oder die Frage in einer anderen Sprache sind, übersetze die Fakten akkurat direkt in diese Zielsprache ({target_language}).
    
    WICHTIG FÜR DIE HISTORISCHE RECHERCHE:
    - Achte besonders auf spezifische Eigennamen, Städte (z.B. Liestal), Firmennamen oder Jahreszahlen.
    - Manchmal ist die gesuchte Verbindung klein (z.B. nur eine beiläufige Erwähnung im Satz). Ignoriere diese Details nicht!
    - Wenn der Kontext irrelevantes Rauschen enthält, filtere es eigenständig heraus.
    - Falls die Anfrage keine konkrete Anweisung zum Zeitraum enthält, gewichte aktuellere Inhalte höher als ältere.
    - Wenn die gesuchten Fakten im Text vorkommen, beantworte die Frage präzise.
    - Zitiere immer die URLs der Quellen, die du für die Aussagen benutzt. Sowohl am Ende der Antwort für allgemeine Quellen als auch direkt in der Antwort, um jede einzelne Aussage zu belegen.
    - Falls die Antwort im Kontext nicht zu finden ist, antworte, dass die Frage aufgrund des Archivs nicht beantwortet werden kann.
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
st.set_page_config(page_title="Greenpeace.ch Archive Assistant", page_icon="🕵️", layout="centered")

# ADDED: Sidebar Layout for Settings & Language Selection
st.sidebar.title("⚙️ Setup")
lang_options = {
    "Deutsch": "German",
    "English": "English",
    "Français": "French",
    "Italiano": "Italian"
}
selected_lang_ui = st.sidebar.selectbox("Output Language / Antwortsprache:", list(lang_options.keys()))
target_language = lang_options[selected_lang_ui]

st.title("🌱 Greenpeace Archive Assistant")
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
                for source in source_urls:
                    st.markdown(source)
                    
        except Exception as e:
            st.error(f"Oops, an error occurred: {e}")