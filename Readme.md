# llm-db

Greenpeace archive search using a PostgreSQL + pgvector knowledge base, Gemini embeddings, and a Streamlit RAG app.

This project has two main steps:
1. Ingest WordPress export data into vector chunks (`convert.py`).
2. Query those vectors with a multilingual Streamlit assistant (`app.py`).

## Architecture

- Data source: `wordpress_export.csv` (Contains columns for ID,Title,Content,Date,Permalink,"Post Type",Status)
- Vector DB: PostgreSQL with `pgvector`
- Embedding model: `gemini-embedding-2` (768 dimensions)
- Generation model: `gemini-3.1-flash-lite`
- App UI: Streamlit with Google sign-in and `@greenpeace.org` domain lock

## Requirements

- Linux server or local machine with Docker
- Python 3.10+
- A Gemini API key (`GEMINI_API_KEY` or `GOOGLE_API_KEY`)
- Google Auth configuration for Streamlit login

## 1. Start PostgreSQL + pgvector

```bash
docker run --name vector-db -e POSTGRES_PASSWORD="<YOUR_DB_PASSWORD>" -p 5432:5432 -d pgvector/pgvector:pg16
```

Create extension + table:

```bash
docker exec -it vector-db psql -U postgres -c "
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS post_chunks (
	id SERIAL PRIMARY KEY,
	post_title TEXT,
	post_url TEXT,
	content_chunk TEXT,
	embedding vector(768)
);"
```

Create ANN index for cosine distance (only run after importing the content):

```bash
docker exec -it vector-db psql -U postgres -c "CREATE INDEX IF NOT EXISTS post_chunks_embedding_hnsw_idx ON post_chunks USING hnsw (embedding vector_cosine_ops);"
```

## 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure secrets

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "<YOUR_GEMINI_API_KEY>"
DATABASE_URL = "postgresql://postgres:<YOUR_DB_PASSWORD>@localhost:5432/postgres"

[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "<RANDOM_SECRET>"

[auth.google]
client_id = "<GOOGLE_CLIENT_ID>"
client_secret = "<GOOGLE_CLIENT_SECRET>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Environment variables can be used instead of secrets for keys/DB URL:

```bash
export GEMINI_API_KEY="<YOUR_GEMINI_API_KEY>"
export DATABASE_URL="postgresql://postgres:<YOUR_DB_PASSWORD>@localhost:5432/postgres"
```

## 4. Build embeddings and ingest data

`convert.py`:
- Loads rows from `wordpress_export.csv`
- Chunks post content (~1000 chars with overlap)
- Embeds each chunk (`RETRIEVAL_DOCUMENT`, 768 dims)
- Inserts chunk vectors into `post_chunks`
- Skips URLs already present in DB

Run:

```bash
python3 convert.py
```

## 5. Run the app

Local:

```bash
streamlit run app.py
```

Background server mode (example):

```bash
nohup venv/bin/streamlit run app.py --server.port 80 --server.address 0.0.0.0 &
```

## How querying works (`app.py`)

1. User question is embedded with `gemini-embedding-2` (`RETRIEVAL_QUERY`).
2. DB query fetches nearest chunks with cosine distance:
   - `SIMILARITY_CUTOFF = 0.85`
   - `MAX_SOURCES = 100`
3. Retrieved context is passed to Gemini generation with strict language instruction.
4. Answer plus source links are returned in the UI.

The app supports output in:
- English
- German
- French
- Italian

Access control:
- User must authenticate with Google.
- Email must end with `@greenpeace.org`.

## Database backup and restore

Backup:

```bash
docker exec -i -e PGPASSWORD="<YOUR_DB_PASSWORD>" vector-db pg_dump -U postgres -d postgres > greenpeace_archive_backup.sql
```

Restore:

```bash
cat greenpeace_archive_backup.sql | docker exec -i -e PGPASSWORD="<YOUR_DB_PASSWORD>" vector-db psql -U postgres -d postgres
```

## Useful checks

Count chunks:

```bash
docker exec -it vector-db psql -U postgres -c "SELECT count(*) FROM post_chunks;"
```

## Notes

- Keep secrets out of shell history and git.
- The ingestion script currently contains a local DB connection string; for production, align it with `DATABASE_URL`.
- Retry logic for embedding API is already included in `convert.py` (exponential backoff on temporary service errors).
