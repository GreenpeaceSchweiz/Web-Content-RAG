import pandas as pd
import psycopg2
from google.genai import types
from psycopg2.extras import execute_values
import time
import random
from google.api_core import exceptions
from gemini_client import get_genai_client

# 1. Setup the new GenAI Client
client = get_genai_client()

# 2. Setup DB Connection
conn = psycopg2.connect("host=localhost dbname=postgres user=postgres password=mysecretpassword")
cur = conn.cursor()

def get_embedding(text, max_retries=5):
    for i in range(max_retries):
        try:
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768
                )
            )
            return result.embeddings[0].values
        except (exceptions.ServiceUnavailable, exceptions.InternalServerError) as e:
            if i == max_retries - 1:
                raise e
            # Exponential backoff: wait 2, 4, 8, 16... seconds + a bit of "jitter"
            wait_time = (2 ** i) + random.random()
            print(f"Service busy. Retrying in {wait_time:.2f}s... (Attempt {i+1}/{max_retries})")
            time.sleep(wait_time)
    return None

def chunk_text(text, size=1000):
    # Simple chunking logic: split by characters with some overlap
    return [text[i:i + size] for i in range(0, len(text), size - 100)]

# 3. Process CSV
df = pd.read_csv("wordpress_export.csv")

print(f"Starting vectorization of {len(df)} posts...")

for index, row in df.iterrows():
    title = str(row['Title'])
    url = str(row['Permalink'])
    content = str(row['Content'])
    
    chunks = chunk_text(content)
    batch_data = []

    # Check if the the URL already exists
    cur.execute("SELECT 1 FROM post_chunks WHERE post_url = %s LIMIT 1", (url,))
    
    if cur.fetchone():
        print(f"Skipping {title} - already in database.")
        continue
    
    for chunk in chunks:
        try:
            vector = get_embedding(chunk)
            batch_data.append((title, url, chunk, vector))
        except Exception as e:
            print(f"Error embedding chunk in '{title}': {e}")
    
    # 4. Batch Insert
    if batch_data:
        execute_values(cur, 
            "INSERT INTO post_chunks (post_title, post_url, content_chunk, embedding) VALUES %s", 
            batch_data)
        conn.commit()
    
    if index % 100 == 0:
        print(f"Processed {index}/{len(df)} posts...")

print("Success! Your WordPress knowledge is now a live Vector Database.")