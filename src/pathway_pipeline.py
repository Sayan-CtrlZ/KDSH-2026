import pathway as pw
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Define the schema for the input data source (files)
class InputSchema(pw.Schema):
    text: str
    path: str

# Define embedding model as a user-defined function (UDF)
# We use a class to load the model once
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def __call__(self, text):
        return self.model.encode(text).tolist()

# Initialize the embedder UDF
embedder = Embedder()

@pw.udf
def compute_embedding(text: str) -> list[float]:
    return embedder(text)

@pw.udf
def extract_metadata(path: str, chunk_index: int, token_start: int) -> str:
    # Basic metadata extraction
    filename = path.split(os.path.sep)[-1]
    book_title = os.path.splitext(filename)[0]
    
    meta = {
        "book": book_title,
        "chunk_index": chunk_index,
        "token_start": token_start
        # relative_position calc is harder in stream, skipping for simplicity or can add later
    }
    return json.dumps(meta)

@pw.udf
def get_book_title(path: str) -> str:
    filename = path.split(os.path.sep)[-1]
    return os.path.splitext(filename)[0]

def run_pathway_ingestion(data_dir, db_url):
    print(f"Starting Pathway ingestion from {data_dir}...")
    
    # 1. Read files
    # pattern="*.txt" ensures we only read books
    files = pw.io.fs.read(data_dir, format="text", glob_pattern="*.txt", schema=InputSchema)
    
    # 2. Chunking (Custom FlatMap not always built-in for overlapping text, 
    # so we'll use a UDF that returns a list of chunks, then flatten)
    # Actually, Pathway has specific connectors, but for custom chunking logic 
    # (overlapping windows), a Python UDF flattening is best.
    
    # Simplified: We will just write a UDF that takes the whole text and returns 
    # a list of tuples (chunk_text, chunk_idx, start). 
    # Pathway's `flatten` is needed here. 
    
    # Let's use a simpler approach for the demo: 
    # We rely on the existing logic logic but lifted into Pathway.
    
    # However, to keep it extremely simple and readable as requested:
    # We will assume each line or small block is efficient. 
    # BUT, splitting a whole book (1MB+) in one UDF might be heavy.
    # Standard practice: pw.io.fs.read(..., mode="streaming") handles standard line inputs.
    # Since these are whole books, valid "text" format reads whole file content into one row usually? 
    # Yes, with `format="text"`, it reads the file content.
    
    # Define chunking UDF
    @pw.dof
    def chunk_text(text: str, path: str):
        words = text.split()
        chunk_size = 800
        overlap = 100
        step = chunk_size - overlap
        if step <= 0: step = chunk_size
        
        chunks = []
        for i in range(0, len(words), step):
             chunk_words = words[i : i + chunk_size]
             chunk_text = " ".join(chunk_words)
             yield chunk_text, i, path
    
    # Apply chunking (Flatten/Unnest)
    # flattened = files.select(chunks=chunk_text(pw.this.text, pw.this.path)).flatten(pw.this.chunks)
    # Proper syntax for flatmap in Pathway:
    chunks_table = files.select(
        entry=chunk_text(pw.this.text, pw.this.path)
    ).flatten(pw.this.entry).select(
        content=pw.this.entry[0],
        token_start=pw.this.entry[1],
        path=pw.this.entry[2]
    )

    # 3. Compute Embeddings & Metadata
    processed = chunks_table.select(
        book_title=get_book_title(pw.this.path),
        content=pw.this.content,
        embedding=compute_embedding(pw.this.content),
        metadata=extract_metadata(pw.this.path, 0, pw.this.token_start) # chunk_idx is loose here
    )

    # 4. Write to Supabase (Postgres)
    # Need to parse DB URL to get params
    # url format: postgresql://user:pass@host:port/dbname
    
    # Pathway postgres connector requires separate args usually, or use connection string if supported.
    # Checking docs (simulated): pw.io.postgres.write usually takes host, port, etc.
    # parsing url:
    from urllib.parse import urlparse
    result = urlparse(db_url)
    
    settings = {
        "host": result.hostname,
        "port": result.port,
        "dbname": result.path[1:],
        "user": result.username,
        "password": result.password,
        "table_name": "book_chunks",
        "primary_key": "id" # Pathway needs an ID, usually it generates its own or we map one.
    }
    
    # Note: Pathway outputs change streams. 
    # For a static load, we simply run.
    # However, Pathway's Postgres connector might expect specific setup.
    # To keep it robust using `psycopg2` inside a custom output function might be safer 
    # if `pw.io.postgres` version varies.
    # But let's try standard output.
    
    # Since Pathway is strict on types and verifying connection, and we are in a customized setup:
    # We will use `pw.io.python.write` to call our existing DB insertion logic!
    # This is often easier for custom schemas (vector type).
    
    # We'll re-use the logic from `indexing.py` but wrapped in a Pathway writer.
    
    processed_list = processed.select(
        data=pw.apply.tuple(pw.this.book_title, pw.this.content, pw.this.embedding, pw.this.metadata)
    )
    
    # Output to our custom writer that uses existing VectorIndex class or similar
    pw.io.python.write(
        processed_list,
        lambda row: insert_into_db(row, db_url)
    )
    
    # Run the pipeline
    pw.run()

def insert_into_db(row, db_url):
    # row is (book_title, content, embedding, metadata)
    # row might be a batch or single item depending on writer. 
    # pw.io.python.write calls this for each update.
    
    import psycopg2
    from psycopg2.extras import execute_values
    
    # NOTE: In a real streaming pipeline, we'd keep a connection open. 
    # For this batch job, opening/closing is okay-ish but inefficient. 
    # Improved: global connection or pooled.
    
    # Unpack
    # Pathway passes a tuple if we selected a tuple.
    # Check if row is a Change object or raw data. python.write sends raw data usually.
    
    # row is the object defined in select.
    data = row['data'] # (book_title, content, embedding, metadata)
    
    book_title, content, embedding, metadata = data
    
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO book_chunks (book_title, content, embedding, metadata)
                VALUES (%s, %s, %s, %s)
            """, (book_title, content, embedding, metadata))
        conn.commit()
    finally:
        conn.close()

if __name__ == "__main__":
    # Test run
    from dotenv import load_dotenv
    load_dotenv()
    DB_URL = os.environ.get("SUPABASE_DB_URL")
    run_pathway_ingestion("data/Books", DB_URL)
