import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
import json
import os
from psycopg2.extras import execute_values

class VectorIndex:
    def __init__(self, db_url=None, model_name=None):
        if model_name is None:
            model_name = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.db_url = db_url or os.environ.get("SUPABASE_DB_URL")
        
        if not self.db_url:
            raise ValueError("SUPABASE_DB_URL environment variable or argument is required.")

        print("Connecting to Supabase...")
        self.conn = psycopg2.connect(self.db_url)
        self.conn.autocommit = True
        
        self._init_db()

    def _init_db(self):
        """Initializes the database with pgvector extension and table."""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table
            # embedding vector(384) matches all-MiniLM-L6-v2 output dimension
            cur.execute("""
                CREATE TABLE IF NOT EXISTS book_chunks (
                    id SERIAL PRIMARY KEY,
                    book_title TEXT,
                    content TEXT,
                    embedding vector(384),
                    metadata JSONB
                );
            """)
            
            # Create HNSW index for faster search (optional but good for performance)
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS book_chunks_embedding_idx 
                    ON book_chunks 
                    USING hnsw (embedding vector_cosine_ops);
                """)
            except Exception as e:
                print(f"Index creation warning (might already exist or not supported on this tier): {e}")

    def build_indices(self, all_chunks, batch_size=100):
        """
        Embeds chunks and inserts them into Supabase.
        """
        print(f"Starting database insertion for {len(all_chunks)} chunks...")
        
        # We can check if data already exists to avoid duplicates or clear it.
        # For this pipeline, let's clear data to ensure fresh state (optional, be careful in prod)
        with self.conn.cursor() as cur:
             cur.execute("TRUNCATE TABLE book_chunks;")
        
        # Process in batches
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            
            # Prepare vectors
            texts = [c['text'] for c in batch]
            embeddings = self.model.encode(texts)
            
            # Prepare insert data
            values = []
            for j, chunk in enumerate(batch):
                embedding_list = embeddings[j].tolist()
                metadata = {
                    "book": chunk['book'],
                    "chunk_index": chunk['chunk_index'],
                    "relative_position": chunk['relative_position'],
                    "token_start": chunk['token_start'],
                    "token_end": chunk['token_end']
                }
                values.append((
                    chunk['book'],
                    chunk['text'],
                    embedding_list,
                    json.dumps(metadata)
                ))
            
            with self.conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO book_chunks (book_title, content, embedding, metadata)
                    VALUES %s
                """, values)
                
            print(f"Inserted batch {i} - {i + len(batch)}")
            
    def get_connection(self):
        return self.conn
