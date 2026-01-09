class Retriever:
    def __init__(self, vector_index):
        """
        Args:
            vector_index (VectorIndex): Instance containing the DB connection and model.
        """
        self.vector_index = vector_index
        self.conn = vector_index.get_connection()
        self.model = vector_index.model

    def retrieve_evidence(self, claim_text, book_title, character=None, top_k=5):
        """
        Retrieves top_k chunks relevant to the claim using SQL vector search.
        """
         # Construct query context
        if character:
            query_text = f"{character}: {claim_text}"
        else:
            query_text = claim_text
            
        # Generate embedding for query
        query_embedding = self.model.encode([query_text])[0].tolist()
        
        result_chunks = []
        
        # Perform Fuzzy / Partial matching for book title if exact match fails
        # But simpler for SQL: Just use ILIKE
        
        with self.conn.cursor() as cur:
            # Query: 
            # 1. Filter by book_title (case insensitive)
            # 2. Order by cosine distance (<=>)
            
            cur.execute("""
                SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity
                FROM book_chunks
                WHERE book_title ILIKE %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s;
            """, (query_embedding, f"%{book_title}%", query_embedding, top_k))
            
            rows = cur.fetchall()
            
            for row in rows:
                result_chunks.append({
                    "text": row[0],
                    "metadata": row[1],
                    "score": row[2]
                })
                
        return result_chunks
