import os
import re

class BookLoader:
    def __init__(self, books_dir="Books"):
        self.books_dir = books_dir

    def load_book(self, book_filename):
        """Loads the content of a book from a text file."""
        path = os.path.join(self.books_dir, book_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Book file not found: {path}")
            
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return text

    def chunk_book(self, text, book_title, chunk_size=800, overlap=100):
        """
        Splits text into overlapping chunks.
        
        Args:
            text (str): Full text of the book.
            book_title (str): Title for metadata.
            chunk_size (int): Approx number of words/tokens per chunk.
            overlap (int): Number of words/tokens to overlap.
            
        Returns:
            list of dict: Each dict contains 'text', 'book', 'start_idx', 'end_idx'.
        """
        # Simple whitespace tokenization is usually sufficient for retrieval chunks
        # closer to 500-1000 tokens.
        words = text.split()
        chunks = []
        
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size  # Prevent infinite loop if overlap >= chunk_size

        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Determine relative position
            if i < len(words) * 0.33:
                pos = "early"
            elif i < len(words) * 0.66:
                pos = "mid"
            else:
                pos = "late"

            chunks.append({
                "text": chunk_text,
                "book": book_title,
                "chunk_index": len(chunks),
                "relative_position": pos,
                "token_start": i,
                "token_end": i + len(chunk_words)
            })
            
        return chunks

    def process_all_books(self):
        """Loads and chunks all books in the directory."""
        all_chunks = []
        # Support common extensions
        for filename in os.listdir(self.books_dir):
            if filename.lower().endswith(".txt"):
                # Use filename without extension as title if needed, 
                # but better to normalize names later if they don't match CSV exactly
                book_title = os.path.splitext(filename)[0] 
                
                print(f"Propcessing book: {book_title}")
                text = self.load_book(filename)
                chunks = self.chunk_book(text, book_title)
                all_chunks.extend(chunks)
                
        return all_chunks
