# Narrative Consistency Verifier

This project uses AI to verify if specific claims (backstories, events) are consistent with the full text of a book. It uses **Supabase (pgvector)** for storage and **Gemini API** for reasoning.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+ installed.
- A **Supabase** project (PostgreSQL database).
- A **Google Gemini** API Key.

### 2. Setup

**Step A: Install Dependencies**
Open your terminal in the project folder:
```bash
pip install -r requirements.txt
```

**Step B: Configure Environment**
1. Copy the example file:
   ```bash
   cp .env.example .env
   # On Windows (cmd): copy .env.example .env
   ```
2. Open `.env` and fill in your keys:
   ```ini
   GEMINI_API_KEY=your_key_here
   SUPABASE_DB_URL=postgresql://postgres.user:pass@host:5432/postgres
   ```
   *(Get your connection string from Supabase Settings -> Database -> Connection Strings -> URI)*

### 3. Run the Pipeline

**Standard Inference (on `test.csv`)**:
```bash
python main.py
```

**Evaluation (on `train.csv`)**:
```bash
python main.py --mode evaluate --limit 10
```

**Re-Ingest Books (Pathway)**:
By default, the script skips ingestion if the DB is ready. To force a refresh:
```bash
python main.py --reingest
```

---
## 🧠 The "Why" - Understanding the Logic

### Why not just "paste the book into ChatGPT"?
**The Problem**: Books are huge (100k+ words). Most LLMs have a token limit or become expensive and "forgetful" with too much text.
**The Solution**: Retrieval-Augmented Generation (RAG). We only fetch the *specific paragraphs* relevant to the claim.

### Why Chunking?
**Rationale**: We split books into overlapping segments (e.g., 800 words).
- If we search for "thalcave's horse", we want the specific paragraph describing the horse, not the whole chapter.
- **Overlap** ensures we don't cut a critical sentence in half.

### Why Vector Search (Embeddings)?
**Rationale**: Keyword search fails on synonyms.
- Limit: Searching for "felony" might miss "crime".
- **Vectors**: Capture meaning. "Felony" and "crime" are mathematically close. We use `sentence-transformers` to turn text into numbers.

### Why Pathway?
**Rationale**: We use **Pathway** for the data ingestion pipeline.
- It handles reading files, processing streams, and scalable vector computation efficiently.
- It manages the flow from `data/Books` -> Chunking -> Embedding -> Supabase.

### Why Supabase (pgvector)?
**Rationale**: We need a place to store thousands of these vectors.
- **Supabase** is a PostgreSQL database.
- `pgvector` allows SQL to perform "nearest neighbor" searches efficiently.
- It's scalable and persistent (unlike a simple Python list).

### Why Gemini API?
**Rationale**: After finding the evidence, we need a "Judge".
- We send the specific **Claim** + **Found Evidence** to Gemini.
- It acts as a logic engine to decide True/False based *only* on the evidence provided.

---

## 📂 Project Structure

```
├── data/
│   ├── Books/           # Raw .txt files
│   ├── test.csv         # Claims to verify
│   └── train.csv
├── src/
│   ├── ingestion.py     # Cuts books into chunks
│   ├── indexing.py      # Uploads chunks to Supabase
│   ├── retrieval.py     # Finds relevant chunks
│   └── reasoning.py     # Asks Gemini for the verdict
├── output/              # Final results
├── main.py              # The script that runs everything
└── requirements.txt     # Python libraries
```
