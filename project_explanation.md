# Narrative Consistency Verification - Project Explanation

This document explains how our system verifies if a specific claim (like a character's backstory or an event) is consistent with a complete book.

## 1. The Problem
We have a list of claims (e.g., "Thalcave was a guide who lived in the mountains"). We have the full text of the book (e.g., *In Search of the Castaways*). Our job is to read the book and decide if the claim is **Consistent** (True) or **Inconsistent** (False).

## 2. The Solution: Step-by-Step

We cannot just "ask AI" to read the whole book at once because books are too long. Instead, we use a 3-step pipeline: **Ingest $\rightarrow$ Retrieve $\rightarrow$ Reason**.

### Step 1: Book Ingestion (Reading & Splitting)
First, we prepare the books so a computer can search them easily.
1.  **Load**: Read the `.txt` file of the book.
2.  **Chunk**: Cut the book into smaller pieces (chunks) of about 500-1000 words. We make them overlap a bit so we don't cut a sentence in half.
3.  **Embed**: Turn each chunk into a list of numbers (a "vector") that represents its *meaning*.

### Step 2: Evidence Retrieval (Finding Clues)
When we get a claim to check:
1.  We look at which book it is talking about.
2.  We turn the claim into a vector.
3.  We query the **Supabase Database** to find the top 5 chunks closest to our claim vector (using `pgvector`).

### Step 3: Reasoning (The Judge)
Now we act like a judge. Use key Google Gemini API.
1.  We give the AI the **Claim**.
2.  We give the AI the **Evidence**.
3.  The AI gives us a final answer.

## 3. System Flowchart

```mermaid
graph TD
    subgraph "Phase 1: Ingestion"
        Book[Book .txt File] --> Split[Chunker]
        Split --> Model[Sentence Transformer]
        Model -->|Vectors| DB[(Supabase Postgres)]
    end

    subgraph "Phase 2: Verification"
        Claim[Input Claim] --> Embed[Embed Claim]
        Embed --> Query[SQL Query]
        DB --> Query
        Query --> Evidence[Relevant Chunks]
        
        Claim --> Prompt
        Evidence --> Prompt
        Prompt --> LLM[Gemini API]
        LLM --> Decision{Consistent?}
    end
```

## 4. Key Components

- **`src/ingestion.py`**: Loads books and creates chunks.
- **`src/indexing.py`**: Uploads vectors to **Supabase**.
- **`src/retrieval.py`**: Performs SQL similarity search.
- **`src/reasoning.py`**: Calls Gemini.
- **`main.py`**: Orchestrator.

## 5. How to Run
1.  Install: `pip install -r requirements.txt`
2.  Set Keys:
    - `set GEMINI_API_KEY=...`
    - `set SUPABASE_DB_URL=postgresql://user:pass@host:5432/postgres`
3.  Run: `python main.py`
