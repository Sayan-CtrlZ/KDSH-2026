import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from src.ingestion import BookLoader
from src.indexing import VectorIndex
from src.retrieval import Retriever
from src.reasoning import ConsistencyVerifier

def main():
    # Load environment variables
    load_dotenv()

    # Configuration
    BOOKS_DIR = os.path.join("data", "Books")
    INPUT_CSV = os.path.join("data", "test.csv") # Change to train.csv for testing/calibration
    OUTPUT_FILE = os.path.join("output", "submission.csv")
    
    # 1. Initialize Components
    print("--- Initializing System ---")
    loader = BookLoader(books_dir=BOOKS_DIR)
    index = VectorIndex()
    
    # 2. Ingest and Index Books
    print("\n--- Phase 1: Ingestion & Indexing ---")
    all_chunks = loader.process_all_books()
    print(f"Total chunks created: {len(all_chunks)}")
    
    index.build_indices(all_chunks)
    
    retriever = Retriever(vector_index=index)
    verifier = ConsistencyVerifier()

    # 3. Load input data
    print(f"\n--- Phase 2: Processing {INPUT_CSV} ---")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Check for required columns
    required_columns = ['id', 'book_name', 'char', 'content'] # Using 'content' as the claim text based on CSV view
    for col in required_columns:
        if col not in df.columns:
            # Fallback for different column naming if needed (e.g. 'text' instead of 'content')
            if col == 'content' and 'text' in df.columns:
                 df.rename(columns={'text': 'content'}, inplace=True)
            else:
                 print(f"Error: Missing column '{col}' in CSV.")
                 return

    results = []
    
    # 4. Processing Loop
    print("Verifying claims...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        claim_id = row['id']
        book_title = row['book_name']
        character = row['char']
        claim_text = row['content']
        
        # A. Retrieve Evidence
        evidence = retriever.retrieve_evidence(
            claim_text=claim_text,
            book_title=book_title,
            character=character,
            top_k=5
        )
        
        # B. Verify Consistency
        prediction = verifier.verify(
            claim=claim_text,
            evidence_chunks=evidence,
            character=character
        )
        
        results.append({
            "id": claim_id,
            "prediction": prediction
        })

    # 5. Output Generation
    print("\n--- Phase 3: Output Generation ---")
    output_df = pd.DataFrame(results)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission saved to: {OUTPUT_FILE}")
    print("Done!")

if __name__ == "__main__":
    main()
