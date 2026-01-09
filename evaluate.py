import os
import pandas as pd
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report

from src.ingestion import BookLoader
from src.indexing import VectorIndex
from src.retrieval import Retriever
from src.reasoning import ConsistencyVerifier

def main():
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Evaluate Narrative Consistency Model")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to evaluate (default: all)")
    parser.add_argument("--reingest", action="store_true", help="Force re-ingestion of books")
    args = parser.parse_args()

    # Configuration
    BOOKS_DIR = os.path.join("data", "Books")
    INPUT_CSV = os.path.join("data", "train.csv") 
    
    # 1. Initialize Components
    print("--- Initializing System (Evaluation Mode) ---")
    index = VectorIndex()
    retriever = Retriever(vector_index=index)
    verifier = ConsistencyVerifier()
    
    # 2. Ingest and Index Books (Optional via flag)
    if args.reingest:
        print("\n--- Phase 1: Ingestion & Indexing ---")
        loader = BookLoader(books_dir=BOOKS_DIR)
        all_chunks = loader.process_all_books()
        print(f"Total chunks found: {len(all_chunks)}")
        index.build_indices(all_chunks)
    else:
        print("\n--- Phase 1: Skipped Ingestion (Default) ---")
        print("Use --reingest to force update of vector store.")

    # 3. Load input data
    print(f"\n--- Phase 2: Processing {INPUT_CSV} ---")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Filter rows if limit is set
    if args.limit:
        print(f"Limiting evaluation to first {args.limit} rows.")
        df = df.head(args.limit)

    # 4. Processing Loop
    print("Verifying claims...")
    predictions = []
    actuals = []
    
    label_map = {
        'consistent': 1,
        'contradict': 0
    }

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Parse ground truth
        label_str = str(row['label']).lower()
        if label_str not in label_map:
            print(f"Warning: Unknown label '{label_str}' at row {idx}. Skipping.")
            continue
            
        actual = label_map[label_str]
        actuals.append(actual)
        
        claim_text = row['content'] # train.csv uses 'content'
        book_title = row['book_name']
        character = row['char']
        
        # A. Retrieve Evidence
        evidence = retriever.retrieve_evidence(
            claim_text=claim_text,
            book_title=book_title,
            character=character,
            top_k=5
        )
        
        # B. Verify Consistency
        pred = verifier.verify(
            claim=claim_text,
            evidence_chunks=evidence,
            character=character
        )
        predictions.append(pred)

    # 5. Report Generation
    print("\n--- Phase 3: Evaluation Report ---")
    if not actuals:
        print("No valid labels found to evaluate.")
        return

    accuracy = accuracy_score(actuals, predictions)
    report = classification_report(actuals, predictions, target_names=['Contradict (0)', 'Consistent (1)'])

    print(f"Filtered Rows Processed: {len(actuals)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
