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
    
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Narrative Consistency Verification System")
    parser.add_argument("--mode", choices=["inference", "evaluate"], default="inference", 
                        help="Mode: 'inference' (predict test.csv) or 'evaluate' (validate against train.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to process")
    parser.add_argument("--reingest", action="store_true", help="Force re-ingestion of books via Pathway")
    parser.add_argument("--input_csv", type=str, default=None, help="Override input CSV path")
    args = parser.parse_args()

    # Configuration
    BOOKS_DIR = os.path.join("data", "Books")
    OUTPUT_FILE = os.path.join("output", "submission.csv")
    
    # 1. Initialize Components
    print(f"--- Initializing System (Mode: {args.mode.upper()}) ---")
    
    # Initialize VectorIndex (connects to Supabase)
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        print("Error: SUPABASE_DB_URL not set in environment.")
        return
        
    index = VectorIndex(db_url=db_url)
    
    # 2. Ingestion
    ingestion_success = False
    
    if args.reingest:
        print("\n--- Phase 1: Ingestion & Indexing ---")
        try:
            # Attempt to use Pathway
            print("Attempting to use Pathway pipeline...")
            from src.pathway_pipeline import run_pathway_ingestion
            run_pathway_ingestion(BOOKS_DIR, db_url)
            ingestion_success = True
        except (ImportError, AttributeError, Exception) as e:
            print(f"\n[WARNING] Pathway ingestion failed or is not supported on this platform: {e}")
            print("Falling back to standard Python ingestion...")
            
            # Fallback to standard ingestion
            try:
                loader = BookLoader(books_dir=BOOKS_DIR)
                all_chunks = loader.process_all_books()
                print(f"Total chunks found: {len(all_chunks)}")
                index.build_indices(all_chunks)
                ingestion_success = True
            except Exception as e2:
                print(f"Standard ingestion also failed: {e2}")

        if ingestion_success:
           print("Ingestion completed successfully.")

    else:
        print("\n--- Phase 1: Skipped Ingestion (Default) ---")
        print("Use --reingest to force update of vector store.")

    retriever = Retriever(vector_index=index)
    verifier = ConsistencyVerifier()

    # 3. Mode Selection
    if args.mode == "inference":
        input_csv = args.input_csv or os.path.join("data", "test.csv")
        run_inference(input_csv, OUTPUT_FILE, retriever, verifier, args.limit)
    elif args.mode == "evaluate":
        input_csv = args.input_csv or os.path.join("data", "train.csv")
        run_evaluation(input_csv, retriever, verifier, args.limit)

def run_inference(input_path, output_path, retriever, verifier, limit=None):
    print(f"\n--- Phase 2: Inference on {input_path} ---")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    if limit:
        df = df.head(limit)
        
    results = []
    
    print("Verifying claims...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        claim_id = row['id']
        book_title = row['book_name']
        character = row['char']
        claim_text = row.get('content', row.get('text', '')) # flexible ref
        
        evidence = retriever.retrieve_evidence(claim_text, book_title, character, top_k=5)
        prediction = verifier.verify(claim_text, evidence, character)
        
        results.append({
            "id": claim_id,
            "prediction": prediction
        })

    print("\n--- Phase 3: Output Generation ---")
    output_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")


def run_evaluation(input_path, retriever, verifier, limit=None):
    print(f"\n--- Phase 2: Evaluation on {input_path} ---")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    if limit:
        df = df.head(limit)
        
    predictions = []
    actuals = []
    label_map = {'consistent': 1, 'contradict': 0}

    print("Verifying claims...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        label_str = str(row['label']).lower()
        if label_str not in label_map:
             continue
        
        actuals.append(label_map[label_str])
        
        claim_text = row.get('content', row.get('text', ''))
        book_title = row['book_name']
        character = row['char']
        
        evidence = retriever.retrieve_evidence(claim_text, book_title, character, top_k=5)
        prediction = verifier.verify(claim_text, evidence, character)
        predictions.append(prediction)

    print("\n--- Phase 3: Evaluation Report ---")
    if not actuals:
        print("No valid labels found to evaluate.")
        return

    accuracy = accuracy_score(actuals, predictions)
    report = classification_report(actuals, predictions, target_names=['Contradict (0)', 'Consistent (1)'])

    print(f"Rows Processed: {len(actuals)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
