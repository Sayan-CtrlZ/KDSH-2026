import google.generativeai as genai
import os
import time

class ConsistencyVerifier:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("WARNING: No Gemini API Key provided. Reasoning will fail or return mock results.")
        else:
            genai.configure(api_key=self.api_key)
            # Use a stable model
            self.model = genai.GenerativeModel('gemini-1.5-flash')

    def verify(self, claim, evidence_chunks, character=None):
        """
        Verifies if the claim is consistent with the evidence.
        
        Returns:
            int: 1 (Consistent) or 0 (Inconsistent)
        """
        if not self.api_key:
            # Fallback for testing without key
            return 0 
            
        # 1. format context
        context_text = ""
        for i, chunk in enumerate(evidence_chunks):
            context_text += f"---\nChunk {i+1}:\n{chunk['text']}\n"
            
        # 2. Construct Prompt
        # Strong instructions for the task
        prompt = f"""
        You are a strict consistency checker for a library database.
        
        Your Task: Determine if the CLAIM is consistent with the provided EVIDENCE from the book.
        
        Claim regarding: {character if character else 'Independent'}
        CLAIM: "{claim}"
        
        EVIDENCE from the book:
        {context_text}
        
        Rules:
        1. If the claim directly contradicts the evidence (e.g., dead vs alive, different location, different parent), output 0.
        2. If the evidence strongly supports the claim (matches details), output 1.
        3. If the evidence implies the claim is consistent (e.g., behavior fits character), output 1.
        4. If the evidence is irrelevant or neutral, but not contradictory, be conservative. However, usually, if we retrieved semantic matches and they don't support it, it might be false. 
           NOTE: If you found NO mentions of the specific event in the evidence, but the claim is specific (like "He killed a bear"), and the evidence talks about the character but never mentions this, it might be inconsistent (hallucinated claim). 
           BUT: Only check for consistency. If not contradicted, lean towards consistency unless it's clearly a made-up event not in the narrative flow.
           
        CRITICAL: Output ONLY the number 0 or 1. No explanation.
        Answer:
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Simple parsing
            text = response.text.strip()
            if "1" in text:
                return 1
            if "0" in text:
                return 0
            return 0 # Default to inconsistent if unclear
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            time.sleep(1) # Basic rate limit handling
            return 0
