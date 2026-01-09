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
            # Use a stable model, configurable via environment variable
            model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
            self.model = genai.GenerativeModel(model_name)

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
        # Chain-of-Thought (CoT) prompting for higher accuracy
        prompt = f"""
        You are an expert literary consistency analyst.
        
        Your Task: Determine if the CLAIM is consistent with the provided BOOK EXCERPTS.
        
        Claim regarding: {character if character else 'Unknown Character'}
        CLAIM: "{claim}"
        
        EVIDENCE from the book:
        {context_text}
        
        Instructions:
        1. Analyze the Claim: Break down the specific facts asserted (who, what, when, where).
        2. Analyze the Evidence: Does the evidence explicitly support or contradict these facts?
        3. Logic Check: 
           - If the text says "He died in 1890" and claim says "He died in 1900", that is a CONTRADICTION (0).
           - If the text describes him as "kind" and claim says "he was cruel", that is a CONTRADICTION (0).
           - If the text matches the claim details, it is CONSISTENT (1).
           - If the text is silent or irrelevant, standard assumption is CONSISTENT (1) unless the claim is wildly out of place for the genre/context.
        
        Output Format:
        First, write a one-sentence "Reasoning" explaining your logic.
        Then, on a new line, write "Final Answer: " followed strictly by 0 or 1.
        
        Example:
        Reasoning: The text explicitly states he died in 1890, but the claim says 1900.
        Final Answer: 0
        """
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Robust Parsing for CoT
            if "Final Answer: 1" in text or text.endswith("1"):
                return 1
            elif "Final Answer: 0" in text or text.endswith("0"):
                return 0
            
            # Fallback simple check
            if "0" in text and "1" not in text: return 0
            if "1" in text and "0" not in text: return 1
            
            return 0 # Default conservative

        except Exception as e:
            print(f"Error calling Gemini: {e}")
            time.sleep(1) # Basic rate limit handling
            return 0
