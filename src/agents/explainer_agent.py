import os
import sys
import json
from typing import Dict, List

# Add project root to Python path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.openai_client import OpenAIClient

class ExplainerAgent:
    """
    An agent that generates a natural language explanation for recommended products.
    """
    def __init__(self, prompt_template_path: str):
        """
        Initializes the ExplainerAgent with a prompt template.

        Args:
            prompt_template_path (str): The file path to the prompt template.
        """
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = f.read()
        self.client = OpenAIClient()

    def get_explanation(self, conversation_history: str, customer_prefs: Dict, candidates: List[Dict]) -> str:
        """
        Generates a friendly explanation for the recommended products.

        Args:
            conversation_history (str): The history of the conversation.
            customer_prefs (Dict): The dictionary of user preferences.
            candidates (List[Dict]): A list of recommended product dictionaries.

        Returns:
            str: A formatted, user-friendly explanation string, or a default message on failure.
        """
        # Convert dicts and lists to string representations for the prompt
        prefs_str = json.dumps(customer_prefs, indent=2)
        candidates_str = json.dumps(candidates, indent=2)

        # Format the prompt with the provided data
        system_prompt = self.prompt_template.format(
            conversation_history=conversation_history,
            customer_prefs_mapping=prefs_str,
            generated_candidates=candidates_str
        )

        # Create the message structure for the API call
        messages = [{"role": "user", "content": system_prompt}]

        # Get the response from the model
        response = self.client.get_chat_completion(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0.4
        )

        return response if response else "I found some items for you, but I'm having trouble explaining them right now."

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Define project paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    prompt_file_path = os.path.join(project_root, 'assets', 'prompts', 'explainer_prompt.text')
    
    # 2. Instantiate the agent
    explainer = ExplainerAgent(prompt_file_path)

    # 3. Provide sample inputs
    sample_conversation =  "Shopper: Something fancy for a birthday party\nAssistant: Sounds exciting! Could you let me know what category you're interested in? For example, are you looking for a dress, top, or something else? Also, do you have a size or budget in mind?\nShopper: dress, size Medium or small and under 150 dollars\nAssistant: Great choice! To help narrow it down, do you have a preference for the neckline style, like v-neck or round neck? Also, would you like the dress to be mini, midi, or maxi length?\nShopper: deep neck, mini pls"

    sample_prefs = {'category': ['dress'],
        'size': ['M', 'S'],
        'price_max': 150,
        'neckline': 'deep',
        'coverage_length': 'mini',
        'fit': 'bodycon',
        'fabric': ['satin', 'silk'],
        'occasion': 'party'}
    
    sample_candidates = [
        {
            'id': 'D002',
            'name': 'Ripple Linen Mini',
            'category': 'dress',
            'price': 95,
            'fit': 'relaxed',
            'fabric': 'linen',
            'sleeve_length': 'sleeveless',
            'available_sizes': 'XS,S,M,L,XL'
        },
        {
            'id': 'D015',
            'name': 'Sage Whisper Tiered Mini',
            'category': 'dress',
            'price': 98,
            'fit': 'relaxed',
            'fabric': 'linen blend',
            'sleeve_length': 'quarter sleeves',
            'available_sizes': 'XS,S,M'
        }
    ]

    # 4. Get the explanation
    explanation = explainer.get_explanation(sample_conversation, sample_prefs, sample_candidates)

    # 5. Print the result
    print("--- Generated Explanation ---")
    print(explanation)
    print("---------------------------") 