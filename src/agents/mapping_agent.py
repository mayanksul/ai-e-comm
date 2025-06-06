import os
import sys
import json
import re

# Add project root to Python path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.openai_client import OpenAIClient

class MappingAgent:
    """
    An agent that extracts structured information from a conversation history.
    It uses a GPT model to generate a JSON object of user preferences.
    """
    def __init__(self, prompt_file):
        """
        Initializes the MappingAgent.

        Args:
            prompt_file (str): Path to the file containing the system prompt for mapping.
        """
        with open(prompt_file, 'r') as f:
            self.system_prompt = f.read()
        self.client = OpenAIClient()

    def get_mapping(self, conversation_history: str) -> dict:
        """
        Generates a JSON mapping from the conversation history.

        Args:
            conversation_history (str): A string containing the full conversation.

        Returns:
            dict: A dictionary containing the extracted user preferences.
        """
        messages = [
            self.client.format_message("system", self.system_prompt),
            self.client.format_message("user", conversation_history)
        ]

        mapping_response = self.client.get_chat_completion(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0.3  # Lower temperature for more deterministic output
        )

        if not mapping_response:
            print("Error: Did not receive a mapping response from the model.")
            return {}

        return self._parse_json_response(mapping_response)

    def _parse_json_response(self, response: str) -> dict:
        """
        Cleans and parses the JSON string from the model's response.
        It removes comments and handles potential formatting issues.

        Args:
            response (str): The raw string response from the GPT model.

        Returns:
            dict: A parsed dictionary of the user preferences.
        """
        # Remove any markdown code block fences
        response = re.sub(r"```json\n?|\n?```", "", response)
        
        # Remove single-line // comments
        response_lines = [line.split('//')[0].strip() for line in response.split('\n') if line.strip()]
        
        # Join lines and attempt to parse
        json_str = ''.join(response_lines)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from response: {e}")
            print(f"Raw response was:\n{response}")
            return {}

if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. Define the project root and prompt file path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    prompt_file_path = os.path.join(project_root, 'assets', 'prompts', 'mapping_builder_prompt.text')
    
    # 2. Instantiate the agent
    mapper = MappingAgent(prompt_file_path)

    # 3. Provide a sample conversation history
    sample_conversation = """
    Shopper: Something casual for a summer brunch
    Agent: Lovely! Do you have a preference between dresses, tops & skirts, or something more casual like jeans?
    Shopper: Probably dresses or tops and skirts
    Agent: Any must-haves like sleeveless, budget range or size to keep in mind?
    Shopper: Want sleeveless, keep under $100, both S and M work
    """

    # 4. Get the mapping
    mapping = mapper.get_mapping(sample_conversation)

    # 5. Print the result
    print("--- Generated Mapping ---")
    print(json.dumps(mapping, indent=2))
    print("-------------------------") 