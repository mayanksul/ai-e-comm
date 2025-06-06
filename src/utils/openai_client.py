import os
from openai import OpenAI
from dotenv import load_dotenv

class OpenAIClient:
    def __init__(self):
        """Initialize the OpenAI client with API key from environment variables."""
        load_dotenv(override=True)
        self.client = OpenAI()
        
    def get_chat_completion(self, messages, model="gpt-4o-mini", temperature=0.3):
        """
        Get a chat completion from OpenAI.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            model (str): The model to use for completion
            temperature (float): Controls randomness in the response
            
        Returns:
            str: The assistant's response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting chat completion: {str(e)}")
            return None
    
    def format_message(self, role, content):
        """
        Format a message for the chat completion API.
        
        Args:
            role (str): The role of the message sender ('user' or 'assistant')
            content (str): The content of the message
            
        Returns:
            dict: Formatted message dictionary
        """
        return {"role": role, "content": content} 