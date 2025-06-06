import os
import sys

# Add project root to Python path so we can import from `utils`
# This allows running the script directly from the command line
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.openai_client import OpenAIClient

class QueryAgent:
    """
    A conversational agent that helps users find clothing.
    It uses OpenAI's GPT models to understand user requests and ask clarifying questions.
    """
    def __init__(self, prompt_file):
        """
        Initializes the QueryAgent.

        Args:
            prompt_file (str): Path to the file containing the system prompt.
        """
        with open(prompt_file, 'r') as f:
            self.system_prompt = f.read()
        self.client = OpenAIClient()
        self.messages = [
            self.client.format_message("system", self.system_prompt)
        ]

    def _ask_gpt(self, user_input):
        """
        Sends the conversation history to GPT and gets a response.

        Args:
            user_input (str): The user's latest message.

        Returns:
            tuple: A tuple containing the agent's response (str) and the conversation history (list).
        """
        self.messages.append(self.client.format_message("user", user_input))
        response_content = self.client.get_chat_completion(
            messages=self.messages,
            model="gpt-4o-mini",
            temperature=0.5
        )
        if response_content:
            self.messages.append(self.client.format_message("assistant", response_content))
        return response_content, self.messages

    def run(self):
        """
        Runs the conversational loop with the user.
        The agent will ask up to two follow-up questions.

        Returns:
            str: The formatted conversation history string, or None if the conversation is cut short.
        """
        print("Agent: Hi! I'm your personal shopping assistant. What are you looking for today?")
        user_reply = input("You: ")

        for i in range(3): # Allow for initial query + 2 follow-ups
            agent_response, conversation_history = self._ask_gpt(user_reply)
            
            if not agent_response:
                print("Agent: I'm sorry, I seem to be having trouble connecting. Please try again later.")
                return

            # The prompt instructs the model to say "Great now hold on—..." when it has enough info.
            if "Great now hold on" in agent_response:
                print(f"Agent: {agent_response}")
                print("\nShopper preferences recorded. Now moving to mapping.")
                user_history = self.extract_and_print_history(conversation_history)
                return user_history

            if i < 2:
                print(f"Agent: {agent_response}")
                user_reply = input("You: ")
                if user_reply.lower() == "exit":
                    print("Exiting.")
                    return
            else:
                print("\nReached maximum follow-ups. Now moving to mapping.")
                conversation_history[-1]["content"] = "Great now hold on—let’s looking for the best options for you!"
                user_history = self.extract_and_print_history(conversation_history)
                return user_history

    def extract_and_print_history(self, conversation_history):
        """
        Formats and prints the conversation history.
        """
        print(f"Conversation history: {conversation_history[1:]}")

        user_history = "\n".join([
            item["role"].replace('user', "Shopper").replace("assistant", "Assistant") + ": " + item["content"]
            for item in conversation_history[1:] # Skip system prompt
        ])
        print("\n--- Conversation History for Mapping ---")

        return user_history

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    prompt_file_path = os.path.join(project_root, 'assets', 'prompts', 'conversation_builder_prompt.text')
    agent = QueryAgent(prompt_file_path)
    user_history = agent.run() 

    print(user_history)
    print("----------------------------------------")
