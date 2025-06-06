import os
import json

from src.agents.query_agent import QueryAgent
from src.agents.mapping_agent import MappingAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.explainer_agent import ExplainerAgent

def main():
    """
    Main orchestration function to run the full agent pipeline.
    It guides a user through a conversation to find and get recommendations for clothing.
    """
    # --- 1. Setup Paths ---
    project_root = os.path.abspath(os.path.dirname(__file__))
    query_prompt_path = os.path.join(project_root, 'assets', 'prompts', 'conversation_builder_prompt.text')
    mapping_prompt_path = os.path.join(project_root, 'assets', 'prompts', 'mapping_builder_prompt.text')
    explainer_prompt_path = os.path.join(project_root, 'assets', 'prompts', 'explainer_prompt.text')
    catalog_path = os.path.join(project_root, 'data', 'raw', 'Apparels_shared.xlsx')

    # --- 2. Instantiate Agents ---
    print("Initializing agents...")
    query_agent = QueryAgent(query_prompt_path)
    mapping_agent = MappingAgent(mapping_prompt_path)
    recommendation_agent = RecommendationAgent(catalog_path)
    explainer_agent = ExplainerAgent(explainer_prompt_path)
    print("All agents ready.\n")

    # --- 3. Step 1: Run Query Agent to get user preferences ---
    conversation_history_str = query_agent.run()

    if not conversation_history_str:
        print("\nConversation ended prematurely. Exiting.")
        return

    # --- 4. Step 2: Run Mapping Agent to extract structured data ---
    print("\n\n--- Extracting Preferences ---")
    mapping_dict = mapping_agent.get_mapping(conversation_history_str)
    
    if not mapping_dict:
        print("Could not extract preferences from the conversation. Exiting.")
        return
        
    print("Extracted Preferences:")
    print(json.dumps(mapping_dict, indent=2))
    
    # --- 5. Step 3: Run Recommendation Agent to find products ---
    print("\n--- Finding Recommendations ---")
    recommendations = recommendation_agent.get_recommendations(mapping_dict, n=2)

    if not recommendations:
        print("\nSorry, I couldn't find any items that match your preferences. Please try again with different criteria.")
        return

    print("Found Top Recommendations:")
    print(json.dumps(recommendations, indent=2))
    
    # --- 6. Step 4: Run Explainer Agent to generate the final response ---
    print("\n--- Generating Final Explanation ---")
    final_explanation = explainer_agent.get_explanation(
        conversation_history=conversation_history_str,
        customer_prefs=mapping_dict,
        candidates=recommendations
    )

    print("\n==============================================")
    print("Stylist Says:")
    print("==============================================")
    print(final_explanation)
    print("==============================================")

if __name__ == '__main__':
    main() 