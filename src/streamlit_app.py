import streamlit as st
import os
import json
import time

# Ensure the script can find the agent modules
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.query_agent import QueryAgent
from agents.mapping_agent import MappingAgent
from agents.recommendation_agent import RecommendationAgent
from agents.explainer_agent import ExplainerAgent

st.set_page_config(page_title="Conversational Shopping Assistant", page_icon="🛍️")
st.title("🛍️ Conversational Shopping Assistant")
st.caption("Your AI-powered guide to finding the perfect outfit.")

# --- Agent Initialization ---
@st.cache_resource
def initialize_agents():
    """
    Initializes all agents and loads necessary resources.
    Cached to run only once per session.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    paths = {
        "query": os.path.join(project_root, 'assets', 'prompts', 'conversation_builder_prompt.text'),
        "mapping": os.path.join(project_root, 'assets', 'prompts', 'mapping_builder_prompt.text'),
        "explainer": os.path.join(project_root, 'assets', 'prompts', 'explainer_prompt.text'),
        "catalog": os.path.join(project_root, 'data', 'raw', 'Apparels_shared.xlsx')
    }

    query_agent = QueryAgent(paths["query"])
    mapping_agent = MappingAgent(paths["mapping"])
    recommendation_agent = RecommendationAgent(paths["catalog"])
    explainer_agent = ExplainerAgent(paths["explainer"])
    
    return query_agent, mapping_agent, recommendation_agent, explainer_agent

query_agent, mapping_agent, recommendation_agent, explainer_agent = initialize_agents()


# --- Session State Management ---
if "messages" not in st.session_state:
    # This is the history for the model
    st.session_state.messages = query_agent.messages.copy() 
if "history" not in st.session_state:
    # This is the history for display
    st.session_state.history = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "user_turns" not in st.session_state:
    st.session_state.user_turns = 0

# --- UI: Display Chat History ---
for speaker, text in st.session_state.history:
    with st.chat_message(speaker):
        st.markdown(text)

# --- Full Pipeline Logic ---
def run_full_pipeline():
    """
    Executes the mapping, recommendation, and explanation steps.
    """
    st.session_state.processing = False # Prevent re-running

    # Format the conversation history string for the downstream agents
    conversation_history_str = "\n".join([
        f"{'Shopper' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in st.session_state.messages[1:]  # Skip system prompt
    ])

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your preferences..."):
            mapping_dict = mapping_agent.get_mapping(conversation_history_str)
            if not mapping_dict:
                st.error("Sorry, I couldn't understand your preferences. Could you please try again?")
                return
            st.write("✅ Preferences Analyzed")
            st.expander("View Extracted Preferences").json(mapping_dict)

        with st.spinner("Searching for the perfect items..."):
            recommendations = recommendation_agent.get_recommendations(mapping_dict, n=2)
            if not recommendations:
                st.warning("I couldn't find any items that perfectly match. Let me find the next best things!")
                # Optional: relax criteria and retry here
                st.error("Sorry, no products found with the current criteria.")
                return
            st.write("✅ Found Recommendations")
            st.expander("View Matching Products").json(recommendations)

        with st.spinner("Putting together the final response..."):
            final_explanation = explainer_agent.get_explanation(
                conversation_history=conversation_history_str,
                customer_prefs=mapping_dict,
                candidates=recommendations
            )
            st.write("✅ Done!")

        st.markdown(final_explanation)
        st.session_state.history.append(("assistant", final_explanation))


# --- UI: Chat Input and Main Interaction Logic ---
if prompt := st.chat_input("What are you looking for today?"):
    # Display user message
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Increment user turn counter and add message to model's history
    st.session_state.user_turns += 1
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_content = query_agent.client.get_chat_completion(
                messages=st.session_state.messages,
                model="gpt-4o-mini",
                temperature=0.5
            )
            
            if response_content:
                # Check if the conversation is complete (after 3 turns or by agent signal)
                if "Great now hold on" in response_content or st.session_state.user_turns >= 3:
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    st.session_state.history.append(("assistant", response_content))
                    st.markdown("Great now hold on—let's looking for the best options for you!")
                    st.session_state.processing = True
                
                else:
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    st.session_state.history.append(("assistant", response_content))
                    st.markdown(response_content)

                
            else:
                st.error("Sorry, I'm having trouble connecting. Please try again.")

# --- Trigger the pipeline if the conversation is complete ---
if st.session_state.processing:
    run_full_pipeline()