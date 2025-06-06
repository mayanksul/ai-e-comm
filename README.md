# E-Commerce Conversational AI Shopping Assistant

This project implements a sophisticated, multi-agent recommendation system that helps users find clothing items through a natural, conversational interface. The system is available as both an interactive Streamlit web application and a command-line interface (CLI).

## Project Structure

The project is organized into several key directories:

```
ai-e-comm/
├── assets/
│   └── prompts/            # Contains prompt templates for the AI agents
├── data/
│   └── raw/                # Raw data files (e.g., product catalog)
├── src/
│   ├── agents/             # Core AI agent modules
│   │   ├── query_agent.py
│   │   ├── mapping_agent.py
│   │   ├── recommendation_agent.py
│   │   └── explainer_agent.py
│   ├── utils/              # Utility functions (e.g., OpenAI client)
│   └── streamlit_app.py    # The Streamlit web application
├── main.py                 # The command-line interface (CLI) application
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Setup Instructions

Follow these steps to set up and run the project locally.

### 1. Create and Activate a Virtual Environment

**Option A: Using Conda (Recommended)**
```bash
conda create -n "ai-e-comm" python=3.10 -y
conda activate ai-e-comm
```

**Option B: Using Python's `venv`**
```bash
python3 -m venv ai-e-comm
source ai-e-comm/bin/activate  # On Windows: .\ai-e-comm\Scripts\activate
```

### 2. Set Up Your OpenAI API Key

This project requires an OpenAI API key to function.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your OpenAI API key to the `.env` file like this:

    ```
    OPENAI_API_KEY='your-secret-api-key-goes-here'
    ```

The application uses the `python-dotenv` library to automatically load this key at runtime.

### 3. Install Dependencies

Install all the required Python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## How to Run the Application

You can run the shopping assistant in two ways:

### A. Streamlit Web Interface (Recommended)

This provides a rich, interactive chat experience.

1.  Run the following command in your terminal:
    ```bash
    streamlit run src/streamlit_app.py
    ```
2.  Your web browser will open with the application.
3.  **To start a new conversation, simply refresh the browser tab.**

### B. Command-Line Interface (CLI)

This runs the same agent pipeline directly in your terminal.

1.  Run the `main.py` script:
    ```bash
    python main.py
    ```
2.  Interact with the agent by typing your responses in the terminal.

## Features

- **Conversational AI:** A friendly, chatbot-style interface for product discovery.
- **Multi-Agent Pipeline:**
  - **QueryAgent:** Asks clarifying questions to understand user needs.
  - **MappingAgent:** Extracts structured JSON preferences from the conversation.
  - **RecommendationAgent:** Scores and retrieves the best-matching products from the catalog.
  - **ExplainerAgent:** Generates a warm, natural language summary explaining the recommendations.
- **Dual Interfaces:** Choose between an interactive Streamlit web app or a terminal-based CLI.

## Future Enhancements
- User preference learning across sessions
- Advanced filtering and sorting options
- Image-based recommendations
- Integration with live inventory databases 