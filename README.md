# Automated LLM Evaluation Engine

An automated MLOps pipeline designed to score the outputs of LLM applications for factual accuracy, context precision, and safety guardrails. 

Instead of just validating static schemas, this framework uses an "LLM-as-a-judge" architecture running entirely locally to evaluate reasoning trajectories and catch hallucinations.

## Tech Stack
* **Language:** Python
* **Backend Framework:** FastAPI (Asynchronous REST endpoints with strict Pydantic schema validation)
* **The "Judge" Model:** Llama 3 (8B) running locally via Ollama for zero-cost, private inference
* **Data Storage:** SQLite & SQLAlchemy for persistence logging
* **Frontend/Dashboard:** Streamlit for real-time visualization and batch processing

## Architecture
1. **Core Evaluator (`evaluator.py`):** Forces the local Llama 3 model to output strict JSON scorecards across three metrics: Faithfulness, Answer Relevancy, and Toxicity.
2. **API Layer (`main.py`):** Ingests evaluation requests, triggers the local judge, and logs results to the database.
3. **Database (`database.py`):** Maintains a historical ledger of all evaluations to track reliability metrics over time.
4. **Dashboard (`app.py`):** A custom-styled Streamlit UI that pulls metrics from the database and allows for both single-prompt manual testing and CSV batch processing.

## How to Run Locally
1. Ensure [Ollama](https://ollama.com/) is installed and the Llama 3 model is pulled (`ollama run llama3`).
2. Install dependencies: `pip install -r requirements.txt`
3. Start the FastAPI backend: `python -m uvicorn main:app --reload`
4. In a separate terminal, start the Streamlit dashboard: `python -m streamlit run app.py`