# S2TD: An Adaptive, Self-Correcting RAG Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Groq](https://img.shields.io/badge/Groq-LPU_Inference-purple)
![RAG](https://img.shields.io/badge/Architecture-Agentic_RAG-green)

## 📖 Objective

**S2TD** is a "System 2" thinking agent designed to solve the common pitfalls of standard RAG (Retrieval-Augmented Generation) systems: **Hallucination** and **Irrelevance**.

Unlike a standard chatbot that blindly answers from retrieved documents, S2TD employs an **"Agentic Workflow"** that mimics a human research team:

1.  **Evaluates Data:** It grades retrieved documents for relevance.
2.  **Self-Corrects:** If data is poor, it automatically rewrites the query and searches the web.
3.  **Ensemble Thinking:** It uses a **"Council of LLMs"** (Creative, Critic, Summarizer) to generate diverse perspectives in parallel.
4.  **Synthesis:** A "Chairman" LLM merges these perspectives into a single, hallucination-free answer.

---

## 🏗️ Architecture

The system is built using **LangGraph** (State Machine), **Groq** (High-speed Llama 3 inference), and **ChromaDB** (Vector Storage).

```text
       [User Query]
            |
            v
    [Retrieve Documents]
            |
            v
   < Relevance Check > ----(No)----> [Web Search]
            |                              |
          (Yes)                            |
            |                              v
            +------------------------------+
            |             |                |
            v             v                v
    [Creative LLM]  [Critic LLM]   [Summarizer LLM]
            |             |                |
            +-------------+----------------+
                          |
                          v
                 [Chairman Synthesis]
                          |
                          v
                    [Final Answer]
```

---

## 📂 Project Structure

The project follows a modular, production-grade directory structure:

```text
S2TD_project/
├── .env                 # API Keys (Groq, Tavily, etc.)
├── requirements.txt     # Python dependencies
├── main.py              # Application Entry Point
└── S2TD/               # Core Application Logic
    ├── __init__.py
    ├── config.py        # Central configuration (LLMs, Tools)
    ├── state.py         # Graph State definition (TypedDict)
    ├── chains.py        # Prompt Engineering & LLM Chains
    ├── nodes.py         # Executable Graph Nodes (Functions)
    ├── graph.py         # Graph Wiring & Orchestration
    └── utils.py         # Database Indexing Script
```

---

## ⚡ Setup & Installation

### 1. Prerequisites

- Python 3.10 or higher.
- **Groq API Key** (Free at [console.groq.com](https://console.groq.com)).
- **Tavily API Key** (Free at [tavily.com](https://tavily.com)).

### 2. Clone & Environment

```bash
git clone [https://github.com/yourusername/S2TD.git](https://github.com/yourusername/S2TD.git)
cd S2TD
python -m venv venv
```

**Activate the Virtual Environment:**

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration (.env)

Create a `.env` file in the root directory and add your keys:

```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
# Optional: For tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2-...
```

---

## 🚀 How to Run

### Initialize the Knowledge Base

Before running the agent, you need to "teach" it by creating the vector database.
The system checks for this automatically, but you can force a rebuild:

```bash
python -m S2TD.utils
```

_This downloads the embedding model (HuggingFace) and indexes the sample URL defined in `utils.py`._

### Run the Agent

```bash
python main.py
```

_Enter your query when prompted. Watch the console logs to see the "thought process" of the agent._

---

### Author

Built by [Your Name].
