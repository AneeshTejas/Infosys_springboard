# Swiggy Review Intelligence Platform

## Project Overview

The Swiggy Review Intelligence Platform is an AI-driven analytics solution designed to transform unstructured customer feedback into actionable business intelligence. By integrating semantic search with Retrieval-Augmented Generation (RAG), the platform enables stakeholders to identify operational bottlenecks, sentiment trends, and specific service failures through a conversational interface and interactive dashboard.

---

## Key Features

* **Review Preprocessing:** Automated cleaning and normalization of raw customer feedback.
* **Sentiment and Aspect Tagging:** High-precision classification of reviews based on emotional tone and service categories (e.g., delivery speed, food quality).
* **Semantic Search:** FAISS-powered vector database allowing for contextual retrieval of reviews beyond simple keyword matching.
* **RAG-based AI Assistant:** A natural language interface for querying complex patterns such as refund issues or support quality.
* **Interactive Dashboard:** A comprehensive Streamlit interface for visualizing high-level analytics and granular reports.

---

## System Architecture

The system follows a modular architecture to separate data ingestion, vector storage, and the application layer.

### File Structure

* **app.py**: Main entry point for the Streamlit dashboard.
* **analytics.py**: Core logic for generating visual insights and trend data.
* **rag_chain.py**: LLM orchestration and RAG pipeline management.
* **retriever.py**: Functions for querying the vector store.
* **faiss_store/**: Directory containing the local vector database (`index.faiss`, `metadata.pkl`).
* **pages/**: Multi-page Streamlit application structure.
* `2_AI_Assistant.py`: Interface for the RAG chatbot.
* `4_Reports_Alerts.py`: Module for automated reporting and critical alerts.


* **utils/**: Directory for modular helper scripts (`analytics_utils.py`, `rag_utils.py`, `retriever_utils.py`).

---

## Technical Specifications

| Component | Technology |
| --- | --- |
| **Language** | Python |
| **Web Framework** | Streamlit |
| **Vector Database** | FAISS |
| **Orchestration** | LangChain |
| **Embeddings** | HuggingFace |
| **LLM API** | Groq (Llama 3 / Mixtral) |

---

## Setup and Installation

### 1. Repository Initialization

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

```

### 2. Virtual Environment Configuration

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate

```

**Mac/Linux:**

```bash
python -m venv venv
source venv/bin/activate

```

### 3. Dependency Installation

```bash
pip install -r requirements.txt

```

### 4. Environment Variables

To enable AI functionalities, configure your API keys as environment variables.
**Windows:**

```bash
set GROQ_API_KEY=your_api_key_here

```

**Mac/Linux:**

```bash
export GROQ_API_KEY=your_api_key_here

```

### 5. Vector Store Initialization

Ensure that the `faiss_store/` directory contains the required index files. If the files are missing, run the build script:

```bash
python build_faiss.py

```

### 6. Execution

```bash
streamlit run app.py

```

---

## Deployment Guidelines

* **Cloud Hosting:** Use CPU-optimized builds for libraries like Torch to comply with Streamlit Cloud resource limits.
* **API Security:** Ensure `.env` files or hardcoded keys are not included in version control.
* **Data Management:** Maintain FAISS index files under 100 MB for faster loading and deployment stability.
* **Path Management:** Always utilize relative paths within the codebase to ensure cross-platform compatibility.
