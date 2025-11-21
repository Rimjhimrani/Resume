# AI-Powered Resume Screening Tool
A smart recruitment assistant powered by RAG (Retrieval-Augmented Generation), FAISS vector search, and OpenAI LLMs.  
This tool evaluates resumes against job descriptions, scores alignment, identifies strengths & gaps, and allows interactive Q&A about the candidate.

---

## Features

- Upload Resume & Job Description
- Automatically extract text from PDFs
- Generates:
  - Match Score
  - Strengths & Skill Gaps
  - Interactive Q&A using Retrieval-Augmented Generation
- Uses:
  - FAISS vector index for retrieval
  - Sentence-Transformer embeddings
  - OpenAI chat completions for reasoning

---

## Tech Stack

| Component | Technology |
|----------|------------|
| Framework | Streamlit |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | FAISS |
| LLM | OpenAI Chat Completions |
| Parsing | PyPDF2 |

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd <project-folder>
pip install -r requirements.txt
