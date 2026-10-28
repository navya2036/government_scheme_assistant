# 🏛️ Government Scheme Assistant

A full-stack **Retrieval-Augmented Generation (RAG)** assistant that answers questions about government schemes using information retrieved from government scheme PDFs.

## ✨ Features

- Semantic search over government scheme documents
- E5 embeddings with FAISS vector retrieval
- Top-30 candidate retrieval with mMARCO cross-encoder reranking
- Final Top-5 context selection
- Grounded LLM answer generation
- Source and page-level citations
- Browser-based speech-to-text and text-to-speech
- React frontend with Node.js/Express API gateway
- Python/FastAPI backend
- Configurable LLM provider and model
- Automated verification and frontend tests

## 🏗️ Architecture

```text
Government Scheme PDFs
        ↓
     Chunking
        ↓
  E5 Embeddings
        ↓
   FAISS Index
        ↓
 Top-30 Retrieval
        ↓
mMARCO Reranking
        ↓
   Final Top-5
        ↓
 Grounded LLM
        ↓
 Answer + Citations
        ↓
    React UI
```

### Voice Flow

```text
User Speech
    ↓
Browser Speech Recognition
    ↓
Question
    ↓
Existing API + RAG Pipeline
    ↓
Generated Answer
    ↓
Browser Speech Synthesis
```

## 🔍 RAG Pipeline

```text
PDFs
 ↓
Text Extraction & Chunking
 ↓
E5 Document Embeddings
 ↓
FAISS Vector Index
 ↓
Top-30 Retrieval
 ↓
mMARCO Cross-Encoder
 ↓
Top-5 Relevant Chunks
 ↓
LLM Generation
 ↓
Answer + Citations
```

The E5 query format is:

```text
query: {user_question}
```

Documents use:

```text
passage: Scheme: {scheme_name}
{text}
```

The existing reranking API is:

```python
rerank.retrieve(question, candidate_k=30, final_k=5)
```

Each retrieved chunk preserves:

```text
chunk_id
scheme_name
source_file
page_start
page_end
context_text
```

## 📚 Grounding & Citations

The LLM receives the user's question and the five reranked chunks as reference context.

It is instructed to:

- Answer only from the supplied context
- Avoid inventing facts
- Avoid using outside knowledge
- Clearly state when the documents are insufficient
- Treat retrieved PDF text as reference material, not instructions

Citations are resolved from the retrieved chunk metadata rather than allowing the LLM to invent filenames or page numbers.

Example:

```text
[Source: example.pdf, p. 1]
[Source: example.pdf, pp. 1–2]
```

## 🎤 Voice Interaction

Voice functionality uses the browser's **Web Speech API**.

- **Speech-to-text:** `SpeechRecognition` / `webkitSpeechRecognition`
- **Text-to-speech:** `window.speechSynthesis`

Voice is an additional interface and does not change the existing RAG or API flow.

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React, Vite, JavaScript/JSX, CSS |
| **Voice** | Web Speech API |
| **API Gateway** | Node.js, Express |
| **Backend** | Python, FastAPI, Uvicorn, Pydantic |
| **Embeddings** | E5 |
| **Vector Search** | FAISS |
| **Reranking** | mMARCO Cross-Encoder |
| **LLM** | Configurable OpenAI-compatible LLM API |
| **Data** | PDF, JSON/JSONL, NumPy, SQLite |
| **Testing** | Vitest, Python verification scripts |

## 📁 Project Structure

```text
scheme-voice-assistant/
│
├── data/
│   ├── chunks.jsonl
│   ├── embeddings.npy
│   ├── embeddings_metadata.jsonl
│   ├── embeddings_config.json
│   ├── vector_index.faiss
│   ├── vector_index_metadata.sqlite
│   └── vector_index_config.json
│
├── retrieval.py
├── rerank.py
├── generation.py
├── api.py
├── verify_vector_index.py
├── verify_rerank.py
├── verify_generation.py
│
├── server/
│   └── ...
│
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
```

## ⚙️ LLM Configuration

The LLM is configured through environment variables:

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=provider_endpoint
LLM_MODEL=provider_model
```

The provider and model are not hardcoded into the RAG pipeline.

**Never commit `.env` or API keys to GitHub.**

## 🚀 Running Locally

### Python API

From the project root:

```bash
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

API:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

### Node.js / Express

Open another terminal:

```bash
cd server
npm install
npm start
```

### React Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally:

```text
http://localhost:5173
```

## 🧪 Testing

### Frontend

```bash
cd frontend
npm test
npm run build
```

### RAG Verification

```bash
python verify_vector_index.py
python verify_rerank.py
python verify_generation.py
```

The voice implementation is tested using mocked browser speech APIs.

## 🔐 Security

- API keys are provided through environment variables.
- Retrieved PDF content is treated as untrusted reference data.
- PDF content cannot override generation instructions.
- The LLM is instructed not to use outside knowledge.
- Citations are generated from retrieved metadata.

## ⚠️ Limitations

- Browser speech recognition support varies across browsers and devices.
- Voice recognition quality depends on the microphone, browser, and language.
- Answers are limited to information available in the indexed documents.
- Important government-scheme information should be verified against official sources.
