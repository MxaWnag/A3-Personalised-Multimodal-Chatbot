# A3 Personalised Multimodal Chatbot (Minimal MVP)

This repository provides an offline/online multimodal RAG pipeline aligned to A3 requirements:

- Offline indexing pipeline (chunking, captioning, embeddings, ChromaDB persistence)
- Online retrieval pipeline (query routing, text/image/hybrid search, context injection)
- Agent orchestration with memory-aware query expansion
- LLM answering via Ollama with grounded evidence fallback
- Quantitative evaluation with baseline + ablations

## Project Structure

- `data/sample/text_docs.jsonl`: text knowledge documents
- `data/sample/image_docs.jsonl`: image metadata/caption documents
- `data/user_profile.json`: user preferences for personalisation
- `data/templates/text_docs.template.jsonl`: template for course text docs
- `data/templates/image_docs.template.jsonl`: template for course image docs
- `data/templates/user_profile.template.json`: template for personalised study profile
- `src/mvp_agent.py`: online agent orchestration over indexed Chroma collections
- `src/pipeline/offline_index.py`: offline indexing CLI (text + image dual index)
- `src/pipeline/retrieval.py`: Chroma retrieval client (text/image/caption/hybrid)
- `src/pipeline/captioning.py`: LLaVA caption generation wrapper
- `src/pipeline/embeddings.py`: text + CLIP embedding wrappers
- `src/evaluate.py`: benchmark and metrics for V0/V1/V2

## System Variants

- `V0_plain_llm`: no retrieval, no memory
- `V1_rag_no_memory`: hybrid retrieval, no memory
- `V2_agent_router_memory`: router + hybrid retrieval + memory
- `V3_agent_router_memory_aligned`: router + memory + alignment-based retrieval (image-to-text projection)
- `V4_agent_router_memory_clip`: router + memory + CLIP text-embedding retrieval

## Offline Indexing

Run once after data updates:

```bash
python3 src/pipeline/offline_index.py
```

This writes indexes to `data/chroma` (or remote Chroma if `CHROMA_HOST` is set).

## Evaluation

```bash
pip install -r requirements.txt
python3 src/evaluate.py
```

If `langgraph` is not installed yet, the code uses a lightweight local fallback runner with the same node flow so you can still run experiments offline.

Expected output:

- Recall@5
- MRR
- Task success rate
- Keyword match score
- Evidence consistency score
- Avg latency (ms)
- Avg tool calls per query
- Token-usage proxy
- Per-family breakdown
- CSV files in `results/`

## Optional CLI Demo

```bash
python3 src/mvp_agent.py
```

Then type a question and choose a variant (`v0`, `v1`, `v2`, `v3`, `v4`).

## Chatbot App (FastAPI + Streamlit)

### 1) Start backend server

```bash
uvicorn src.server:app --reload --port 8000
```

### 2) Start frontend

```bash
python3 -m streamlit run src/streamlit_app.py
```

The Streamlit UI supports:

- variant switch (`v0`, `v1`, `v2`, `v3`, `v4`)
- session memory (same `session_id`)
- reset session
- runtime trace (`route`, retrieved ids, latency, tool calls, llm usage)

### Optional LLM Runtime

The answer node tries to call a local Ollama model at:

- `http://localhost:11434/api/generate`
- model: `llama3.1:8b`

If Ollama is unavailable, the system falls back to template-based grounded answers so experiments still run.
Evaluation output now includes:

- `LLMAvailable`: whether model endpoint is reachable in this run
- `LLMUsedRate`: fraction of queries that used live LLM response (not fallback)

## Online Agentic Retrieval Workflow

- `router`: classify query into text/image/hybrid retrieval
- `retrieval`: query Chroma collections (`text_chunks`, `image_clip`, `image_caption`)
- `top-k fusion`: merge candidates and prepare context
- `answer`: generate concise response using retrieved evidence
- `trace`: expose route, filters, backend, and retrieved ids in sidebar

### Variant Mapping

- `V0`: plain LLM (no retrieval)
- `V1`: text-only retrieval (no router)
- `V2`: router-driven retrieval + memory (main system)
- `V3`: image retrieval focused ablation
- `V4`: full hybrid retrieval (text + CLIP image + caption index)

## Recommended Data Format (Course-Material Personalised KB)

Use your own course materials as the main data source, and keep a consistent ingestion format.

- **Text docs (`jsonl`) fields**
  - `id`, `modality`, `course`, `title`, `content`, `topic_tags`, `week`, `source_file`
  - optional: `difficulty`
- **Image docs (`jsonl`) fields**
  - `id`, `modality`, `course`, `title`, `caption`, `topic_tags`, `week`, `source_image`
  - optional: `related_text_id`
- **User profile (`json`) fields**
  - study goals, weak topics, learning style, response preferences

### Quick Start with Templates

1. Fill the template files in `data/templates/` with your own content.
2. Copy the completed files into runtime paths:
   - `data/templates/text_docs.template.jsonl` -> `data/sample/text_docs.jsonl`
   - `data/templates/image_docs.template.jsonl` -> `data/sample/image_docs.jsonl`
   - `data/templates/user_profile.template.json` -> `data/user_profile.json`
3. Run:
   - `python3 src/evaluate.py`

4. Backend:
    - `uvicorn src.server:app --reload --port 8000`

5. Frontend:
    - `python3 -m streamlit run src/streamlit_app.py`

## Docker Deployment (Chroma + API + Streamlit + Ollama)

### Start all services

```bash
docker compose up --build -d
```

Services:

- API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`
- Chroma: `http://localhost:8001`
- Ollama API (host): `http://localhost:11435` (container internal: `11434`)

### Pull model into Ollama container (first run)

```bash
docker exec -it a3-ollama ollama pull llama3.1:8b
```

### Check logs

```bash
docker compose logs -f api
docker compose logs -f streamlit
docker compose logs -f chroma
docker compose logs -f ollama
```

### Stop

```bash
docker compose down
```
