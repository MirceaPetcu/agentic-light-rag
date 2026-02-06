# 🤖 Multi-Agent LightRAG Framework

A sophisticated **Retrieval-Augmented Generation (RAG)** system built on a **multi-agent architecture** that iteratively refines knowledge retrieval until convergence. This framework combines LightRAG's knowledge graph capabilities with an agentic pipeline for intelligent, self-improving query answering.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Services](#-services)
  - [Backend Services](#backend-services)
  - [LightRAG Microservice](#lightrag-microservice)
  - [ColBERT Service](#colbert-service)
  - [Frontend Application](#frontend-application)
- [Multi-Agent Pipeline](#-multi-agent-pipeline)
- [Databases](#-databases)
- [Deployment](#-deployment)
  - [Prerequisites](#prerequisites)
  - [Quick Start with Docker Compose](#quick-start-with-docker-compose)
  - [Environment Configuration](#environment-configuration)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Development](#-development)

---

## 🎯 Overview

The Multi-Agent LightRAG Framework is designed to provide high-quality answers to complex queries through an iterative retrieval process. Unlike traditional RAG systems that perform a single retrieval pass, this system:

1. **Decomposes** complex queries into atomic sub-queries
2. **Retrieves** relevant context from a knowledge graph
3. **Infers** both answers and the addressed query from context
4. **Judges** whether the retrieval is sufficient using semantic similarity
5. **Iterates** with refined sub-queries until convergence
6. **Generates** comprehensive, cited responses

### Key Features

- 🔄 **Iterative Refinement**: Self-improving retrieval with convergence detection
- 🧠 **Multi-Agent Architecture**: Specialized agents for each step of the process
- 📊 **Knowledge Graph RAG**: Powered by LightRAG for entity and relationship-aware retrieval
- 📝 **Structured Output**: vLLM with Outlines for reliable JSON generation
- 🎯 **ColBERT Similarity**: Precise semantic matching for convergence checking
- 📚 **Citation Support**: Full source attribution in responses
- 🖥️ **Modern Frontend**: React-based chat interface with real-time streaming

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React/Vite)                          │
│                                   Port 3000                                 │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ /api/*
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN API GATEWAY (app.py)                         │
│                                   Port 8000                                 │
│  • /ingest - Document ingestion                                             │
│  • /query - Multi-agent pipeline queries                                    │
│  • /query/simple - Direct LightRAG queries                                  │
│  • /health - Health checks                                                  │
└────────────────┬──────────────────────────────────────┬────────────────────┘
                 │                                      │
                 ▼                                      ▼
┌────────────────────────────────┐    ┌────────────────────────────────────────┐
│   PIPELINE SERVICE (pipeline.py)│    │    LIGHTRAG SERVICE (lightrag_fastapi.py)│
│         Port 8003               │    │              Port 8005                  │
│                                 │    │                                        │
│  Multi-Agent Orchestration:     │    │  Knowledge Graph Operations:           │
│  • QueryRewriter Agent          │    │  • Document ingestion                  │
│  • Retriever Agent              │◄───┤  • Graph queries (local/global/hybrid) │
│  • Deducer Agent                │    │  • Entity extraction                   │
│  • Judge Agent                  │    │  • Relationship mapping                │
│  • Response Agent               │    │                                        │
└────────────┬───────────────────┘    └──────────────┬─────────────────────────┘
             │                                       │
             ▼                                       │
┌────────────────────────────────┐                   │
│  COLBERT SERVICE (colbert_service.py)│             │
│         Port 8002               │                  │
│                                 │                  │
│  • Semantic similarity scoring  │                  │
│  • Convergence detection        │                  │
│  • OpenAI-compatible embeddings │                  │
└─────────────────────────────────┘                  │
                                                     │
             ┌───────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INFRASTRUCTURE                                   │
├─────────────────────┬─────────────────────┬─────────────────────────────────┤
│    Neo4j (7474)     │   MongoDB (27017)   │     PostgreSQL (5432)          │
│    Graph Storage    │   Document Store    │     Status Tracking            │
├─────────────────────┴─────────────────────┴─────────────────────────────────┤
│                            LLM SERVICES                                     │
├─────────────────────────────────────────────┬───────────────────────────────┤
│           vLLM (8001)                       │        Ollama (11435)         │
│    Main LLM Inference                       │       Embeddings              │
│    Structured Output (Outlines)             │    (qwen3-embedding:0.6b)     │
└─────────────────────────────────────────────┴───────────────────────────────┘
```

---

## 🔧 Services

### Backend Services

#### Main API Gateway (`app.py`)

The primary entry point for all client requests. Handles:

- **Document Ingestion**: Upload and process documents into the knowledge graph
- **Query Routing**: Proxies requests to the appropriate backend services
- **Health Monitoring**: Aggregated health status of all services
- **SSE Streaming**: Real-time query progress updates

**Port**: `8000`

**Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/ingest` | POST | Ingest text content |
| `/ingest/file` | POST | Ingest file (multipart) |
| `/query` | POST | Multi-agent pipeline query |
| `/query/simple` | POST | Simple LightRAG query |
| `/documents/status` | POST | Get document processing status |
| `/reasoning_trace/{query_id}` | GET | Get reasoning trace for a query |

#### Pipeline Service (`pipeline.py`)

Orchestrates the multi-agent pipeline with Server-Sent Events (SSE) for real-time progress updates.

**Port**: `8003`

**Features**:
- Manages agent lifecycle and communication
- Streams progress events (step updates, agent actions, final results)
- Handles convergence detection and iteration control
- Stores reasoning traces for debugging

---

### LightRAG Microservice

#### LightRAG FastAPI Server (`lightrag_fastapi.py`)

A dedicated microservice wrapping the LightRAG library for knowledge graph operations.

**Port**: `8005`

**Capabilities**:
- **Document Processing**: Chunking, entity extraction, relationship mapping
- **Query Modes**: 
  - `local`: Entity-focused retrieval
  - `global`: High-level summarization
  - `hybrid`: Combined local and global
  - `naive`: Traditional vector similarity
  - `mix`: Reranked combination of all modes
- **Graph Storage**: Neo4j integration for persistent knowledge graphs
- **Streaming**: Optional streaming responses for long queries

**Configuration**:
```bash
LIGHTRAG_WORKING_DIR=/app/rag_storage
LIGHTRAG_GRAPH_STORAGE=Neo4JStorage
LLM_BINDING_HOST=http://vllm:8000/v1
EMBEDDING_BINDING_HOST=http://ollama:11434
```

---

### ColBERT Service

#### ColBERT Embeddings Service (`colbert_service.py`)

Provides semantic similarity scoring using the ColBERT v2 model for convergence detection.

**Port**: `8002`

**Features**:
- **OpenAI-Compatible API**: Drop-in replacement for embedding endpoints
- **MaxSim Similarity**: ColBERT's late interaction scoring
- **Token-Level Embeddings**: Rich contextual representations

**Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/embeddings` | POST | Get embeddings for text(s) |
| `/similarity` | POST | Compute similarity between query and document |
| `/health` | GET | Service health check |

**Usage in Pipeline**:
The Judge Agent uses ColBERT to compare the original query with the inferred query from retrieved context. When similarity exceeds the threshold (default: 0.75), the retrieval is considered converged.

---

### Frontend Application

A modern React SPA built with Vite, TypeScript, and TailwindCSS.

**Port**: `3000`

**Technology Stack**:
- **React 19** with TypeScript
- **TailwindCSS 4** for styling
- **React Query** for data fetching
- **React Router** for navigation
- **Lucide React** for icons

**Pages**:
| Route | Component | Description |
|-------|-----------|-------------|
| `/` | ChatPage | Main chat interface with streaming responses |
| `/query` | QueryPage | Advanced query with parameter controls |
| `/ingest` | IngestPage | Document upload and ingestion |

**Features**:
- 💬 Real-time chat with streaming responses
- 📎 File attachment support
- 🔗 Inline citations with expandable references
- 🧠 Reasoning trace visualization
- ⚙️ Configurable query parameters (max steps, similarity threshold, top-k)
- 📱 Responsive design

---

## 🤖 Multi-Agent Pipeline

The core of the system is a sophisticated multi-agent pipeline that iteratively refines retrieval:

### Agent Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            USER QUERY                                   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Step 1-2: QUERY REWRITER AGENT   (OPTIONAL)          │
│                                                                         │
│  • Decomposes complex query into atomic sub-queries                     │
│  • Assigns priority and keywords to each sub-query                      │
│  • Preserves user intent while maximizing retrieval coverage            │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ Sub-queries
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Step 3: RETRIEVER AGENT                           │
│                                                                         │
│  • Queries LightRAG knowledge graph with each sub-query                 │
│  • Applies Reciprocal Rank Fusion (RRF) to merge results                │
│  • Returns top-k fused contexts with citations                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ Retrieved Context
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Steps 4-5: DEDUCER AGENT                             │
│                                                                         │
│  • Step 4: Infer possible answer from context                           │
│  • Step 5: Infer what query the context addresses                       │
│  • Runs steps 4 & 5 in parallel for efficiency                          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ Inferred Answer + Inferred Query
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Steps 6-8: JUDGE AGENT                            │
│                                                                         │
│  • Step 6: Compute ColBERT similarity (original vs inferred query)      │
│  • If similarity >= threshold: CONVERGED → proceed to response          │
│  • Step 7: Identify missing context using LLM                           │
│  • Step 8: Generate new sub-queries from gaps                           │
└───────────────────────┬────────────────────────┬────────────────────────┘
                        │ Not Converged          │ Converged
                        │ (loop back)            │
                        ▼                        ▼
              ┌─────────────────┐    ┌────────────────────────────────────┐
              │  Step 3 (retry) │    │        RESPONSE AGENT              │
              │  with new       │    │                                    │
              │  sub-queries    │    │  • Generates comprehensive answer  │
              └─────────────────┘    │  • Includes inline citations       │
                                     │  • Provides confidence score       │
                                     │  • Lists key points & limitations  │
                                     └────────────────────────────────────┘
```

### Agents

#### 1. QueryRewriterAgent (`query_rewriter.py`)

Decomposes user queries into structured sub-queries using vLLM with guided JSON output.

**Input**: User query string  
**Output**: `QueryDecomposition` with list of `SubQuery` objects

```python
class SubQuery(BaseModel):
    id: int                      # Sequential ID
    question: str                # Fact-focused question
    canonical_form: str          # Normalized for retrieval
    requires_retrieval: bool     # Needs external sources?
    evidence_types: List[str]    # Types of evidence needed
    rationale: str               # Why this sub-query is needed
    priority: int                # 1-5 (1 = highest)
    keywords: List[str]          # Search terms
```

#### 2. RetrieverAgent (`retriever.py`)

Retrieves context from LightRAG and applies Reciprocal Rank Fusion (RRF).

**Input**: List of sub-queries  
**Output**: `RetrieverResult` with fused contexts and RRF scores

**Features**:
- Parallel retrieval for multiple sub-queries
- RRF fusion to combine rankings from different queries
- Context deduplication using content hashing
- Citation metadata preservation (file paths, entity names)

#### 3. DeducerAgent (`deducer_model.py`)

Infers both the answer and the original query from retrieved context.

**Input**: Retrieved context  
**Output**: `InferredAnswer` + `InferredQuery`

**Parallel Processing**: Steps 4 and 5 run concurrently using `asyncio.gather()`.

#### 4. JudgeAgent (`judge.py`)

Determines convergence and generates new sub-queries if needed.

**Input**: Original query, inferred query, context  
**Output**: `JudgementResult` with convergence status

**Convergence Check**:
```python
similarity = await colbert.compute_similarity(original_query, inferred_query)
converged = similarity >= similarity_threshold  # Default: 0.75
```

**Gap Analysis**: If not converged, uses LLM to identify missing information and generate targeted sub-queries.

#### 5. ResponseAgent (`respone_agent.py`)

Generates the final response with full citations.

**Input**: All accumulated context, inferred answers, user query  
**Output**: `GeneratedResponse` with inline citations

**Response Structure**:
- Comprehensive answer with `[1]`, `[2]` citations
- List of `Citation` objects with source details
- Confidence score (0-1)
- Limitations and key points

---

## 🗄 Databases

### Neo4j (Graph Storage)

Stores the LightRAG knowledge graph with entities and relationships.

**Port**: `7474` (HTTP Browser), `7687` (Bolt)

**Credentials**:
```bash
NEO4J_AUTH=neo4j/password
```

**Access**: http://localhost:7474

### MongoDB (Document Storage)

Used by LightRAG for document chunk storage and metadata.

**Port**: `27017`

**Credentials**:
```bash
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=password
MONGO_DATABASE=lightrag
```

### PostgreSQL (Status Tracking)

Tracks document processing status for the ingestion pipeline.

**Port**: `5432`

**Credentials**:
```bash
POSTGRES_USER=admin
POSTGRES_PASSWORD=password
POSTGRES_DB=lightrag
```

**Schema**:
```sql
CREATE TABLE documents (
    document_id VARCHAR PRIMARY KEY,
    document_status VARCHAR,
    file_path VARCHAR,
    updated_at TIMESTAMP
);
```

---

## 🚀 Deployment

### Prerequisites

- **Docker** 20.10+ with Docker Compose v2
- **NVIDIA GPU** (optional, for vLLM acceleration)
- **8GB+ RAM** minimum (16GB+ recommended)
- **20GB+ disk space** for models and data

### Quick Start with Docker Compose

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd is
   ```

2. **Create environment file**:
   ```bash
   cp env.example .env
   ```

3. **Configure environment** (edit `.env`):
   ```bash
   # Required: Set your LLM API key if using external LLM
   LLM_BINDING_API_KEY=your-api-key
   
   # Optional: Adjust model settings
   VLLM_MODEL=LiquidAI/LFM2-2.6B
   EMBEDDING_MODEL=qwen3-embedding:0.6b
   ```

4. **Start all services**:
   ```bash
   docker compose up -d
   ```

5. **Verify services**:
   ```bash
   # Check all containers are running
   docker compose ps
   
   # View logs
   docker compose logs -f
   ```

6. **Access the application**:
   - **Frontend**: http://localhost:3000
   - **API Gateway**: http://localhost:8000
   - **Neo4j Browser**: http://localhost:7474

### Service Startup Order

The services have the following dependency chain (though explicit `depends_on` is commented out for flexibility):

```
1. Databases: Neo4j, MongoDB, PostgreSQL
2. LLM Services: vLLM, Ollama
3. ColBERT Service
4. LightRAG Service
5. Pipeline Service
6. Main API (app)
7. Frontend
```

### Individual Service Deployment

To run services individually for development:

```bash
# Start infrastructure only
docker compose up -d neo4j mongodb postgres ollama

# Run backend locally
pip install -r requirements.txt
pip install -e ./light_rag
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Run frontend locally
cd frontend
npm install
npm run dev
```

---

### Environment Configuration

#### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LIGHTRAG_WORKING_DIR` | `./rag_storage` | Storage directory for LightRAG |
| `LIGHTRAG_GRAPH_STORAGE` | `Neo4JStorage` | Graph backend (Neo4JStorage, NetworkXStorage) |
| `MAX_STEPS` | `5` | Maximum iteration steps in pipeline |
| `SIMILARITY_THRESHOLD` | `0.75` | ColBERT convergence threshold |
| `TOP_K` | `10` | Number of results to retrieve |

#### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BINDING_HOST` | `http://vllm:8000/v1` | LLM API endpoint |
| `LLM_MODEL` | `LiquidAI/LFM2-2.6B` | Model for inference |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM server URL |
| `VLLM_API_KEY` | `EMPTY` | vLLM API key |

#### Embedding Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Ollama embedding model |
| `EMBEDDING_BINDING_HOST` | `http://ollama:11434` | Ollama server URL |
| `EMBEDDING_DIM` | `1024` | Embedding dimension |

#### Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `MONGO_URI` | `mongodb://admin:password@mongodb:27017` | MongoDB connection URI |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_DB` | `lightrag` | PostgreSQL database |

#### Service Ports

| Service | Default Port | Environment Variable |
|---------|--------------|----------------------|
| Frontend | 3000 | `FRONTEND_PORT` |
| Main API | 8000 | `PORT` |
| Pipeline | 8003 | `PIPELINE_PORT` |
| LightRAG | 8005 | `LIGHTRAG_PORT` |
| ColBERT | 8002 | `COLBERT_PORT` |
| vLLM | 8001 | `VLLM_PORT` |
| Ollama | 11435 | `OLLAMA_PORT` |
| Neo4j HTTP | 7474 | `NEO4J_HTTP_PORT` |
| Neo4j Bolt | 7687 | `NEO4J_BOLT_PORT` |
| MongoDB | 27017 | `MONGODB_PORT` |
| PostgreSQL | 5432 | `POSTGRES_PORT` |

---

## 📖 API Reference

### Health Check

```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "rag_initialized": true,
  "pipeline_service_healthy": true
}
```

### Document Ingestion

#### Ingest Text

```bash
POST /ingest
Content-Type: application/json

{
  "content": "Document text content...",
  "doc_id": "optional-doc-id",
  "file_path": "optional/file/path.txt"
}
```

#### Ingest File

```bash
POST /ingest/file
Content-Type: multipart/form-data

file: <binary file data>
```

### Query Endpoints

#### Multi-Agent Query

```bash
POST /query
Content-Type: application/json

{
  "query": "What is the relationship between X and Y?",
  "max_steps": 5,
  "similarity_threshold": 0.75,
  "top_k": 10,
  "stream": true
}
```

**Response**:
```json
{
  "status": "success",
  "response": "Based on the retrieved information [1], X and Y are related through...",
  "citations": [
    {
      "id": 1,
      "source_type": "entity",
      "source_name": "X",
      "excerpt": "X is defined as...",
      "file_path": "documents/source.pdf"
    }
  ],
  "confidence": 0.85,
  "steps_taken": 3,
  "converged": true,
  "metadata": {
    "query_id": "abc123",
    "processing_time_ms": 2500
  }
}
```

#### Simple Query

```bash
POST /query/simple
Content-Type: application/json

{
  "query": "What is X?",
  "top_k": 10
}
```

### Reasoning Trace

```bash
GET /reasoning_trace/{query_id}
```

**Response**:
```json
{
  "query_id": "abc123",
  "steps": [
    {
      "step": 1,
      "agent": "QueryRewriter",
      "action": "decompose",
      "output": { "subqueries": [...] }
    },
    {
      "step": 2,
      "agent": "Retriever",
      "action": "retrieve",
      "output": { "contexts": [...] }
    }
  ],
  "final_result": { ... }
}
```

---

## ⚙ Configuration

### LightRAG Configuration

The LightRAG instance can be configured via environment variables or the `config.ini` file:

```ini
[lightrag]
working_dir = ./rag_storage
graph_storage = Neo4JStorage
kv_storage = JsonKVStorage
vector_storage = NanoVectorDBStorage

[llm]
binding_host = http://vllm:8000/v1
model = LiquidAI/LFM2-2.6B
api_key = EMPTY

[embedding]
model = qwen3-embedding:0.6b
host = http://ollama:11434
dim = 1024
```

### Pipeline Tuning

| Parameter | Range | Impact |
|-----------|-------|--------|
| `max_steps` | 1-10 | More steps = better coverage, slower |
| `similarity_threshold` | 0.5-0.95 | Lower = stricter convergence |
| `top_k` | 5-50 | More results = richer context |

---

## 🛠 Development

### Project Structure

```
is/
├── app.py                    # Main API gateway
├── pipeline.py               # Multi-agent pipeline service
├── lightrag_fastapi.py       # LightRAG microservice
├── colbert_service.py        # ColBERT embeddings service
├── base_agent.py             # Base agent class
├── query_rewriter.py         # Query decomposition agent
├── retriever.py              # Knowledge retrieval agent
├── deducer_model.py          # Answer/query inference agent
├── judge.py                  # Convergence checking agent
├── respone_agent.py          # Response generation agent
├── observation.py            # Inter-agent data structure
├── prompts.py                # LLM prompts
├── requirements.txt          # Python dependencies
├── requirements.colbert.txt  # ColBERT dependencies
├── docker-compose.yml        # Service orchestration
├── Dockerfile.app            # Main app/pipeline image
├── Dockerfile.frontend       # Frontend image
├── Dockerfile.lightrag_fastapi # LightRAG image
├── Dockerfile.colbert        # ColBERT image
├── env.example               # Environment template
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/            # Route pages
│   │   ├── components/       # UI components
│   │   ├── services/         # API client
│   │   ├── contexts/         # React contexts
│   │   └── types/            # TypeScript types
│   ├── package.json
│   └── nginx.conf
├── light_rag/                # LightRAG submodule
└── rag_storage/              # Knowledge graph storage
```

### Running Tests

```bash
# Backend tests
pytest tests/

# Frontend tests
cd frontend && npm test
```

### Adding New Agents

1. Create a new agent file extending `BaseAgent`:
   ```python
   from base_agent import BaseAgent
   
   class MyAgent(BaseAgent):
       async def act(self, observation):
           # Process observation
           return result
       
       async def think(self, data):
           # Internal reasoning
           return analysis
   ```

2. Register the agent in `pipeline.py`

3. Update the `Observation` dataclass if needed

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ./light_rag

# Start services (databases + LLM)
docker compose up -d neo4j mongodb postgres ollama vllm

# Run backend
python -m uvicorn app:app --reload --port 8000

# Run pipeline service
python -m uvicorn pipeline:app --reload --port 8003

# Run frontend
cd frontend && npm run dev
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LightRAG](https://github.com/HKUDS/LightRAG) - The knowledge graph RAG framework
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) - Neural retrieval model
- [Ollama](https://ollama.ai/) - Local LLM serving

---

## 📬 Support

For issues and feature requests, please open a GitHub issue.
