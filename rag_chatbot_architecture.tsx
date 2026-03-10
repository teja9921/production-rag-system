import { useState } from 'react';
import { ChevronRight, Database, FileText, Layers, Zap, Server, MessageSquare } from 'lucide-react';

export default function RAGArchitecture() {
  const [selectedLayer, setSelectedLayer] = useState(null);

  const layers = [
    {
      id: 'ingestion',
      title: 'Data Ingestion Layer',
      color: 'bg-green-500',
      icon: FileText,
      files: [
        { name: 'loader.py', desc: 'Loads PDF files, extracts text per page, generates SHA-256 doc_id' },
        { name: 'splitter.py', desc: 'Chunks pages into fixed-size segments with overlap (500 tokens, 50 overlap)' }
      ],
      connections: ['rag']
    },
    {
      id: 'rag',
      title: 'RAG Core Layer',
      color: 'bg-yellow-500',
      icon: Layers,
      files: [
        { name: 'embedder.py', desc: 'Singleton service using sentence-transformers/all-MiniLM-L6-v2 for vectorization' },
        { name: 'faiss_store.py', desc: 'FAISS IndexFlatIP vector store with normalized embeddings' },
        { name: 'retriever.py', desc: 'Performs similarity search with threshold gating (0.45 default)' },
        { name: 'schemas.py', desc: 'Data models and type definitions' }
      ],
      connections: ['orchestration']
    },
    {
      id: 'orchestration',
      title: 'Orchestration Layer (LangGraph)',
      color: 'bg-purple-500',
      icon: Zap,
      files: [
        { name: 'lc_retriever.py', desc: 'LangChain Runnable wrapper for retriever with query rewriting' },
        { name: 'lc_llm.py', desc: 'LLM Runnable using HuggingFace Inference (Meta-Llama-3-8B-Instruct)' },
        { name: 'graph.py', desc: 'LangGraph state machine: retrieve → route → llm (or NO_ANSWER)' },
        { name: 'agent_graph.py', desc: 'Advanced agentic graph with memory and query rewriting' },
        { name: 'state.py', desc: 'GraphState schema for stateful conversations' },
        { name: 'memory.py', desc: 'Conversation memory management' },
        { name: 'rewrite.py', desc: 'Query rewriting for better retrieval' }
      ],
      connections: ['api']
    },
    {
      id: 'api',
      title: 'API Layer (FastAPI)',
      color: 'bg-red-500',
      icon: Server,
      files: [
        { name: 'main.py', desc: 'FastAPI endpoints: /users, /conversations, /query, /messages' },
        { name: 'deps.py', desc: 'Dependency injection - initializes entire RAG pipeline (GRAPH singleton)' },
        { name: 'config.py', desc: 'Settings using pydantic-settings (HF_TOKEN, thresholds, models)' },
        { name: 'schemas.py', desc: 'Request/response models for API validation' },
        { name: 'agent_deps.py', desc: 'Agent-specific dependencies' }
      ],
      connections: ['db']
    },
    {
      id: 'db',
      title: 'Database Layer (SQLite)',
      color: 'bg-orange-500',
      icon: Database,
      files: [
        { name: 'models.py', desc: 'SQLAlchemy models: User, Conversation, Message with UUIDs' },
        { name: 'crud.py', desc: 'Database operations: create_user, add_message, get_messages' },
        { name: 'session.py', desc: 'Database session management' },
        { name: 'base.py', desc: 'SQLAlchemy Base class' }
      ],
      connections: []
    },
    {
      id: 'ui',
      title: 'UI Layer',
      color: 'bg-blue-500',
      icon: MessageSquare,
      files: [
        { name: 'streamlit_app/app.py', desc: 'Streamlit chat interface for end users' }
      ],
      connections: ['api']
    }
  ];

  const dataFlow = [
    'PDF Files → loader.py → Pages with metadata',
    'Pages → splitter.py → Chunks with IDs',
    'Chunks → embedder.py → Vector embeddings',
    'Embeddings → faiss_store.py → Indexed vectors',
    'Query → retriever.py → Top-K chunks (if score > threshold)',
    'Chunks → lc_retriever.py → LangChain format',
    'Context → lc_llm.py → Generated answer',
    'Answer → main.py → HTTP response',
    'Conversation → models.py → Database persistence'
  ];

  const keyFiles = [
    { name: 'run_graph.py', desc: 'Test script demonstrating end-to-end flow', color: 'bg-cyan-500' },
    { name: 'create_db.py', desc: 'Database initialization script', color: 'bg-cyan-500' },
    { name: 'requirements.txt', desc: 'Python dependencies', color: 'bg-gray-500' },
    { name: '.env', desc: 'Environment variables (HF_TOKEN)', color: 'bg-gray-500' }
  ];

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-2 text-gray-800">RAG Medical Chatbot Architecture</h1>
        <p className="text-gray-600 mb-6">Click on any layer to see detailed file information</p>

        {/* Architecture Layers */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {layers.map((layer) => {
            const Icon = layer.icon;
            return (
              <div
                key={layer.id}
                onClick={() => setSelectedLayer(selectedLayer === layer.id ? null : layer.id)}
                className={`${layer.color} bg-opacity-10 border-2 ${layer.color.replace('bg-', 'border-')} rounded-lg p-4 cursor-pointer hover:shadow-lg transition-all ${
                  selectedLayer === layer.id ? 'ring-4 ring-offset-2' : ''
                }`}
              >
                <div className="flex items-center mb-2">
                  <Icon className={`mr-2 ${layer.color.replace('bg-', 'text-')}`} size={24} />
                  <h3 className="font-bold text-lg">{layer.title}</h3>
                </div>
                <p className="text-sm text-gray-600 mb-2">{layer.files.length} files</p>
                {selectedLayer === layer.id && (
                  <div className="mt-4 space-y-2">
                    {layer.files.map((file, idx) => (
                      <div key={idx} className="bg-white p-3 rounded shadow-sm">
                        <p className="font-mono text-sm font-semibold text-gray-800">{file.name}</p>
                        <p className="text-xs text-gray-600 mt-1">{file.desc}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Data Flow */}
        <div className="bg-white rounded-lg p-6 shadow-md mb-8">
          <h2 className="text-2xl font-bold mb-4 flex items-center">
            <ChevronRight className="mr-2 text-blue-500" />
            Data Flow Pipeline
          </h2>
          <div className="space-y-3">
            {dataFlow.map((flow, idx) => (
              <div key={idx} className="flex items-start">
                <span className="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold mr-3 flex-shrink-0 mt-0.5">
                  {idx + 1}
                </span>
                <p className="text-gray-700">{flow}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Key Design Principles */}
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-6 shadow-md mb-8">
          <h2 className="text-2xl font-bold mb-4 text-purple-800">Key Design Principles</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white p-4 rounded-lg shadow">
              <h3 className="font-bold text-purple-700 mb-2">Safety First</h3>
              <p className="text-sm text-gray-700">Threshold-based gating (0.45 default). Returns NO_ANSWER when confidence is low. Prevents hallucinations.</p>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <h3 className="font-bold text-purple-700 mb-2">Deterministic Behavior</h3>
              <p className="text-sm text-gray-700">Same input always produces same chunks, embeddings, and retrieval results. Chunk ID: doc_id_p&#123;page&#125;_c&#123;index&#125;</p>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <h3 className="font-bold text-purple-700 mb-2">Framework Isolation</h3>
              <p className="text-sm text-gray-700">LangChain/LangGraph used for orchestration only. Core logic (chunking, retrieval, safety) remains custom.</p>
            </div>
            <div className="bg-white p-4 rounded-lg shadow">
              <h3 className="font-bold text-purple-700 mb-2">Stateful Conversations</h3>
              <p className="text-sm text-gray-700">SQLite persistence with Users, Conversations, Messages. Enables multi-turn chat and auditability.</p>
            </div>
          </div>
        </div>

        {/* Supporting Files */}
        <div className="bg-white rounded-lg p-6 shadow-md">
          <h2 className="text-2xl font-bold mb-4">Supporting Files</h2>
          <div className="grid md:grid-cols-2 gap-3">
            {keyFiles.map((file, idx) => (
              <div key={idx} className={`${file.color} bg-opacity-10 border-2 ${file.color.replace('bg-', 'border-')} rounded p-3`}>
                <p className="font-mono font-semibold text-sm">{file.name}</p>
                <p className="text-xs text-gray-600 mt-1">{file.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Connection Map */}
        <div className="bg-white rounded-lg p-6 shadow-md mt-8">
          <h2 className="text-2xl font-bold mb-4">Module Dependencies</h2>
          <div className="space-y-2 text-sm font-mono">
            <p><span className="text-green-600">ingestion/loader.py</span> → Used by deps.py</p>
            <p><span className="text-green-600">ingestion/splitter.py</span> → Uses loader output, used by deps.py</p>
            <p><span className="text-yellow-600">rag/embedder.py</span> → Used by faiss_store, retriever, deps.py</p>
            <p><span className="text-yellow-600">rag/faiss_store.py</span> → Uses embedder, used by retriever</p>
            <p><span className="text-yellow-600">rag/retriever.py</span> → Uses embedder + faiss_store, used by lc_retriever</p>
            <p><span className="text-purple-600">orchestration/lc_retriever.py</span> → Wraps retriever.py</p>
            <p><span className="text-purple-600">orchestration/lc_llm.py</span> → Uses HF Inference API</p>
            <p><span className="text-purple-600">orchestration/graph.py</span> → Composes lc_retriever + lc_llm</p>
            <p><span className="text-red-600">api/deps.py</span> → Initializes entire pipeline → GRAPH</p>
            <p><span className="text-red-600">api/main.py</span> → Uses GRAPH from deps.py, crud from db/</p>
            <p><span className="text-orange-600">db/models.py</span> → Defines schema</p>
            <p><span className="text-orange-600">db/crud.py</span> → Uses models.py for operations</p>
          </div>
        </div>
      </div>
    </div>
  );
}