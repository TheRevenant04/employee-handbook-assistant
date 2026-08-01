-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;


-- Handbook documents table (for RAG retrieval)
CREATE TABLE IF NOT EXISTS handbook_documents (
    id BIGSERIAL PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(content, ''))
    ) STORED
);

CREATE INDEX IF NOT EXISTS handbook_documents_embedding_hnsw_idx
    ON handbook_documents
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS handbook_documents_content_tsv_idx
    ON handbook_documents
    USING GIN (content_tsv);


-- Chat conversations and messages
CREATE TABLE IF NOT EXISTS conversations (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New conversation',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT,
    rating SMALLINT CHECK (rating IS NULL OR rating IN (-1, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, created_at);

CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations (updated_at DESC);


-- Message performance metrics
CREATE TABLE IF NOT EXISTS message_metrics (
    id BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    total_latency_ms FLOAT NOT NULL,
    retrieval_latency_ms FLOAT,
    llm_latency_ms FLOAT,
    num_results INT,
    avg_distance FLOAT,
    min_distance FLOAT,
    model TEXT,
    success BOOLEAN NOT NULL DEFAULT true,
    input_tokens INT,
    output_tokens INT,
    cost NUMERIC(12, 6) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_metrics_message
    ON message_metrics (message_id);

CREATE INDEX IF NOT EXISTS idx_metrics_created
    ON message_metrics (created_at DESC);


-- Evaluation runs and results
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id BIGSERIAL PRIMARY KEY,
    judge_model TEXT NOT NULL,
    evaluated_model TEXT NOT NULL,
    num_questions INT NOT NULL CHECK (num_questions >= 0),
    config JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    message_id BIGINT NOT NULL REFERENCES messages(id),
    expected_document TEXT,
    retrieved_context TEXT,
    faithfulness_score SMALLINT NOT NULL CHECK (faithfulness_score BETWEEN 1 AND 5),
    faithfulness_reasoning TEXT,
    context_relevance_score SMALLINT NOT NULL CHECK (context_relevance_score BETWEEN 1 AND 5),
    context_relevance_reasoning TEXT,
    completeness_score SMALLINT NOT NULL CHECK (completeness_score BETWEEN 1 AND 5),
    completeness_reasoning TEXT,
    judge_input_tokens INT,
    judge_output_tokens INT,
    judge_cost NUMERIC(12, 6) NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_eval_results_run
    ON evaluation_results (run_id);

CREATE INDEX IF NOT EXISTS idx_eval_results_msg
    ON evaluation_results (message_id);

CREATE INDEX IF NOT EXISTS idx_eval_runs_created
    ON evaluation_runs (created_at DESC);


-- Monitoring: error log

CREATE TABLE IF NOT EXISTS error_log (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_error_log_created
    ON error_log (created_at DESC);


-- Trigger to keep conversations.updated_at fresh
DROP TRIGGER IF EXISTS trg_messages_update_conversation ON messages;
DROP FUNCTION IF EXISTS update_conversation_timestamp();

CREATE OR REPLACE FUNCTION update_conversation_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE conversations
    SET updated_at = now()
    WHERE id = NEW.conversation_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_messages_update_conversation
AFTER INSERT ON messages
FOR EACH ROW
EXECUTE FUNCTION update_conversation_timestamp();