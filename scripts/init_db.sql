CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS agent_memory (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(64) NOT NULL,
    key VARCHAR(256) NOT NULL,
    value JSONB NOT NULL,
    embedding vector(256),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(agent_id, key)
);

CREATE TABLE IF NOT EXISTS request_log (
    id SERIAL PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    provider VARCHAR(64) NOT NULL,
    model VARCHAR(128) NOT NULL,
    prompt_tokens INT NOT NULL DEFAULT 0,
    completion_tokens INT NOT NULL DEFAULT 0,
    cost_usd NUMERIC(10, 6) NOT NULL DEFAULT 0,
    latency_ms INT NOT NULL DEFAULT 0,
    status VARCHAR(32) NOT NULL,
    task_type VARCHAR(64),
    routed_by VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS provider_health (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(64) NOT NULL,
    model VARCHAR(128) NOT NULL,
    is_healthy BOOLEAN NOT NULL DEFAULT TRUE,
    avg_latency_ms INT,
    error_rate NUMERIC(5, 4),
    last_checked_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(provider, model)
);

CREATE TABLE IF NOT EXISTS prompt_cache (
    id SERIAL PRIMARY KEY,
    prompt_hash VARCHAR(64) NOT NULL UNIQUE,
    embedding vector(256),
    response JSONB NOT NULL,
    model VARCHAR(128) NOT NULL,
    hit_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX idx_request_log_created ON request_log(created_at DESC);
CREATE INDEX idx_request_log_provider ON request_log(provider, created_at DESC);
CREATE INDEX idx_prompt_cache_embedding ON prompt_cache USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_agent_memory_embedding ON agent_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
