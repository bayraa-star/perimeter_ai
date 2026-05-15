CREATE TABLE IF NOT EXISTS container_profiles (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    image TEXT NOT NULL,
    container_name TEXT NOT NULL UNIQUE,
    env_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    port_bindings_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    volume_bindings_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    network_mode TEXT NOT NULL DEFAULT 'bridge',
    restart_policy_name TEXT NOT NULL DEFAULT 'unless-stopped',
    restart_policy_max_retry_count INTEGER NOT NULL DEFAULT 0,
    gpu_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    gpu_vendor TEXT NOT NULL DEFAULT 'none',
    gpu_count INTEGER NOT NULL DEFAULT 0,
    docker_image_tag TEXT,
    docker_container_id TEXT,
    runtime_status TEXT NOT NULL DEFAULT 'draft',
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT container_profiles_gpu_vendor_check
        CHECK (gpu_vendor IN ('none', 'intel', 'nvidia')),
    CONSTRAINT container_profiles_runtime_status_check
        CHECK (runtime_status IN ('draft', 'built', 'running', 'stopped', 'error'))
);

CREATE INDEX IF NOT EXISTS idx_container_profiles_status
    ON container_profiles (runtime_status);

CREATE INDEX IF NOT EXISTS idx_container_profiles_gpu_vendor
    ON container_profiles (gpu_vendor);

CREATE INDEX IF NOT EXISTS idx_container_profiles_env_json
    ON container_profiles
    USING GIN (env_json);
