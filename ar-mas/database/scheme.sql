CREATE TABLE IF NOT EXISTS candidates (
    id SERIAL PRIMARY KEY,
    name TEXT,
    email TEXT,
    skills JSONB,
    experience JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS jobs (
    id SERIAL PRIMARY KEY,
    title TEXT,
    description TEXT,
    embeddings VECTOR(768),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS interviews (
    id SERIAL PRIMARY KEY,
    candidate_id INT REFERENCES candidates(id),
    transcript TEXT,
    score FLOAT,
    summary JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
