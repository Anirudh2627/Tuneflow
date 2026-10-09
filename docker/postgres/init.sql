-- TuneFlow database bootstrap (extensions & tuning for local dev)
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- faster ILIKE search on catalog text
