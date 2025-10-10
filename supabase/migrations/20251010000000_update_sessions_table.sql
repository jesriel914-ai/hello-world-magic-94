-- Migration: Update sessions table
-- 1. Add status column
-- 2. Remove description and capacity columns

-- Add status column with enum constraint
-- Default to 'not completed' for all new sessions
ALTER TABLE public.sessions 
ADD COLUMN status text NOT NULL DEFAULT 'not completed';

-- Add check constraint for status values
-- Only two states: 'not completed' (default) and 'completed' (user-initiated)
ALTER TABLE public.sessions 
ADD CONSTRAINT sessions_status_check 
CHECK (status = ANY (ARRAY['not completed'::text, 'completed'::text]));

-- Drop description column
ALTER TABLE public.sessions 
DROP COLUMN IF EXISTS description;

-- Drop capacity column
ALTER TABLE public.sessions 
DROP COLUMN IF EXISTS capacity;

-- Add comment to status column
COMMENT ON COLUMN public.sessions.status IS 'Session completion status: not completed (default) or completed (user-initiated)';
