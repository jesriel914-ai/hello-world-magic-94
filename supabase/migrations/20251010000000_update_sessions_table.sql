-- Migration: Update sessions table
-- 1. Add status column
-- 2. Remove description and capacity columns

-- Add status column with enum constraint
ALTER TABLE public.sessions 
ADD COLUMN status text NOT NULL DEFAULT 'upcoming';

-- Add check constraint for status values
ALTER TABLE public.sessions 
ADD CONSTRAINT sessions_status_check 
CHECK (status = ANY (ARRAY['upcoming'::text, 'ongoing'::text, 'completed'::text]));

-- Drop description column
ALTER TABLE public.sessions 
DROP COLUMN IF EXISTS description;

-- Drop capacity column
ALTER TABLE public.sessions 
DROP COLUMN IF EXISTS capacity;

-- Add comment to status column
COMMENT ON COLUMN public.sessions.status IS 'Session status: upcoming, ongoing, or completed';
