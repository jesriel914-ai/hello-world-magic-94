-- Migration: Update sessions table
-- 1. Add status column (if not exists)
-- 2. Remove description and capacity columns
-- 3. Update existing status values to new format

-- Add status column if it doesn't exist
DO $$ 
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = 'sessions' 
    AND column_name = 'status'
  ) THEN
    ALTER TABLE public.sessions 
    ADD COLUMN status text NOT NULL DEFAULT 'not completed';
  END IF;
END $$;

-- Drop old constraint if it exists
ALTER TABLE public.sessions 
DROP CONSTRAINT IF EXISTS sessions_status_check;

-- Update existing status values to new format
UPDATE public.sessions 
SET status = 'not completed' 
WHERE status IN ('upcoming', 'ongoing');

UPDATE public.sessions 
SET status = 'completed' 
WHERE status = 'completed';

-- Add new check constraint for status values
-- Only two states: 'not completed' (default) and 'completed' (user-initiated)
ALTER TABLE public.sessions 
ADD CONSTRAINT sessions_status_check 
CHECK (status = ANY (ARRAY['not completed'::text, 'completed'::text]));

-- Set default value for status column
ALTER TABLE public.sessions 
ALTER COLUMN status SET DEFAULT 'not completed';

-- Drop description column
ALTER TABLE public.sessions 
DROP COLUMN IF EXISTS description;

-- Drop capacity column
ALTER TABLE public.sessions 
DROP COLUMN IF EXISTS capacity;

-- Add comment to status column
COMMENT ON COLUMN public.sessions.status IS 'Session completion status: not completed (default) or completed (user-initiated)';
