import { supabase } from "@/lib/supabase";

// Normalized application roles
export type AppRole = 'admin' | 'Instructor' | 'SSG officer' | 'ROTC admin' | 'ROTC officer' | 'user';

// Map legacy/variant roles to normalized ones
const normalizeRole = (raw: string | null | undefined): AppRole => {
  const val = (raw || '').trim();
  if (!val) return 'user';
  if (val.toLowerCase() === 'admin') return 'admin';
  if (val.toLowerCase() === 'staff' || val === 'Instructor') return 'Instructor';
  if (val.toLowerCase() === 'ssg officer' || val.toLowerCase() === 'ssg_officer') return 'SSG officer';
  if (val === 'ROTC admin') return 'ROTC admin';
  if (val === 'ROTC officer') return 'ROTC officer';
  return 'user';
};

export const fetchUserRole = async (userId: string | null): Promise<AppRole> => {
  if (!userId) return 'user';

  // Check admin table first
  const { data: adminRec, error: adminErr } = await supabase
    .from('admin')
    .select('id')
    .eq('id', userId)
    .maybeSingle();
  if (adminErr) {
    console.error('fetchUserRole admin error:', adminErr);
  }
  if (adminRec) return 'admin';

  // Check users table
  const { data: userRec, error: userErr } = await supabase
    .from('users')
    .select('role')
    .eq('id', userId)
    .maybeSingle();
  if (userErr) {
    console.error('fetchUserRole user error:', userErr);
  }
  if (userRec?.role) {
    return normalizeRole(userRec.role);
  }

  return 'user';
};
