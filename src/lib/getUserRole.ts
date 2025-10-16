import { supabase } from "@/lib/supabase";

// Normalized application roles
export type AppRole = 'admin' | 'attendance checker' | 'user';

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

  // Check attendance_checker table
  const { data: checkerRec, error: checkerErr } = await supabase
    .from('attendance_checker')
    .select('id')
    .eq('id', userId)
    .maybeSingle();
  if (checkerErr) {
    console.error('fetchUserRole attendance_checker error:', checkerErr);
  }
  if (checkerRec) return 'attendance checker';

  return 'user';
};
