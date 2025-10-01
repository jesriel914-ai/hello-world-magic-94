// Local type definitions to work around Supabase type generation issues
// This is a temporary fix while Supabase types are being regenerated

export interface LocalSession {
  id: number;
  title: string;
  type: 'class' | 'event' | 'other';
  time_in?: string;
  time_out?: string;
  created_by_user_id?: string;
  program: string;
  year: string;
  section: string;
  description?: string;
  capacity: string | number;
  date: string;
  created_at: string;
  updated_at: string;
}

export interface LocalStudent {
  id: number;
  student_id: string;
  firstname: string;
  surname: string;
  middlename?: string;
  middle_initial?: string;
  program: string;
  year: string;
  section: string;
  email?: string;
  contact_number?: string;
  address?: string;
  created_at?: string;
  updated_at?: string;
}

export interface LocalAttendanceRecord {
  id: number;
  session_id: number;
  student_id: number;
  time_in?: string;
  time_out?: string;
  status: 'present' | 'absent' | 'late' | 'excused';
  created_at?: string;
  updated_at?: string;
}
