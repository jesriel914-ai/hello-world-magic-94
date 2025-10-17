import { supabase } from '@/integrations/supabase/client';
import type { 
  Session, 
  Student, 
  AttendanceRecord, 
  SessionWithStudents, 
  User
} from '@/types';

// Session operations
export const fetchSessions = async (startDate?: string, endDate?: string): Promise<Session[]> => {
  try {
    console.log('Fetching sessions with date range:', { startDate, endDate });
    
    let query = supabase
      .from('sessions')
      .select(`
        *,
        creator:created_by_user_id(
          first_name,
          last_name,
          role
        )
      `)
      .order('date', { ascending: true });

    // Ensure dates are in YYYY-MM-DD format for comparison
    if (startDate) {
      const start = new Date(startDate);
      const formattedStart = start.toISOString().split('T')[0];
      query = query.gte('date', formattedStart);
    }
    
    if (endDate) {
      const end = new Date(endDate);
      const formattedEnd = end.toISOString().split('T')[0];
      query = query.lte('date', formattedEnd);
    }

    const { data, error } = await query;
    
    if (error) throw error;
    
    console.log('Fetched sessions:', data);
    return (data as any) || [];
  } catch (error) {
    console.error('Error fetching sessions:', error);
    throw error;
  }
};

export const createSession = async (sessionData: Omit<Session, 'id' | 'created_at' | 'updated_at' | 'created_by_user_id'>): Promise<Session> => {
  // Get current user
  const { data: { user }, error: userError } = await supabase.auth.getUser();
  if (userError || !user) {
    throw new Error('User must be authenticated to create sessions');
  }

  // Add the creator's user ID automatically
  const sessionWithCreator = {
    ...sessionData,
    created_by_user_id: user.id
  };

  const { data, error } = await supabase
    .from('sessions')
    .insert([sessionWithCreator] as any)
    .select()
    .single();
  
  if (error) throw error;
  return data as Session;
};

export const updateSession = async (id: number, sessionData: Partial<Session>): Promise<Session> => {
  const { data, error } = await supabase
    .from('sessions')
    .update(sessionData as any)
    .eq('id', id)
    .select()
    .single();
  
  if (error) throw error;
  return data as Session;
};

export const deleteSession = async (id: number): Promise<void> => {
  const { error } = await supabase
    .from('sessions')
    .delete()
    .eq('id', id);
  
  if (error) throw error;
};

// Student operations
export const fetchStudents = async (program?: string, year?: string): Promise<Student[]> => {
  try {
    
    // Helper function to fetch all students with pagination
    const fetchAllStudents = async (baseQuery: any): Promise<Student[]> => {
      const allStudents: Student[] = [];
      let from = 0;
      const pageSize = 1000; // Supabase default limit
      
      while (true) {
        const { data, error } = await baseQuery.range(from, from + pageSize - 1);
        if (error) throw error;
        
        if (!data || data.length === 0) break;
        
        allStudents.push(...data);
        
        // If we got less than pageSize, we've reached the end
        if (data.length < pageSize) break;
        
        from += pageSize;
      }
      
      return allStudents;
    };
    
    // Build the base query
    let baseQuery = supabase
      .from('students')
      .select('*')
      .order('surname', { ascending: true });

    // Apply program filter if it's a specific program (not 'all' or empty string)
    if (program && program !== 'All Programs') {
      baseQuery = baseQuery.eq('program', program.trim());
    }

    // Apply year filter if it's a specific year (not 'all')
    if (year && year !== 'All Years') {
      // Use the correct field name 'year' instead of 'year_level'
      baseQuery = baseQuery.eq('year', year.trim());
    }

    // Fetch all matching students with pagination
    const allStudents = await fetchAllStudents(baseQuery);
    
    return allStudents?.map(s => ({
      ...s,
      full_name: `${s.firstname} ${s.surname}`
    })) || [];
  } catch (error) {
    console.error('Error fetching students:', error);
    throw error;
  }
};

export const createStudent = async (studentData: Omit<Student, 'id' | 'created_at'>): Promise<Student> => {
  const { data, error } = await supabase
    .from('students')
    .insert([studentData] as any)
    .select()
    .single();
  
  if (error) throw error;
  return data as Student;
};

export const updateStudent = async (id: number, studentData: Partial<Student>): Promise<Student> => {
  const { data, error } = await supabase
    .from('students')
    .update(studentData as any)
    .eq('id', id)
    .select()
    .single();
  
  if (error) throw error;
  return data as Student;
};

export const deleteStudent = async (id: number): Promise<void> => {
  const { error } = await supabase
    .from('students')
    .delete()
    .eq('id', id);
  
  if (error) throw error;
};

// Attendance operations
export const fetchSessionStudents = async (sessionId: number): Promise<SessionWithStudents> => {
  if (!sessionId) {
    throw new Error('Session ID is required');
  }
  
  // Get session details with status
  const { data: session, error: sessionError } = await supabase
    .from('sessions')
    .select('*, status')
    .eq('id', sessionId)
    .single();
  
  if (sessionError) throw sessionError;
  if (!session) throw new Error('Session not found');

  const sessionData = session as any;
  
  console.log('Fetched session status:', sessionData.status);

  console.log('Full session object:', JSON.stringify(sessionData, null, 2));

  // Helper function to check if a value represents 'all' (case insensitive and includes 'all')
  const isAllValue = (value?: string) => {
    if (!value) return true;
    const lowerValue = value.toLowerCase().trim();
    return lowerValue.includes('all') || lowerValue === '' || lowerValue === 'all programs' || lowerValue === 'all year levels';
  };

  // Helper function to fetch all students with pagination
  const fetchAllStudents = async (baseQuery: any): Promise<Student[]> => {
    const allStudents: Student[] = [];
    let from = 0;
    const pageSize = 1000; // Supabase default limit
    
    while (true) {
      const { data, error } = await baseQuery.range(from, from + pageSize - 1);
      if (error) throw error;
      
      if (!data || data.length === 0) break;
      
      allStudents.push(...data);
      
      // If we got less than pageSize, we've reached the end
      if (data.length < pageSize) break;
      
      from += pageSize;
    }
    
    return allStudents;
  };


  let allStudents: Student[] = [];
  
  // Build the base query for fetching students
  try {
    
    // Build the base query
    let baseQuery = supabase
      .from('students')
      .select('*')
      .order('id');
      
      
    // Apply program filter if specified and not "all"
    if (sessionData.program && !isAllValue(sessionData.program)) {
      const programValue = sessionData.program.trim();
      baseQuery = baseQuery.eq('program', programValue);
    }
    
    // Apply year filter if specified and not "all"
    if (sessionData.year && !isAllValue(sessionData.year)) {
      let yearValue = sessionData.year.trim();
      // Convert year format if needed (e.g., '1st Year' to '1st')
      if (yearValue.endsWith(' Year')) {
        yearValue = yearValue.replace(' Year', '');
      }
      baseQuery = baseQuery.eq('year', yearValue);
    }
    
    
    // Execute the query with pagination
    allStudents = await fetchAllStudents(baseQuery);
    
  } catch (error) {
    console.error('Error fetching students:', error);
    throw error;
  }

  const students = allStudents || [];
  
  if (!sessionData) {
    throw new Error('Session not found');
  }

  // Get attendance records for this session
  const { data: attendance, error: attendanceError } = await supabase
    .from('attendance')
    .select('*')
    .eq('session_id', sessionId);
  
  if (attendanceError) throw attendanceError;

  // Merge student data with attendance
  const studentsWithAttendance = students.map(student => {
    const attendanceRecord = (attendance as any)?.find((a: any) => a.student_id === student.id) || null;
    return {
      ...student,
      full_name: `${student.firstname} ${student.surname}`,
      status: attendanceRecord?.status || null,
      time_in: attendanceRecord?.time_in || null,
      time_out: attendanceRecord?.time_out || null,
    };
  });

  const sessionInfo = {
    id: sessionData.id,
    title: sessionData.title,
    date: sessionData.date,
    time_in: sessionData.time_in || null,
    time_out: sessionData.time_out || null,
    program: sessionData.program,
    year: sessionData.year,
    description: sessionData.description || '',
    type: sessionData.type || 'class',
    status: sessionData.status || 'not completed' // Include status field
  };

  return {
    session: sessionInfo,
    students: studentsWithAttendance,
    count: studentsWithAttendance.length,
  };
};

export const markAttendance = async (
  sessionId: number,
  studentId: number,
  status: 'present' | 'absent' | 'late' | 'excused',
  timeIn?: string,
  timeOut?: string
): Promise<AttendanceRecord> => {
  const { data, error } = await supabase
    .from('attendance')
    .upsert(
      {
        session_id: sessionId,
        student_id: studentId,
        status,
        time_in: timeIn || null,
        time_out: timeOut || null,
        updated_at: new Date().toISOString(),
      } as any,
      { onConflict: 'session_id,student_id' }
    )
    .select();

  if (error) {
    console.error('Error fetching attendance records:', error);
    throw new Error(error.message);
  }

  if (!data || data.length === 0) {
    throw new Error('No attendance record was created or updated');
  }
  return data[0] as unknown as AttendanceRecord;
};

// User operations
export const getCurrentUser = async (): Promise<User | null> => {
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;
  
  // Get additional user data from admin/attendance_checker tables
  const { data: adminData } = await supabase
    .from('admin')
    .select('*')
    .eq('id', user.id)
    .maybeSingle();
  const { data: checkerData } = adminData ? { data: null as any } : await supabase
    .from('attendance_checker')
    .select('*')
    .eq('id', user.id)
    .maybeSingle();
  const profile: any = adminData || checkerData || {};
  
  return {
    id: user.id,
    email: user.email || '',
    name: `${profile?.first_name || ''} ${profile?.last_name || ''}`.trim() || user.user_metadata?.full_name || '',
    role: (profile as any)?.role || (adminData ? 'admin' : 'attendance checker'),
    avatar_url: (profile as any)?.avatar_url || '',
  };
};

export const updateUserProfile = async (updates: Partial<User>): Promise<User> => {
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) throw new Error('Not authenticated');
  
  // Update auth user if email is being changed
  if (updates.email) {
    const { error } = await supabase.auth.updateUser({
      email: updates.email,
      data: { full_name: updates.name },
    });
    if (error) throw error;
  }
  
  // Update account in the correct table
  const isAdmin = !!(await supabase.from('admin').select('id').eq('id', user.id).maybeSingle()).data;
  const target = isAdmin ? 'admin' : 'attendance_checker';
  const payload: any = {
    updated_at: new Date().toISOString(),
  };
  if (updates.name) {
    const [first, ...rest] = updates.name.split(' ');
    payload.first_name = first;
    payload.last_name = rest.join(' ');
  }
  if (updates.role && !isAdmin) {
    payload.role = updates.role;
  }
  const { data, error } = await supabase
    .from(target)
    .update(payload)
    .eq('id', user.id)
    .select()
    .maybeSingle();
  
  if (error) throw error;
  
  return {
    id: user.id,
    email: updates.email || user.email || '',
    name: (data as any).full_name,
    role: (data as any).role,
    avatar_url: (data as any).avatar_url || '',
  };
};