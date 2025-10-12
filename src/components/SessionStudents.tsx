import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, Users } from "lucide-react";
import { fetchSessionStudents } from "@/lib/supabaseService";
import { supabase } from "@/lib/supabase";
import type { Student } from "@/types";
import { clearAllSessionCaches, clearTakeAttendanceCache } from "@/lib/cacheManager";

interface SessionResponse {
  id: number;
  title: string;
  program: string;
  year: string;
  section: string;
  date: string;
  time: string;
  description: string;
  type: 'class' | 'event' | 'other';
}

interface Session {
  id: number;
  title: string;
  program: string;
  year: string;
  section: string;
  date: string;
  time_in: string;
  time_out: string;
  description: string;
  type: 'class' | 'event' | 'other';
}

interface SessionStudentsFormProps {
  sessionId: number;
  onClose?: () => void;
  onSessionUpdated?: () => void;
}

// Cache for session data
const sessionCache = new Map<number, { session: Session; students: Student[]; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export default function SessionStudents({ sessionId, onClose, onSessionUpdated }: SessionStudentsFormProps) {
  const [session, setSession] = useState<Session | null>(null);
  const [allStudents, setAllStudents] = useState<Student[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [attendanceRecords, setAttendanceRecords] = useState<Map<number, any>>(new Map());
  const [displayLimit, setDisplayLimit] = useState(50);
  const [isMarkingCompleted, setIsMarkingCompleted] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const hasFetched = useRef(false);

  // Fetch session and student data - always fetch fresh to get latest status
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Always fetch fresh data from database to get latest status
        const response = await fetchSessionStudents(sessionId);
        
        if (!response?.session) {
          return;
        }
        
        const sessionData = response.session;
        const sessionStudents = response.students || [];
        
        console.log('SessionStudents: Received session data:', { 
          id: sessionData.id, 
          status: sessionData.status,
          isCompleted: sessionData.status === 'completed'
        });
        setIsCompleted(sessionData.status === 'completed');
        
        // Transform session data
        const session: Session = {
          id: sessionData.id,
          title: sessionData.title,
          program: sessionData.program,
          year: sessionData.year,
          section: sessionData.section,
          date: sessionData.date,
          time_in: sessionData.time_in || '',
          time_out: sessionData.time_out || '',
          description: sessionData.description || '',
          type: sessionData.type || 'class',
          status: sessionData.status || 'not completed'
        };
        
        setSession(session);
        
        // Fetch attendance records
        const { data: attendanceData } = await supabase
          .from('attendance')
          .select('*')
          .eq('session_id', sessionData.id);
        
        const attendanceMap = new Map();
        (attendanceData || []).forEach((record: any) => {
          attendanceMap.set(record.student_id, record);
        });
        setAttendanceRecords(attendanceMap);
        
        // Transform students data
        const studentsList = sessionStudents
          .map(student => ({
            ...student,
            full_name: student.full_name || `${student.firstname || ''} ${student.surname || ''}`.trim(),
            attendance: attendanceMap.get(student.id) || null
          }))
          .sort((a, b) => a.full_name.localeCompare(b.full_name));
        
        setAllStudents(studentsList);
        
        // Cache the data
        sessionCache.set(sessionId, {
          session,
          students: studentsList,
          timestamp: Date.now()
        });
        
      } catch (err: unknown) {
        console.error('Error fetching session students:', err);
      } finally {
        setLoading(false);
      }
    };
    
    // Reset hasFetched when sessionId changes to always fetch fresh data
    hasFetched.current = false;
    fetchData();
  }, [sessionId]);

  // Format time to AM/PM format
  const formatTime = (timeString: string) => {
    if (!timeString) return '';
    const [hours, minutes] = timeString.split(':');
    const hour = parseInt(hours, 10);
    const ampm = hour >= 12 ? 'PM' : 'AM';
    const hour12 = hour % 12 || 12;
    return `${hour12}:${minutes} ${ampm}`;
  };

  const filteredStudents = allStudents
    .filter(student =>
      student.full_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      student.student_id.toLowerCase().includes(searchQuery.toLowerCase())
    )
    .sort((a, b) => {
      // Sort by most recent attendance record first
      const aRecord = attendanceRecords.get(a.id);
      const bRecord = attendanceRecords.get(b.id);
      
      // Students with attendance records go first
      if (aRecord && !bRecord) return -1;
      if (!aRecord && bRecord) return 1;
      
      // Both have records, sort by most recent update
      if (aRecord && bRecord) {
        const aTime = new Date(aRecord.updated_at || aRecord.created_at || 0).getTime();
        const bTime = new Date(bRecord.updated_at || bRecord.created_at || 0).getTime();
        return bTime - aTime; // Most recent first
      }
      
      // Neither has records, maintain original order
      return 0;
    });

  // Handle marking session as completed
  const handleMarkCompleted = async () => {
    if (!session || isMarkingCompleted || isCompleted) return;

    try {
      setIsMarkingCompleted(true);

      // Get all students with "Not recorded" status
      const studentsToMarkAbsent = allStudents.filter(student => {
        const record = attendanceRecords.get(student.id);
        return !record || record.status === 'Not recorded';
      });

      // Mark all "Not recorded" students as absent
      if (studentsToMarkAbsent.length > 0) {
        const absentRecords = studentsToMarkAbsent.map(student => ({
          session_id: sessionId,
          student_id: student.id,
          status: 'absent'
        }));

        const { error: attendanceError } = await supabase
          .from('attendance')
          .upsert(absentRecords, {
            onConflict: 'session_id,student_id'
          });

        if (attendanceError) throw attendanceError;
      }

      // Update session status to completed
      const { error: sessionError } = await supabase
        .from('sessions')
        .update({ status: 'completed' })
        .eq('id', sessionId);

      if (sessionError) throw sessionError;

      // Update local state
      setIsCompleted(true);
      if (session) {
        const updatedSession = { ...session, status: 'completed' as const };
        setSession(updatedSession);
      }

      // Clear all caches and notify other components
      sessionCache.delete(sessionId);
      
      console.log('SessionStudents: Marking completed, clearing caches and dispatching events...');
      clearAllSessionCaches();
      clearTakeAttendanceCache();

      // Refresh attendance records
      const { data: attendanceData } = await supabase
        .from('attendance')
        .select('*')
        .eq('session_id', sessionId);
      
      const attendanceMap = new Map();
      (attendanceData || []).forEach((record: any) => {
        attendanceMap.set(record.student_id, record);
      });
      setAttendanceRecords(attendanceMap);

      // Update students with new attendance
      const studentsWithAttendance = allStudents.map(student => ({
        ...student,
        attendance: attendanceMap.get(student.id) || null
      }));
      setAllStudents(studentsWithAttendance);

      // Notify parent component to refresh
      if (onSessionUpdated) {
        onSessionUpdated();
      }

    } catch (error) {
      console.error('Error marking session as completed:', error);
    } finally {
      setIsMarkingCompleted(false);
    }
  };

  return (
    <div className="w-full flex flex-col" style={{ height: '670px' }}>
      {/* Header */}
      <div className="pb-2 mb-3 flex-shrink-0">
        <h2 className="text-education-navy text-xl font-semibold">
          Session Details
        </h2>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <div className="grid grid-cols-[400px_1px_1fr] gap-0 overflow-hidden" style={{ height: 'calc(100% - 50px)' }}>
          {/* Left Column - Session Info (Read-only) */}
          <div className="pr-6 space-y-3 overflow-y-auto">
            {/* Type */}
            <div className="space-y-1.5">
              <Label className="text-sm">Type</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input capitalize flex items-center">
                {loading && !session ? (
                  <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                ) : (
                  session?.type || ''
                )}
              </div>
            </div>

            {/* Title */}
            <div className="space-y-1.5">
              <Label className="text-sm">Title</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                {loading && !session ? (
                  <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                ) : (
                  session?.title || ''
                )}
              </div>
            </div>

            {/* Program */}
            <div className="space-y-1.5">
              <Label className="text-sm">Program</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                {loading && !session ? (
                  <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                ) : (
                  session?.program || ''
                )}
              </div>
            </div>

            {/* Year */}
            <div className="space-y-1.5">
              <Label className="text-sm">Year</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                {loading && !session ? (
                  <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                ) : (
                  session?.year || ''
                )}
              </div>
            </div>

            {/* Section */}
            <div className="space-y-1.5">
              <Label className="text-sm">Section</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                {loading && !session ? (
                  <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                ) : (
                  session?.section || ''
                )}
              </div>
            </div>

            {/* Date */}
            <div className="space-y-1.5">
              <Label className="text-sm">Date</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                {loading && !session ? (
                  <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                ) : session ? (
                  new Date(session.date).toLocaleDateString('en-US', {
                    weekday: 'short',
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                  })
                ) : (
                  ''
                )}
              </div>
            </div>

            {/* Time Fields Side by Side */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-sm">Start Time</Label>
                <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                  {loading && !session ? (
                    <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                  ) : (
                    formatTime(session?.time_in || '')
                  )}
                </div>
              </div>

              <div className="space-y-1.5">
                <Label className="text-sm">End Time</Label>
                <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                  {loading && !session ? (
                    <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
                  ) : (
                    session?.time_out ? formatTime(session.time_out) : 'TBD'
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Vertical Divider - Full height */}
          <div className="bg-gray-200 w-px"></div>

          {/* Right Column - Students Section (Scrollable) */}
          <div className="pl-6 flex flex-col min-h-0 overflow-hidden">
            {/* Students Label and Search - Same Row */}
            <div className="flex items-center justify-between mb-3 flex-shrink-0">
              <h3 className="font-semibold text-sm">
                Students: {filteredStudents.length}
              </h3>
              {/* Search Bar */}
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Search:</span>
                <Input
                  placeholder="Search by name or ID..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-8 w-64 text-xs"
                />
              </div>
            </div>
            
            {/* Students Table - Scrollable with visible scrollbar */}
            <div className="border rounded-lg flex-1 min-h-0 overflow-hidden flex flex-col">
              <div className="overflow-y-auto flex-1 visible-scrollbar">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr className="text-xs text-gray-700 uppercase">
                      <th scope="col" className="px-3 py-1.5 text-left font-semibold">Student</th>
                      <th scope="col" className="px-3 py-1.5 text-left font-semibold">Student ID</th>
                      <th scope="col" className="px-3 py-1.5 text-left font-semibold">Program</th>
                      <th scope="col" className="px-3 py-1.5 text-left font-semibold">Year</th>
                      <th scope="col" className="px-3 py-1.5 text-left font-semibold">Section</th>
                      <th scope="col" className="px-3 py-1.5 text-left font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {loading && allStudents.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="px-3 py-12 text-center">
                          <Loader2 className="h-6 w-6 animate-spin text-gray-400 mx-auto" />
                        </td>
                      </tr>
                    ) : allStudents.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="px-3 py-12 text-center">
                          <div className="flex flex-col items-center text-gray-400">
                            <Users className="h-10 w-10 mb-2" />
                            <p className="text-sm">No students found</p>
                          </div>
                        </td>
                      </tr>
                    ) : (
                      filteredStudents.slice(0, displayLimit).map((student) => {
                        const attendance = student.attendance;
                        return (
                          <tr key={student.id} className="hover:bg-gray-50">
                            <td className="px-3 py-1.5 text-xs text-gray-900">{student.full_name}</td>
                            <td className="px-3 py-1.5 text-xs text-gray-600">{student.student_id}</td>
                            <td className="px-3 py-1.5 text-xs text-gray-600">{student.program}</td>
                            <td className="px-3 py-1.5 text-xs text-gray-600">{student.year}</td>
                            <td className="px-3 py-1.5 text-xs text-gray-600">{student.section}</td>
                            <td className="px-3 py-1.5 text-xs">
                              {attendance ? (
                                attendance.status === 'present' ? (
                                  <span className="text-green-600 font-medium">Present</span>
                                ) : attendance.status === 'absent' ? (
                                  <span className="text-red-600 font-medium">Absent</span>
                                ) : attendance.status === 'late' ? (
                                  <span className="text-yellow-600 font-medium">Late</span>
                                ) : (
                                  <span className="text-blue-600 font-medium capitalize">{attendance.status}</span>
                                )
                              ) : (
                                <span className="text-gray-400 italic">Not recorded</span>
                              )}
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
              {/* See More/All Buttons */}
              {filteredStudents.length > 50 && displayLimit < filteredStudents.length && (
                <div className="flex gap-4 py-2 px-3 border-t justify-center">
                  <button
                    type="button"
                    onClick={() => setDisplayLimit(prev => prev + 50)}
                    className="text-xs text-blue-600 hover:text-blue-800 hover:underline"
                  >
                    see more...
                  </button>
                  <button
                    type="button"
                    onClick={() => setDisplayLimit(filteredStudents.length)}
                    className="text-xs text-blue-600 hover:text-blue-800 hover:underline"
                  >
                    see all...
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Fixed Mark Completed Button at Bottom Right */}
        <div className="pt-1 flex justify-end flex-shrink-0">
          {isCompleted ? (
            <div className="text-sm text-gray-600 font-medium py-2">
              This session is completed
            </div>
          ) : (
            <Button 
              type="button"
              className="bg-education-blue hover:bg-education-blue/90"
              onClick={handleMarkCompleted}
              disabled={isMarkingCompleted}
            >
              {isMarkingCompleted ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Marking...
                </>
              ) : (
                'Mark Completed'
              )}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};
