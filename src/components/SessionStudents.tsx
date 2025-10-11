import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Search, ArrowLeft, User, RefreshCw, CheckCircle, XCircle, Clock } from "lucide-react";
import { fetchSessionStudents } from "@/lib/supabaseService";
import { supabase } from "@/lib/supabase";
import type { Student } from "@/types";

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

interface PaginationState {
  currentPage: number;
  pageSize: number;
  totalCount: number;
  totalPages: number;
}

interface SessionStudentsFormProps {
  sessionId: number;
  onClose?: () => void;
}

export default function SessionStudents({ sessionId, onClose }: SessionStudentsFormProps) {
  const [session, setSession] = useState<Session | null>(null);
  const [allStudents, setAllStudents] = useState<Student[]>([]);
  const [filteredStudents, setFilteredStudents] = useState<Student[]>([]);
  const [paginatedStudents, setPaginatedStudents] = useState<Student[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attendanceRecords, setAttendanceRecords] = useState<Map<number, any>>(new Map());

  // Pagination state
  const [pagination, setPagination] = useState<PaginationState>({
    currentPage: 1,
    pageSize: 25,
    totalCount: 0,
    totalPages: 0
  });

  // Fetch session and student data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Validate session ID
        if (!sessionId || sessionId <= 0) {
          throw new Error('Invalid session ID');
        }
        
        // Fetch session details and students
        console.log(`Fetching students for session ID: ${sessionId}`);
        const response = await fetchSessionStudents(sessionId);
        
        if (!response?.session) {
          console.error('No session data in response:', response);
          throw new Error('Session data not found in response');
        }
        
        const sessionData = response.session;
        const sessionStudents = response.students || [];
        
        // Transform session data to match our component's needs
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
          type: sessionData.type || 'class'
        };
        
        setSession(session);
        
        // Fetch attendance records for this session
        const { data: attendanceData } = await supabase
          .from('attendance')
          .select('*')
          .eq('session_id', sessionData.id);
        
        // Create a map of student_id -> attendance record
        const attendanceMap = new Map();
        (attendanceData || []).forEach((record: any) => {
          attendanceMap.set(record.student_id, record);
        });
        setAttendanceRecords(attendanceMap);
        
        // Transform students data to match expected format
        const studentsList = sessionStudents
          .map(student => ({
            ...student,
            full_name: student.full_name || `${student.firstname || ''} ${student.surname || ''}`.trim(),
            attendance: attendanceMap.get(student.id) || null
          }))
          // Sort students alphabetically by full_name
          .sort((a, b) => a.full_name.localeCompare(b.full_name));
        
        setAllStudents(studentsList);
        setFilteredStudents(studentsList);
        
      } catch (err: unknown) {
        // Properly type the error for better TypeScript support
        interface AxiosError {
          response?: {
            data?: { message?: string };
            status?: number;
            headers?: Record<string, unknown>;
          };
          request?: unknown;
          message: string;
        }
        
        const error = err as Error & Partial<AxiosError>;
        
        console.error('Error fetching session students:', {
          error: error.message,
          sessionId,
          responseStatus: error.response?.status,
          responseData: error.response?.data
        });
        
        if (error.response) {
          // The request was made and the server responded with a status code
          // that falls out of the range of 2xx
          setError(`Server error: ${error.response.status} - ${error.response.data?.message || 'Unknown error'}`);
        } else if (error.request) {
          // The request was made but no response was received
          setError('No response from server. Please check your connection.');
        } else {
          // Something happened in setting up the request that triggered an Error
          setError(`Error: ${error.message || 'An unknown error occurred'}`);
        }
      } finally {
        setLoading(false);
      }
    };
    
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

  // Filter students based on search query
  useEffect(() => {
    if (!searchQuery) {
      setFilteredStudents(allStudents);
      return;
    }

    const lowercasedQuery = searchQuery.toLowerCase();
    const filtered = allStudents.filter(
      (student) =>
        student.full_name.toLowerCase().includes(lowercasedQuery) ||
        student.student_id.toLowerCase().includes(lowercasedQuery)
    );
    setFilteredStudents(filtered);
  }, [searchQuery, allStudents]);

  // Update pagination when filtered students change
  useEffect(() => {
    const totalCount = filteredStudents.length;
    const totalPages = Math.ceil(totalCount / pagination.pageSize);
    
    // Reset to page 1 if current page exceeds total pages
    const currentPage = pagination.currentPage > totalPages && totalPages > 0 ? 1 : pagination.currentPage;
    
    setPagination(prev => ({
      ...prev,
      currentPage,
      totalCount,
      totalPages
    }));
    
    // Calculate paginated students
    const startIndex = (currentPage - 1) * pagination.pageSize;
    const endIndex = startIndex + pagination.pageSize;
    setPaginatedStudents(filteredStudents.slice(startIndex, endIndex));
  }, [filteredStudents, pagination.pageSize, pagination.currentPage]);

  // Handle page change
  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > pagination.totalPages) return;
    
    setPagination(prev => ({ ...prev, currentPage: newPage }));
  };

  // Handle page size change
  const handlePageSizeChange = (value: string) => {
    const newPageSize = value === 'all' ? filteredStudents.length : parseInt(value);
    setPagination(prev => ({ 
      ...prev, 
      pageSize: newPageSize, 
      currentPage: 1 
    }));
  };


  if (loading) {
    return (
      <div className="p-8 flex flex-col items-center justify-center" style={{ height: '600px' }}>
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
        <p className="text-muted-foreground">Loading session details...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8" style={{ height: '600px' }}>
        <div className="text-center py-12">
          <h2 className="text-2xl font-bold text-education-navy mb-4">
            {error.includes('not found') ? 'Session Not Found' : 'Error Loading Session'}
          </h2>
          <p className="text-lg text-muted-foreground mb-6">
            {error.includes('ID') 
              ? error 
              : error.includes('not found')
                ? 'The requested session could not be found. It may have been deleted or moved.'
                : 'An error occurred while loading the session details. Please try again.'}
          </p>
          <div className="flex gap-4 justify-center">
            <Button onClick={onClose} variant="outline">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            {!error.includes('ID') && (
              <Button onClick={() => window.location.reload()} variant="default">
                <RefreshCw className="w-4 h-4 mr-2" />
                Try Again
              </Button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="p-8" style={{ height: '600px' }}>
        <div className="text-center py-12">
          <h2 className="text-2xl font-bold text-education-navy mb-4">Session not found</h2>
          <Button onClick={onClose} variant="outline">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Go Back
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col" style={{ height: '600px' }}>
      {/* Scrollable Content */}
      <ScrollArea className="flex-1 hide-scrollbar">
        <div className="space-y-6 pr-4">
          {/* Session Information - No Card */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="md:col-span-2">
              <span className="text-xs text-gray-500 uppercase font-medium">Title</span>
              <p className="text-sm font-medium text-gray-900 mt-1">{session.title}</p>
            </div>
            <div>
              <span className="text-xs text-gray-500 uppercase font-medium">Type</span>
              <p className="text-sm font-medium text-gray-900 capitalize mt-1">{session.type}</p>
            </div>
            <div>
              <span className="text-xs text-gray-500 uppercase font-medium">Program</span>
              <p className="text-sm font-medium text-gray-900 mt-1">{session.program}</p>
            </div>
            <div>
              <span className="text-xs text-gray-500 uppercase font-medium">Year</span>
              <p className="text-sm font-medium text-gray-900 mt-1">{session.year}</p>
            </div>
            <div>
              <span className="text-xs text-gray-500 uppercase font-medium">Section</span>
              <p className="text-sm font-medium text-gray-900 mt-1">{session.section}</p>
            </div>
            <div>
              <span className="text-xs text-gray-500 uppercase font-medium">Date</span>
              <p className="text-sm font-medium text-gray-900 mt-1">
                {new Date(session.date).toLocaleDateString('en-US', {
                  weekday: 'short',
                  year: 'numeric',
                  month: 'short',
                  day: 'numeric'
                })}
              </p>
            </div>
            <div>
              <span className="text-xs text-gray-500 uppercase font-medium">Time</span>
              <p className="text-sm font-medium text-gray-900 mt-1">
                {formatTime(session.time_in)} - {session.time_out ? formatTime(session.time_out) : 'TBD'}
              </p>
            </div>
            <div className="md:col-span-2">
              <span className="text-xs text-gray-500 uppercase font-medium">Total Attendees</span>
              <p className="text-sm font-medium text-gray-900 mt-1">{pagination.totalCount} students</p>
            </div>
          </div>
          
          {session.description && (
            <div className="md:col-span-2">
              <span className="text-xs text-gray-500 uppercase font-medium">Notes</span>
              <p className="text-sm text-gray-700 mt-2">{session.description}</p>
            </div>
          )}

          {/* Separator Line */}
          <div className="border-t border-gray-300 my-6"></div>

          {/* Students List Controls */}
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <h3 className="text-base font-semibold text-gray-900">Students</h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Showed:</span>
              <Select
                value={pagination.pageSize.toString()}
                onValueChange={(value) => handlePageSizeChange(parseInt(value))}
              >
                <SelectTrigger className="w-20 h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">10</SelectItem>
                  <SelectItem value="25">25</SelectItem>
                  <SelectItem value="50">50</SelectItem>
                  <SelectItem value="100">100</SelectItem>
                  <SelectItem value="all">ALL</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="relative w-full sm:w-64">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
              <Input
                placeholder="Search by name or ID..."
                className="pl-10 h-9 w-full bg-background/80 border-border/50 hover:border-border/70 focus:border-primary/50 transition-colors"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>
        </div>

          {/* Students Table */}
          {paginatedStudents.length > 0 ? (
            <div className="bg-white rounded-md border border-gray-200 overflow-hidden">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr className="text-xs text-gray-700 uppercase">
                    <th scope="col" className="px-4 py-3 text-left font-semibold">Student</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold">ID</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold">Program</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold">Year & Section</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold">Status</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {paginatedStudents.map((student) => {
                    const attendance = student.attendance;
                    return (
                      <tr key={student.id} className="hover:bg-gray-50 transition-colors">
                        <td className="px-4 py-3 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">
                            {student.full_name}
                          </div>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-gray-600 text-sm">
                          {student.student_id}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <div className="text-sm text-gray-600 truncate max-w-[120px]">{student.program}</div>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <div className="text-sm text-gray-600">
                            {student.year} • {student.section}
                          </div>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          {attendance ? (
                            attendance.status === 'present' ? (
                              <span className="text-sm font-medium text-green-600">
                                Present
                              </span>
                            ) : attendance.status === 'absent' ? (
                              <span className="text-sm font-medium text-red-600">
                                Absent
                              </span>
                            ) : attendance.status === 'late' ? (
                              <span className="text-sm font-medium text-yellow-600">
                                Late
                              </span>
                            ) : (
                              <span className="text-sm font-medium text-blue-600">
                                {attendance.status}
                              </span>
                            )
                          ) : (
                            <span className="text-xs text-gray-400 italic">Not recorded</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
              <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
                <div className="p-4 rounded-full bg-muted/30 mb-4">
                  <User className="w-12 h-12 text-muted-foreground/60" />
                </div>
                <h3 className="text-lg font-medium text-foreground mb-2">
                  {searchQuery ? 'No matching students found' : 'No students enrolled'}
                </h3>
                <p className="text-muted-foreground max-w-md mx-auto mb-6">
                  {searchQuery 
                    ? 'Try adjusting your search or filter criteria'
                    : 'There are no students enrolled in this session matching the selected criteria.'}
                </p>
                {searchQuery && (
                  <Button 
                    variant="outline" 
                    onClick={() => setSearchQuery('')}
                    className="rounded-full px-6"
                  >
                    Clear search
                  </Button>
                )}
              </div>
            )}
        </div>
      </ScrollArea>
      
      {/* Mark Completed Button - Fixed at bottom */}
      <div className="p-4 border-t bg-white flex justify-end">
        <Button 
          type="button"
        >
          Mark Completed
        </Button>
      </div>
    </div>
  );
}