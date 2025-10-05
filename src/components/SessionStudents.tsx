import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, ArrowLeft, User, RefreshCw } from "lucide-react";
import { fetchSessionStudents } from "@/lib/supabaseService";
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
        
        // Transform students data to match expected format
        const studentsList = sessionStudents
          .map(student => ({
            ...student,
            full_name: student.full_name || `${student.firstname || ''} ${student.surname || ''}`.trim(),
            status: student.status || 'absent',
            time_in: student.time_in || null,
            time_out: student.time_out || null
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
      <div className="p-8 flex flex-col items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
        <p className="text-muted-foreground">Loading session details...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
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
      <div className="p-8">
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
    <div className="space-y-4">

      {/* Session Details - Inline Format */}
      <div className="mb-8">
        <div className="space-y-2">
          <p className="text-sm">
            <span className="text-muted-foreground">Type:</span> <span className="font-medium capitalize">{session.type}</span>
          </p>
          <p className="text-sm">
            <span className="text-muted-foreground">Program:</span> <span className="font-medium">{session.program}</span>
          </p>
          <p className="text-sm">
            <span className="text-muted-foreground">Date:</span> <span className="font-medium">
              {new Date(session.date).toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric'
              })}
            </span>
          </p>
          <p className="text-sm">
            <span className="text-muted-foreground">Year:</span> <span className="font-medium">{session.year}</span>
          </p>
          <p className="text-sm">
            <span className="text-muted-foreground">Time:</span> <span className="font-medium">
              {formatTime(session.time_in)} - {session.time_out ? formatTime(session.time_out) : 'TBD'}
            </span>
          </p>
          <p className="text-sm">
            <span className="text-muted-foreground">Section:</span> <span className="font-medium">{session.section}</span>
          </p>
          <p className="text-sm">
            <span className="text-muted-foreground">Attendees:</span> <span className="font-medium">{pagination.totalCount} students</span>
          </p>
        </div>
      </div>
      
      {session.description && (
        <div className="mb-6 p-4 bg-muted/30 rounded-lg border border-border/50">
          <p className="text-sm font-medium text-muted-foreground mb-2">Notes</p>
          <p className="text-sm text-foreground">{session.description}</p>
        </div>
      )}

      {/* Students List Section */}
      <div className="pt-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
          <h3 className="text-lg font-medium">Students</h3>
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

            {paginatedStudents.length > 0 ? (
              <div className="bg-white rounded-md border border-gray-200 overflow-hidden">
                <table className="min-w-full divide-y divide-gray-200 text-sm">
                  <thead className="bg-gray-50">
                    <tr className="text-xs text-gray-500">
                      <th scope="col" className="px-6 py-2 text-left font-medium">Student</th>
                      <th scope="col" className="px-6 py-2 text-left font-medium">ID</th>
                      <th scope="col" className="px-6 py-2 text-left font-medium">Program</th>
                      <th scope="col" className="px-6 py-2 text-left font-medium">Year & Section</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {paginatedStudents.map((student) => {
                      return (
                        <tr key={student.id} className="hover:bg-gray-50">
                          <td className="px-6 py-2 whitespace-nowrap">
                            <div className="text-sm font-medium text-gray-900">
                              {student.full_name}
                            </div>
                          </td>
                          <td className="px-6 py-2 whitespace-nowrap text-gray-500 text-sm">
                            {student.student_id}
                          </td>
                          <td className="px-6 py-2 whitespace-nowrap">
                            <div className="text-sm text-gray-500 truncate max-w-[120px]">{student.program}</div>
                          </td>
                          <td className="px-6 py-2 whitespace-nowrap">
                            <div className="text-sm text-gray-500">
                              {student.year} • {student.section}
                            </div>
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
    </div>
  );
}