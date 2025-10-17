import { useState, useEffect, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Users, Loader2, CalendarClock, Search } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { useAuth } from "@/hooks/useAuth";
import { fetchUserRole } from "@/lib/getUserRole";

export type AttendanceType = "class" | "event" | "other";

export interface SessionData {
  title: string;
  program: string;
  year: string;
  date: string;
  timeIn: string;
  timeOut: string;
  attendanceType: AttendanceType;
}

interface AttendanceFormProps {
  onSuccess?: () => void;
  onSubmit?: (session: SessionData) => void;
  initialData?: Partial<SessionData> & { id?: number };
}

interface Student {
  id: number;
  student_id: string;
  firstname: string;
  surname: string;
  full_name: string;
  program: string;
  year: string;
}

const AttendanceForm = ({ onSuccess, onSubmit, initialData }: AttendanceFormProps) => {
  const { user } = useAuth();

  const [currentRole, setCurrentRole] = useState<string>("");
  const [roleReady, setRoleReady] = useState<boolean>(false);

  // Read cached role
  const getCachedUserRole = () => {
    try {
      return localStorage.getItem('userRole') || '';
    } catch {
      return '';
    }
  };

  useEffect(() => {
    let isMounted = true;
    const resolveRole = async () => {
      const cached = getCachedUserRole();
      if (isMounted && cached) {
        setCurrentRole(cached);
        setRoleReady(true);
      }

      if (user?.id) {
        try {
          const dbRole = await fetchUserRole(user.id);
          if (isMounted && dbRole) {
            setCurrentRole(prevRole => {
              if (prevRole !== dbRole) {
                return dbRole;
              }
              return prevRole;
            });
          }
        } catch (error) {
          console.error('Error resolving user role:', error);
        }
      }

      if (isMounted && !cached) {
        setRoleReady(true);
      }
    };
    
    resolveRole();
    return () => {
      isMounted = false;
    };
  }, [user?.id]);

  const allowedTypes: AttendanceType[] = useMemo(() => {
    if (currentRole === "admin") {
      return ["class", "event", "other"];
    }
    
    if (currentRole === "attendance checker") {
      return ["class", "event", "other"];
    }
    
    if (!currentRole && !roleReady) {
      const cached = getCachedUserRole();
      if (cached === "admin" || cached === "attendance checker") {
        return ["class", "event", "other"];
      }
    }
    
    return ["class"];
  }, [currentRole, roleReady]);

  const [attendanceType, setAttendanceType] = useState<AttendanceType>(() => {
    if (initialData?.attendanceType) {
      return initialData.attendanceType;
    }
    return "class";
  });

  const [formData, setFormData] = useState({
    title: initialData?.title || "",
    program: initialData?.program || "",
    year: initialData?.year || "",
    date: initialData?.date || "",
    timeIn: initialData?.timeIn || "",
    timeOut: initialData?.timeOut || "",
  });
  
  useEffect(() => {
    if (roleReady && allowedTypes.length > 0) {
      if (!allowedTypes.includes(attendanceType)) {
        setAttendanceType(allowedTypes[0]);
      }
    }
  }, [roleReady, allowedTypes, attendanceType, currentRole]);
  
  // State for dropdown options
  const [loadingOptions, setLoadingOptions] = useState(false);
  const [studentOptions, setStudentOptions] = useState<{
    programs: string[];
    years: string[];
  }>({ programs: [], years: [] });
  
  // State for students table
  const [students, setStudents] = useState<Student[]>([]);
  const [loadingStudents, setLoadingStudents] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [displayLimit, setDisplayLimit] = useState(50);
  
  
  // Fetch students for the table
  const fetchStudentsForTable = useCallback(async () => {
    if (!formData.program || !formData.year) {
      setStudents([]);
      return;
    }

    setLoadingStudents(true);
    try {
      const yearShort = formData.year === 'All Years' ? null : formData.year.replace(' Year', '');
      
      // Helper function to fetch all students with pagination
      const fetchAllStudents = async () => {
        const allStudents: any[] = [];
        let from = 0;
        const pageSize = 1000; // Supabase default limit
        
        while (true) {
          let query = supabase
            .from('students')
            .select('id, student_id, firstname, surname, program, year')
            .order('surname', { ascending: true })
            .range(from, from + pageSize - 1);

          // Apply program filter only if not "All Programs"
          if (formData.program && formData.program !== 'All Programs') {
            query = query.eq('program', formData.program);
          }

          // Apply year filter only if not "All Years"
          if (yearShort) {
            query = query.eq('year', yearShort);
          }


          const { data, error } = await query;

          if (error) throw error;
          
          if (!data || data.length === 0) break;
          
          allStudents.push(...data);
          
          // If we got less than pageSize, we've reached the end
          if (data.length < pageSize) break;
          
          from += pageSize;
        }
        
        return allStudents;
      };

      const data = await fetchAllStudents();

      const studentsWithFullName = (data || []).map(student => ({
        ...student,
        full_name: `${student.firstname} ${student.surname}`
      }));

      setStudents(studentsWithFullName);
    } catch (error) {
      console.error('Error fetching students:', error);
      setStudents([]);
    } finally {
      setLoadingStudents(false);
    }
  }, [formData.program, formData.year]);

  // Fetch students when program or year changes
  useEffect(() => {
    fetchStudentsForTable();
  }, [fetchStudentsForTable]);



  // Fetch programs from students table
  const loadPrograms = useCallback(async () => {
    try {
      let allStudents: { program: string | null }[] = [];
      let page = 0;
      const pageSize = 1000;
      let hasMore = true;
      
      while (hasMore) {
        const { data, error } = await supabase
          .from('students')
          .select('program')
          .not('program', 'is', null)
          .range(page * pageSize, (page + 1) * pageSize - 1);
          
        if (error) throw error;
        
        if (data && data.length > 0) {
          allStudents = [...allStudents, ...data];
          page++;
          
          if (data.length < pageSize) hasMore = false;
        } else {
          hasMore = false;
        }
      }
      
      const programSet = new Set<string>();
      allStudents.forEach(student => {
        if (student.program) {
          const program = student.program.toString().trim();
          if (program) programSet.add(program);
        }
      });
      
      const uniquePrograms = Array.from(programSet).sort((a, b) => 
        a.localeCompare(b, 'en', { sensitivity: 'base' })
      );
      
      setStudentOptions(prev => ({
        ...prev,
        programs: uniquePrograms,
        years: ['1st Year', '2nd Year', '3rd Year', '4th Year']
      }));
    } catch (error) {
      console.error('Error fetching programs:', error);
      toast.error('Failed to load programs');
      setStudentOptions(prev => ({
        ...prev,
        programs: [],
        years: ['1st Year', '2nd Year', '3rd Year', '4th Year']
      }));
    }
  }, []);

  useEffect(() => {
    loadPrograms();
  }, [loadPrograms]);


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const sessionData: SessionData = {
      ...formData,
      program: formData.program === 'All Programs' ? 'All Programs' : formData.program,
      year: formData.year === 'All Year Levels' ? 'All Years' : formData.year,
      attendanceType: attendanceType
    };
    
    try {
      if (onSubmit) {
        onSubmit(sessionData);
      }
      
      toast.success(`${formData.title} session has been ${initialData?.id ? 'updated' : 'created'} successfully.`);
      
      if (!initialData?.id) {
        setFormData({
          title: "",
          program: "",
          year: "",
          date: "",
          timeIn: "",
          timeOut: "",
        });
        setAttendanceType("class");
      }
      
      if (onSuccess) {
        onSuccess();
      }
    } catch (error) {
      console.error("Error saving session:", error);
      toast.error(`Failed to ${initialData?.id ? 'update' : 'create'} session. Please try again.`);
    }
  };

  const getTypeColor = (type: AttendanceType) => {
    switch (type) {
      case "class": return "bg-gradient-primary";
      case "event": return "bg-gradient-accent";
      case "other": return "bg-education-navy";
    }
  };

  return (
    <div className="w-full flex flex-col" style={{ height: '685px' }}>
      {/* Header - Reduced spacing */}
      <div className="pb-2 mb-3 flex-shrink-0">
        <h2 className="text-education-navy text-xl font-semibold">
          {initialData?.id ? 'Edit Session' : 'Create New Session'}
        </h2>
      </div>

      {/* Main Content - Fixed left, scrollable right */}
      <form onSubmit={handleSubmit} className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <div className="grid grid-cols-[400px_1px_1fr] gap-0 overflow-hidden" style={{ height: 'calc(100% - 60px)' }}>
          {/* Left Column - Form Fields (Fixed) */}
          <div className="pr-6 space-y-3 overflow-y-auto">
            {/* Type */}
            <div className="space-y-1.5">
              <Label htmlFor="type" className="text-sm">Type</Label>
              <Select 
                value={attendanceType}
                onValueChange={(value: AttendanceType) => setAttendanceType(value)}
              >
                <SelectTrigger className="w-full h-9 text-sm bg-gray-100">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {allowedTypes.includes("class") && <SelectItem value="class">Class</SelectItem>}
                  {allowedTypes.includes("event") && <SelectItem value="event">Event</SelectItem>}
                  {allowedTypes.includes("other") && <SelectItem value="other">Other</SelectItem>}
                </SelectContent>
              </Select>
            </div>

            {/* Title */}
            <div className="space-y-1.5">
              <Label htmlFor="title" className="text-sm">Title</Label>
              <Input
                id="title"
                value={formData.title}
                onChange={(e) => setFormData({...formData, title: e.target.value})}
                placeholder="Enter title"
                className="h-9 text-sm bg-gray-100"
                required
              />
            </div>

            {/* Program */}
            <div className="space-y-1.5">
              <Label htmlFor="program" className="text-sm">Program</Label>
              <Select 
                value={formData.program}
                onValueChange={(value) => {
                  setFormData(prev => ({
                    ...prev,
                    program: value,
                    year: prev.program !== value ? "" : prev.year,
                  }));
                }}
                disabled={loadingOptions}
              >
                <SelectTrigger className="w-full h-9 text-sm bg-gray-100">
                  <SelectValue placeholder={loadingOptions ? "Loading..." : "Select program"} />
                </SelectTrigger>
                <SelectContent>
                  {loadingOptions ? (
                    <div className="flex items-center justify-center p-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                    </div>
                  ) : (
                    <>
                      <SelectItem value="All Programs">All Programs</SelectItem>
                      {studentOptions.programs.map((program) => (
                        <SelectItem key={program} value={program}>
                          {program}
                        </SelectItem>
                      ))}
                    </>
                  )}
                </SelectContent>
              </Select>
            </div>

            {/* Year */}
            <div className="space-y-1.5">
              <Label htmlFor="year" className="text-sm">Year</Label>
              <Select 
                value={formData.year === 'All Years' ? 'All Year Levels' : formData.year}
                onValueChange={(value) => {
                  const newYear = value === 'All Year Levels' ? 'All Years' : value;
                  setFormData(prev => ({
                    ...prev,
                    year: newYear,
                  }));
                }}
                disabled={loadingOptions}
              >
                <SelectTrigger className="w-full h-9 text-sm bg-gray-100">
                  <SelectValue placeholder="Select year" />
                </SelectTrigger>
                <SelectContent>
                  {loadingOptions ? (
                    <div className="flex items-center justify-center p-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                    </div>
                  ) : (
                    <>
                      <SelectItem value="All Year Levels">All Year Levels</SelectItem>
                      {studentOptions.years.map((year) => (
                        <SelectItem key={year} value={year}>
                          {year}
                        </SelectItem>
                      ))}
                    </>
                  )}
                </SelectContent>
              </Select>
            </div>


            {/* Date */}
            <div className="space-y-1.5">
              <Label htmlFor="date" className="text-sm">Date</Label>
              <Input
                id="date"
                type="date"
                value={formData.date}
                onChange={(e) => setFormData({...formData, date: e.target.value})}
                className="h-9 text-sm bg-gray-100"
                required
              />
            </div>

            {/* Time Fields Side by Side */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label htmlFor="timeIn" className="text-sm">Start Time</Label>
                <Input
                  id="timeIn"
                  type="time"
                  value={formData.timeIn}
                  onChange={(e) => setFormData({...formData, timeIn: e.target.value})}
                  className="h-9 text-sm bg-gray-100"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="timeOut" className="text-sm">End Time</Label>
                <Input
                  id="timeOut"
                  type="time"
                  value={formData.timeOut}
                  onChange={(e) => setFormData({...formData, timeOut: e.target.value})}
                  className="h-9 text-sm bg-gray-100"
                />
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
                Students: {students.filter(s => 
                  s.full_name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                  s.student_id.toLowerCase().includes(searchQuery.toLowerCase())
                ).length}
              </h3>
              {/* Search Bar - Matches Sessions page */}
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
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {loadingStudents ? (
                      <tr>
                        <td colSpan={5} className="px-3 py-12 text-center">
                          <Loader2 className="h-6 w-6 animate-spin text-gray-400 mx-auto" />
                        </td>
                      </tr>
                    ) : students.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="px-3 py-12 text-center">
                          <div className="flex flex-col items-center text-gray-400">
                            <Users className="h-10 w-10 mb-2" />
                            <p className="text-sm">No students found</p>
                            <p className="text-xs">Select program and year to view students</p>
                          </div>
                        </td>
                      </tr>
                    ) : (
                      students
                        .filter(student => 
                          student.full_name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          student.student_id.toLowerCase().includes(searchQuery.toLowerCase())
                        )
                        .slice(0, displayLimit)
                        .map((student) => (
                          <tr key={student.id} className="hover:bg-gray-50">
                            <td className="px-3 py-1.5 text-xs text-gray-900">{student.full_name}</td>
                            <td className="px-3 py-1.5 text-xs text-gray-600">{student.student_id}</td>
                            <td className="px-3 py-1.5 text-xs text-gray-600">{student.program}</td>
                            <td className="px-3 py-1.5 text-xs text-gray-600">{student.year}</td>
                          </tr>
                        ))
                    )}
                  </tbody>
                </table>
              </div>
              {/* See More/All Buttons */}
              {students.filter(student => 
                student.full_name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                student.student_id.toLowerCase().includes(searchQuery.toLowerCase())
              ).length > 50 && displayLimit < students.filter(student => 
                student.full_name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                student.student_id.toLowerCase().includes(searchQuery.toLowerCase())
              ).length && (
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
                    onClick={() => setDisplayLimit(students.filter(student => 
                      student.full_name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                      student.student_id.toLowerCase().includes(searchQuery.toLowerCase())
                    ).length)}
                    className="text-xs text-blue-600 hover:text-blue-800 hover:underline"
                  >
                    see all...
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Fixed Submit Button at Bottom Right */}
        <div className="pt-1 flex justify-end flex-shrink-0">
          <Button 
            type="submit" 
            className="bg-education-blue hover:bg-education-blue/90"
            disabled={false}
          >
            {initialData?.id ? 'Update Session' : 'Create Session'}
          </Button>
        </div>
      </form>
    </div>
  );
};

export default AttendanceForm;
