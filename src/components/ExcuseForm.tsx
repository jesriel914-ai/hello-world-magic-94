import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Loader2, CheckCircle, FileImage, Upload } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { format } from "date-fns";
import { useToast } from "@/hooks/use-toast";

interface ExcuseFormProps {
  initialData?: any;
  onSuccess?: () => void;
  onCancel?: () => void;
  markAsChanged?: () => void;
}

export default function ExcuseForm({ initialData, onSuccess, onCancel, markAsChanged }: ExcuseFormProps) {
  const { toast } = useToast();
  const [selectedSession, setSelectedSession] = useState<any>(null);
  const [selectedStudent, setSelectedStudent] = useState<any>(null);
  const [sessions, setSessions] = useState<any[]>([]);
  const [absentStudents, setAbsentStudents] = useState<any[]>([]);
  const [excuseImage, setExcuseImage] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string>('');
  const [sessionSearch, setSessionSearch] = useState('');
  const [studentSearch, setStudentSearch] = useState('');
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [loadingStudents, setLoadingStudents] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [rightView, setRightView] = useState<'sessions' | 'students' | 'excuse-letter'>('sessions');

  // Fetch completed sessions on mount
  useEffect(() => {
    fetchCompletedSessions();
  }, []);

  // Load initial data if editing
  useEffect(() => {
    if (initialData) {
      if (initialData.documentation_url) {
        setImagePreviewUrl(initialData.documentation_url);
        setRightView('excuse-letter');
      }
      // Load session and student if IDs provided
      if (initialData.session_id) {
        loadSessionById(parseInt(initialData.session_id));
      }
      if (initialData.student_id) {
        loadStudentById(parseInt(initialData.student_id));
      }
    }
  }, [initialData]);

  // Fetch absent students when session changes
  useEffect(() => {
    if (selectedSession) {
      fetchAbsentStudents(selectedSession.id);
    } else {
      setAbsentStudents([]);
    }
  }, [selectedSession]);

  const fetchCompletedSessions = async () => {
    setLoadingSessions(true);
    try {
      const { data: sessionsData, error } = await supabase
        .from('sessions')
        .select('id, title, type, date, time_in, time_out, program, year, section, status')
        .eq('status', 'completed')
        .order('date', { ascending: false });

      if (error) throw error;

      // Fetch attendance counts for each session
      const sessionsWithCounts = await Promise.all(
        (sessionsData || []).map(async (session) => {
          const { data: attendance } = await supabase
            .from('attendance')
            .select('status')
            .eq('session_id', session.id);

          const total = attendance?.length || 0;
          const present = attendance?.filter(a => a.status === 'present').length || 0;
          const absent = attendance?.filter(a => a.status === 'absent').length || 0;

          return {
            ...session,
            students: total,
            present,
            absent
          };
        })
      );

      setSessions(sessionsWithCounts);
    } catch (error) {
      console.error('Error fetching sessions:', error);
      toast({ title: "Error", description: "Failed to load sessions", variant: "destructive" });
    } finally {
      setLoadingSessions(false);
    }
  };

  const fetchAbsentStudents = async (sessionId: number) => {
    setLoadingStudents(true);
    try {
      const { data, error } = await supabase
        .from('attendance')
        .select(`
          student_id,
          students:student_id (
            id,
            student_id,
            firstname,
            surname,
            program,
            year,
            section
          )
        `)
        .eq('session_id', sessionId)
        .eq('status', 'absent');

      if (error) throw error;

      const studentsData = (data || [])
        .filter(record => record.students)
        .map(record => ({
          id: record.students.id,
          student_id: record.students.student_id,
          firstname: record.students.firstname,
          surname: record.students.surname,
          full_name: `${record.students.firstname} ${record.students.surname}`,
          program: record.students.program,
          year: record.students.year,
          section: record.students.section
        }));

      setAbsentStudents(studentsData);
    } catch (error) {
      console.error('Error fetching absent students:', error);
      toast({ title: "Error", description: "Failed to load absent students", variant: "destructive" });
    } finally {
      setLoadingStudents(false);
    }
  };

  const loadSessionById = async (sessionId: number) => {
    const { data, error } = await supabase
      .from('sessions')
      .select('id, title, type, date, time_in, time_out, program, year, section')
      .eq('id', sessionId)
      .single();
    if (data) setSelectedSession(data);
  };

  const loadStudentById = async (studentId: number) => {
    const { data, error } = await supabase
      .from('students')
      .select('id, firstname, surname, student_id, program, year, section')
      .eq('id', studentId)
      .single();
    if (data) setSelectedStudent(data);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setExcuseImage(file);
      // Create new preview URL and revoke old one to prevent memory leaks
      if (imagePreviewUrl && imagePreviewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(imagePreviewUrl);
      }
      setImagePreviewUrl(URL.createObjectURL(file));
      markAsChanged?.();
    }
  };

  const handleSubmit = async () => {
    if (!selectedSession || !selectedStudent || !excuseImage) return;

    try {
      setSubmitting(true);

      // Upload image to storage
      const fileExt = excuseImage.name.split('.').pop();
      const fileName = `${Date.now()}.${fileExt}`;
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('excuse-letters')
        .upload(fileName, excuseImage);

      if (uploadError) throw uploadError;

      // Get public URL
      const { data: { publicUrl } } = supabase.storage
        .from('excuse-letters')
        .getPublicUrl(fileName);

      // Insert excuse application
      const { error: insertError } = await supabase
        .from('excuse_applications')
        .insert({
          session_id: selectedSession.id,
          student_id: selectedStudent.id,
          absence_date: selectedSession.date,
          documentation_url: publicUrl,
          status: 'pending'
        });

      if (insertError) throw insertError;

      toast({ title: "Success", description: "Excuse application submitted successfully" });
      onSuccess?.();
    } catch (error) {
      console.error('Error submitting excuse:', error);
      toast({ title: "Error", description: "Failed to submit excuse application", variant: "destructive" });
    } finally {
      setSubmitting(false);
    }
  };

  const filteredSessions = sessions.filter(session =>
    session.title.toLowerCase().includes(sessionSearch.toLowerCase()) ||
    session.type.toLowerCase().includes(sessionSearch.toLowerCase())
  );

  const filteredStudents = absentStudents.filter(student =>
    student.full_name.toLowerCase().includes(studentSearch.toLowerCase()) ||
    student.student_id.toLowerCase().includes(studentSearch.toLowerCase())
  );

  const getTypeText = (type: string) => {
    switch (type) {
      case 'class':
        return 'Class';
      case 'event':
        return 'Event';
      default:
        return 'Other';
    }
  };

  return (
    <div className="w-full flex flex-col" style={{ height: '673px' }}>
      {/* Header */}
      <div className="pb-2 mb-3 flex-shrink-0">
        <h2 className="text-education-navy text-xl font-semibold">
          {initialData?.id ? 'Edit Excuse Application' : 'New Excuse Application'}
        </h2>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        <div className="grid grid-cols-[400px_1px_1fr] gap-0 overflow-hidden" style={{ height: 'calc(100% - 50px)' }}>
          {/* Left Column - Details Fields */}
          <div className="pr-6 space-y-3 overflow-y-auto">
            {/* Student Details */}
            <div className="space-y-1.5">
              <Label className="text-sm">Student Name</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                {selectedStudent ? (
                  <span className="truncate">{selectedStudent.firstname} {selectedStudent.surname}</span>
                ) : (
                  <span className="text-muted-foreground">No student selected</span>
                )}
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-sm">Student ID</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                {selectedStudent?.student_id || '-'}
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-sm">Program</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                {selectedStudent?.program || '-'}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <Label className="text-sm">Year</Label>
                <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                  {selectedStudent?.year || '-'}
                </div>
              </div>
              <div className="space-y-1.5">
                <Label className="text-sm">Section</Label>
                <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                  {selectedStudent?.section || '-'}
                </div>
              </div>
            </div>

            {/* Session Details */}
            <div className="space-y-1.5">
              <Label className="text-sm">Session</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                {selectedSession ? (
                  <span className="truncate">{selectedSession.title}</span>
                ) : (
                  <span className="text-muted-foreground">No session selected</span>
                )}
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-sm">Absence Date</Label>
              <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center">
                {selectedSession ? format(new Date(selectedSession.date), 'MMM d, yyyy') : '-'}
              </div>
            </div>
          </div>

          {/* Vertical Divider */}
          <div className="bg-gray-200 w-px"></div>

          {/* Right Column - Dropdown Controlled View */}
          <div className="pl-6 flex flex-col min-h-0 overflow-hidden">
            {/* View Content */}
            {rightView === 'sessions' ? (
              // Sessions Table
              <>
                <div className="flex items-center justify-between mb-3 flex-shrink-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">Select:</span>
                    <Select value={rightView} onValueChange={(val: any) => setRightView(val)}>
                      <SelectTrigger className="w-64 h-8 text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sessions">Sessions</SelectItem>
                        <SelectItem value="students">Students</SelectItem>
                        <SelectItem value="excuse-letter">Excuse Letter</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">Search:</span>
                    <Input
                      placeholder="Search sessions..."
                      value={sessionSearch}
                      onChange={(e) => setSessionSearch(e.target.value)}
                      className="h-8 text-sm w-64"
                    />
                  </div>
                </div>

                <div className="border rounded-md flex-1 min-h-0 overflow-auto visible-scrollbar">
                  <table className="min-w-full divide-y divide-gray-200 text-xs">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr className="text-xs text-black h-7">
                        <th className="px-2 py-1 text-left font-semibold uppercase">Type</th>
                        <th className="px-2 py-1 text-left font-semibold uppercase">Title</th>
                        <th className="px-2 py-1 text-left font-semibold uppercase">Date</th>
                        <th className="px-2 py-1 text-left font-semibold uppercase">Students</th>
                        <th className="px-2 py-1 text-left font-semibold uppercase">Present</th>
                        <th className="px-2 py-1 text-left font-semibold uppercase">Absent</th>
                        <th className="px-2 py-1 text-right font-semibold uppercase"></th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200 text-xs text-gray-500">
                      {loadingSessions ? (
                        <tr>
                          <td colSpan={7} className="px-2 py-4 text-center">
                            <Loader2 className="h-4 w-4 animate-spin mx-auto" />
                          </td>
                        </tr>
                      ) : filteredSessions.length === 0 ? (
                        <tr>
                          <td colSpan={7} className="px-2 py-4 text-center text-gray-400">
                            No completed sessions found
                          </td>
                        </tr>
                      ) : (
                        filteredSessions.map((session) => (
                          <tr key={session.id} className="hover:bg-gray-50 h-7">
                            <td className="px-2 py-1 whitespace-nowrap">{getTypeText(session.type)}</td>
                            <td className="px-2 py-1 whitespace-nowrap">{session.title}</td>
                            <td className="px-2 py-1 whitespace-nowrap">
                              {format(new Date(session.date), 'MMM d, yyyy')}
                            </td>
                            <td className="px-2 py-1 whitespace-nowrap">{session.students}</td>
                            <td className="px-2 py-1 whitespace-nowrap">{session.present}</td>
                            <td className="px-2 py-1 whitespace-nowrap">{session.absent}</td>
                            <td className="px-2 py-1 whitespace-nowrap text-right">
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-5 px-2 text-xs"
                                onClick={() => {
                                  // If changing session, clear student
                                  if (selectedSession?.id !== session.id) {
                                    setSelectedStudent(null);
                                  }
                                  setSelectedSession(session);
                                  markAsChanged?.();
                                }}
                              >
                                {selectedSession?.id === session.id ? (
                                  <>
                                    <CheckCircle className="h-3 w-3 mr-1 text-green-600" />
                                    Selected
                                  </>
                                ) : (
                                  'Select'
                                )}
                              </Button>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </>
            ) : rightView === 'students' ? (
              // Students Table
              <>
                <div className="flex items-center justify-between mb-3 flex-shrink-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">Select:</span>
                    <Select value={rightView} onValueChange={(val: any) => setRightView(val)}>
                      <SelectTrigger className="w-64 h-8 text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sessions">Sessions</SelectItem>
                        <SelectItem value="students">Students</SelectItem>
                        <SelectItem value="excuse-letter">Excuse Letter</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">Search:</span>
                    <Input
                      placeholder="Search students..."
                      value={studentSearch}
                      onChange={(e) => setStudentSearch(e.target.value)}
                      className="h-8 text-sm w-64"
                    />
                  </div>
                </div>

                <div className="border rounded-md flex-1 min-h-0 overflow-auto visible-scrollbar">
                  <table className="min-w-full divide-y divide-gray-200 text-xs">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr className="text-xs text-black h-7">
                        <th className="px-2 py-1 text-left font-semibold uppercase">Student ID</th>
                        <th className="px-2 py-1 text-left font-semibold uppercase">Name</th>
                        <th className="px-2 py-1 text-left font-semibold uppercase">Program</th>
                        <th className="px-2 py-1 text-left font-semibold uppercase">Year</th>
                        <th className="px-2 py-1 text-left font-semibold uppercase">Section</th>
                        <th className="px-2 py-1 text-right font-semibold uppercase"></th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200 text-xs text-gray-500">
                      {!selectedSession ? (
                        <tr>
                          <td colSpan={6} className="px-2 py-4 text-center text-gray-400">
                            Select session first
                          </td>
                        </tr>
                      ) : loadingStudents ? (
                        <tr>
                          <td colSpan={6} className="px-2 py-4 text-center">
                            <Loader2 className="h-4 w-4 animate-spin mx-auto" />
                          </td>
                        </tr>
                      ) : filteredStudents.length === 0 ? (
                        <tr>
                          <td colSpan={6} className="px-2 py-4 text-center text-gray-400">
                            No absent students found
                          </td>
                        </tr>
                      ) : (
                        filteredStudents.map((student) => (
                          <tr key={student.id} className="hover:bg-gray-50 h-7">
                            <td className="px-2 py-1 whitespace-nowrap">{student.student_id}</td>
                            <td className="px-2 py-1 whitespace-nowrap">{student.full_name}</td>
                            <td className="px-2 py-1 whitespace-nowrap">{student.program}</td>
                            <td className="px-2 py-1 whitespace-nowrap">{student.year}</td>
                            <td className="px-2 py-1 whitespace-nowrap">{student.section}</td>
                            <td className="px-2 py-1 whitespace-nowrap text-right">
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-5 px-2 text-xs"
                                onClick={() => {
                                  setSelectedStudent(student);
                                  markAsChanged?.();
                                }}
                              >
                                {selectedStudent?.id === student.id ? (
                                  <>
                                    <CheckCircle className="h-3 w-3 mr-1 text-green-600" />
                                    Selected
                                  </>
                                ) : (
                                  'Select'
                                )}
                              </Button>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              // Excuse Letter Preview
              <>
                <div className="flex items-center justify-between mb-3 flex-shrink-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">Select:</span>
                    <Select value={rightView} onValueChange={(val: any) => setRightView(val)}>
                      <SelectTrigger className="w-64 h-8 text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sessions">Sessions</SelectItem>
                        <SelectItem value="students">Students</SelectItem>
                        <SelectItem value="excuse-letter">Excuse Letter</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="border rounded-lg flex-1 min-h-0 overflow-hidden bg-gray-50 relative group">
                {imagePreviewUrl ? (
                  <>
                    <div 
                      className="h-full overflow-auto visible-scrollbar"
                      style={{
                        cursor: 'grab',
                        userSelect: 'none'
                      }}
                    >
                      <img 
                        src={imagePreviewUrl} 
                        alt="Excuse letter preview" 
                        className="w-full h-full object-contain"
                        style={{
                          minHeight: '100%',
                          imageRendering: 'high-quality'
                        }}
                      />
                    </div>
                    {/* Upload button - show on hover when image exists */}
                    <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <label htmlFor="excuse-file-replace" className="cursor-pointer">
                        <Button type="button" variant="outline" className="bg-white/90 hover:bg-white shadow-lg" asChild>
                          <span>
                            <Upload className="h-4 w-4 mr-2" />
                            Replace Image
                          </span>
                        </Button>
                      </label>
                      <input
                        id="excuse-file-replace"
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                    </div>
                  </>
                ) : (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center text-gray-400">
                      <FileImage className="h-16 w-16 mx-auto mb-2 opacity-50" />
                      <p className="text-sm mb-4">No image uploaded</p>
                      <label htmlFor="excuse-file-initial" className="cursor-pointer">
                        <Button type="button" variant="outline" asChild>
                          <span>
                            <Upload className="h-4 w-4 mr-2" />
                            Upload Image
                          </span>
                        </Button>
                      </label>
                      <input
                        id="excuse-file-initial"
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                    </div>
                  </div>
                )}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div className="pt-1 mt-1 flex justify-end flex-shrink-0">
          <Button 
            onClick={handleSubmit}
            disabled={!selectedSession || !selectedStudent || !excuseImage || submitting}
            className="bg-education-blue hover:bg-education-blue/90"
          >
            {submitting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Submitting...
              </>
            ) : (
              'Submit Application'
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
