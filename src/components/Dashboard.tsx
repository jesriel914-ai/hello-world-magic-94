import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CalendarDays, Users, UserCheck, BarChart3, CalendarClock, CheckCircle, TrendingUp, TrendingDown, Activity } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { useNavigate } from "react-router-dom";
import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { supabase } from "@/lib/supabase";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchUserRole } from "@/lib/getUserRole";
import SessionStudents from "@/components/SessionStudents";
import { format as formatDate } from 'date-fns';

// Use the same role caching system as navigation
const getCachedUserRole = (): string | null => {
  try {
    return localStorage.getItem('userRole');
  } catch {
    return null;
  }
};

const getCachedUserId = (): string | null => {
  try {
    return localStorage.getItem('userId');
  } catch {
    return null;
  }
};

const setCachedUserRole = (role: string, userId: string) => {
  try {
    localStorage.setItem('userRole', role);
    localStorage.setItem('userId', userId);
  } catch {
    // Ignore localStorage errors
  }
};

let cachedUserRole: string | null = getCachedUserRole();
let cachedUserId: string | null = getCachedUserId();

// Enhanced mock data generation with more realistic patterns
const generateMockData = (period: 'daily' | 'weekly' | 'monthly') => {
  if (period === 'daily') {
    return Array.from({ length: 7 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (6 - i));
      const dayName = date.toLocaleDateString('en-US', { weekday: 'short' });
      const isWeekend = date.getDay() === 0 || date.getDay() === 6;
      const baseAttendance = isWeekend ? 0.85 : 0.92; // Lower attendance on weekends
      const totalStudents = 1500;
      const present = Math.floor(totalStudents * baseAttendance + (Math.random() - 0.5) * 50);
      const absent = totalStudents - present;
      return {
        name: dayName,
        present,
        absent,
        date: date.toISOString().split('T')[0]
      };
    });
  } else if (period === 'weekly') {
    return Array.from({ length: 8 }, (_, i) => {
      const baseAttendance = 0.90 + (Math.random() - 0.5) * 0.1;
      const totalStudents = 1500;
      const present = Math.floor(totalStudents * baseAttendance);
      const absent = totalStudents - present;
      return {
        name: `Week ${i + 1}`,
        present,
        absent,
        week: i + 1
      };
    });
  } else { // monthly
    return [
      { name: 'Jan', present: 1350, absent: 150, month: 1 },
      { name: 'Feb', present: 1320, absent: 180, month: 2 },
      { name: 'Mar', present: 1380, absent: 120, month: 3 },
      { name: 'Apr', present: 1410, absent: 90, month: 4 },
      { name: 'May', present: 1440, absent: 60, month: 5 },
      { name: 'Jun', present: 1470, absent: 30, month: 6 },
      { name: 'Jul', present: 1485, absent: 15, month: 7 },
      { name: 'Aug', present: 1500, absent: 0, month: 8 },
    ];
  }
};



const Dashboard = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [userProfile, setUserProfile] = useState<any>(null);
  const [userRole, setUserRole] = useState<string>(() => {
    return cachedUserRole || 'user';
  });
  const [totalStudents, setTotalStudents] = useState(0);
  const [loading, setLoading] = useState(true);
  const isInitialMount = useRef(true);
  const [timePeriod, setTimePeriod] = useState<'daily' | 'weekly' | 'monthly'>('monthly');
  const [chartData, setChartData] = useState<Array<{name: string, present: number, absent: number}>>([]);
  const [realTimeStats, setRealTimeStats] = useState({
    todayAttendance: 0,
    todaySessions: 0,
    activeClasses: 0,
    pendingExcuses: 0
  });

  // Mock placeholders for new dashboard subtitles (replace with real data later)
  const mockYesterdayPresents = 1354; // Count of presents yesterday
  const mockYesterdayAttendanceRate = 93.8; // % yesterday
  const mockCompletedSessionsToday = 13; // completed sessions today

  const yesterdayHigherAttendance = mockYesterdayPresents > realTimeStats.todayAttendance;
  const yesterdayHigherRate = mockYesterdayAttendanceRate > 94.2; // today's mock rate
  
  // Academic Year state
  const [academicYear, setAcademicYear] = useState<{
    year: string;
    semester: string;
    status: 'active' | 'inactive';
  } | null>(null);
  
  // Recent Sessions interface and state
  interface RecentSession {
    id: number;
    title: string;
    students: number;
    date: string;
    time_in: string;
    time_out: string;
    type: 'class' | 'event' | 'other';
  }
  
  const [recentSessions, setRecentSessions] = useState<RecentSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(null);
  const [isStudentsModalOpen, setIsStudentsModalOpen] = useState(false);

  const fetchRecentSessions = useCallback(async () => {
    try {
      // Fetch 5 latest COMPLETED sessions ordered by updated_at (when they were marked completed)
      const { data: sessions, error } = await supabase
        .from('sessions')
        .select('id, title, type, date, time_in, time_out, program, year, section, status')
        .eq('status', 'completed')
        .order('updated_at', { ascending: false })
        .limit(5);
      
      if (error) throw error;
      
      // Calculate student count for each session
      const sessionsWithCounts = await Promise.all(
        (sessions || []).map(async (session) => {
          let studentCount = 0;
          
          try {
            let countQuery = supabase
              .from('students')
              .select('*', { count: 'exact', head: true });
            
            if (session.program && !session.program.toLowerCase().includes('all')) {
              countQuery = countQuery.eq('program', session.program);
            }
            
            if (session.year && !session.year.toLowerCase().includes('all')) {
              let yearValue = session.year;
              if (yearValue.endsWith(' Year')) {
                yearValue = yearValue.replace(' Year', '');
              }
              countQuery = countQuery.eq('year', yearValue);
            }
            
            if (session.section && !session.section.toLowerCase().includes('all')) {
              countQuery = countQuery.eq('section', session.section);
            }
            
            const { count } = await countQuery;
            studentCount = count || 0;
          } catch (error) {
            console.error('Error counting students:', error);
          }
          
          return {
            id: session.id,
            title: session.title,
            students: studentCount,
            date: session.date,
            time_in: session.time_in,
            time_out: session.time_out,
            type: session.type
          };
        })
      );
      
      setRecentSessions(sessionsWithCounts);
    } catch (error) {
      console.error('Error fetching recent sessions:', error);
    }
  }, []);

  // Fetch real-time statistics
  const fetchRealTimeStats = useCallback(async () => {
    try {
      setRealTimeStats({
        todayAttendance: 1410,
        todaySessions: 8,
        activeClasses: 24,
        pendingExcuses: 12
      });
    } catch (error) {
      console.error('Error fetching real-time stats:', error);
    }
  }, []);

  // Fetch current academic year
  const fetchAcademicYear = useCallback(async () => {
    try {
      setAcademicYear({
        year: '2025-2026',
        semester: 'First Semester',
        status: 'active'
      });
    } catch (error) {
      console.error('Error fetching academic year:', error);
    }
  }, []);

  useEffect(() => {
    loadUser();
    fetchTotalStudents();
    fetchRecentSessions();
    fetchRealTimeStats();
    fetchAcademicYear();
    
    // Initialize chart data
    setChartData(generateMockData(timePeriod));
  }, [user, timePeriod]);

  const loadUser = async () => {
    // If we have cached role for the same user, don't refetch
    if (cachedUserRole && cachedUserId === user?.id) {
      setUserRole(cachedUserRole);
      // Still fetch profile for other data, but don't wait for it
      fetchProfileData();
      return;
    }

    if (!user) {
      const defaultRole = 'user';
      setUserRole(defaultRole);
      cachedUserRole = defaultRole;
      cachedUserId = null;
      return;
    }
    
    try {
      // Use new helper to resolve role from admin/users
      const role = await fetchUserRole(user.id);
      setUserRole(role);
      cachedUserRole = role;
      cachedUserId = user.id;
      setCachedUserRole(role, user.id);
      // fetch profile-like data from admin/users for display
      await fetchProfileData();
    } catch (error) {
      console.error('Error resolving user role:', error);
      const defaultRole = 'user';
      setUserRole(defaultRole);
      if (user?.id) {
        cachedUserRole = defaultRole;
        cachedUserId = user.id;
        setCachedUserRole(defaultRole, user.id);
      }
    }
  };

  const fetchProfileData = async () => {
    if (!user) return;
    try {
      // Try admin first
      let profile: any = null;
      const { data: adminData } = await supabase
        .from('admin')
        .select('id, email, first_name, last_name, status, created_at, updated_at')
        .eq('id', user.id)
        .maybeSingle();
      if (adminData) profile = adminData;
      if (!profile) {
        const { data: checkerData } = await supabase
          .from('attendance_checker')
          .select('id, email, first_name, last_name, status, created_at, updated_at')
          .eq('id', user.id)
          .maybeSingle();
        if (checkerData) profile = checkerData;
      }
      setUserProfile(profile);
    } catch (error) {
      console.error('Error fetching account profile:', error);
    }
  };

  const fetchTotalStudents = async () => {
    try {
      const { count, error } = await supabase
        .from('students')
        .select('*', { count: 'exact', head: true });
      
      if (error) throw error;
      setTotalStudents(count || 0);
    } catch (error) {
      console.error('Error fetching total students:', error);
    } finally {
      setLoading(false);
    }
  };

  // Handle session click
  const handleSessionClick = (session: RecentSession) => {
    setSelectedSessionId(session.id);
    setIsStudentsModalOpen(true);
  };
  
  // Format time to 12-hour format
  const formatTime = (timeString: string) => {
    if (!timeString) return '--:--';
    const [hours, minutes] = timeString.split(':');
    const hour = parseInt(hours, 10);
    const period = hour >= 12 ? 'PM' : 'AM';
    const displayHour = hour % 12 || 12;
    return `${displayHour}:${minutes} ${period}`;
  };



  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return "Good morning";
    if (hour < 18) return "Good afternoon";
    return "Good evening";
  };

  const getDashboardTitle = () => {
    const roleLabels = {
      admin: 'ADMIN',
      'attendance checker': 'ATTENDANCE CHECKER'
    };
    return `${roleLabels[userRole as keyof typeof roleLabels] || 'User'} DASHBOARD`;
  };

  const getUserDisplayName = () => {
    // Don't show email while profile is still loading
    if (loading && !userProfile) {
      return '';
    }
    
    if (userProfile?.first_name && userProfile?.last_name) {
      return `${userProfile.first_name} ${userProfile.last_name}`;
    }
    if (userProfile?.first_name) {
      return userProfile.first_name;
    }
    if (user?.email) {
      return user.email.split('@')[0];
    }
    return 'User';
  };

  // Update chart data when time period changes
  useEffect(() => {
    setChartData(generateMockData(timePeriod));
  }, [timePeriod]);

  // Calculate percentages for the chart
  const chartDataWithPercentage = useMemo(() => {
    return chartData.map(item => ({
      ...item,
      presentPercentage: Math.round((item.present / (item.present + item.absent)) * 100),
      absentPercentage: Math.round((item.absent / (item.present + item.absent)) * 100)
    }));
  }, [chartData]);

  // No redirect needed for attendance checker

  return (
    <div className="flex-1 space-y-4 px-6 py-4 opacity-100 transition-opacity duration-300">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold tracking-tight text-gray-900">{getDashboardTitle()}</h2>
          <p className="text-sm text-gray-600">
            {getGreeting()}! Here's your attendance overview.
          </p>
        </div>
      </div>
      
      {/* Minimalist Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 px-6 pt-6">
            <CardTitle className="text-sm font-medium text-gray-700">Total Students</CardTitle>
            <div className="p-2 bg-blue-100 rounded-lg">
              <Users className="h-4 w-4 text-blue-700" />
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-6 pb-6">
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {loading ? '' : totalStudents.toLocaleString()}
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <BarChart3 className="h-3 w-3 mr-1" />
              +5.2% this month
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 px-6 pt-6">
            <CardTitle className="text-sm font-medium text-gray-700">Today's Attendance</CardTitle>
            <div className="p-2 bg-purple-100 rounded-lg">
              <UserCheck className="h-4 w-4 text-purple-700" />
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-6 pb-6">
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {realTimeStats.todayAttendance.toLocaleString()}
            </div>
            <div className="flex items-center text-sm text-gray-600">
              {yesterdayHigherAttendance ? (
                <TrendingUp className="h-3 w-3 mr-1" />
              ) : (
                <TrendingDown className="h-3 w-3 mr-1" />
              )}
              {mockYesterdayPresents.toLocaleString()} yesterday
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 px-6 pt-6">
            <CardTitle className="text-sm font-medium text-gray-700">Attendance Rate</CardTitle>
            <div className="p-2 bg-green-100 rounded-lg">
              <Activity className="h-4 w-4 text-green-700" />
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-6 pb-6">
            <div className="text-3xl font-bold text-gray-900 mb-1">94.2%</div>
            <div className="flex items-center text-sm text-gray-600">
              {yesterdayHigherRate ? (
                <TrendingUp className="h-3 w-3 mr-1" />
              ) : (
                <TrendingDown className="h-3 w-3 mr-1" />
              )}
              {mockYesterdayAttendanceRate}% yesterday
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 px-6 pt-6">
            <CardTitle className="text-sm font-medium text-gray-700">Today's Sessions</CardTitle>
            <div className="p-2 bg-orange-100 rounded-lg">
              <CalendarClock className="h-4 w-4 text-orange-700" />
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-6 pb-6">
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {realTimeStats.activeClasses}
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <CheckCircle className="h-3 w-3 mr-1" />
              {mockCompletedSessionsToday} completed
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-white border border-gray-200 shadow-sm">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 px-6 pt-6">
            <CardTitle className="text-sm font-medium text-gray-700">Academic Year</CardTitle>
            <div className="p-2 bg-indigo-100 rounded-lg">
              <CalendarDays className="h-4 w-4 text-indigo-700" />
            </div>
          </CardHeader>
          <CardContent className="pt-0 px-6 pb-6">
            <div className="text-2xl font-bold text-gray-900 mb-1">
              {academicYear?.year || '2025-2026'}
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <CalendarDays className="h-3 w-3 mr-1" />
              {academicYear?.semester || 'First Semester'}
            </div>
          </CardContent>
        </Card>
      </div>
      


      {/* Chart and Recent Sessions Section */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4 bg-white border border-gray-200 shadow-sm">
          <CardHeader className="pb-4">
            <div className="flex justify-between items-start">
              <div>
                <CardTitle className="text-lg font-semibold text-gray-900">Attendance Overview</CardTitle>
              </div>
              <Select value={timePeriod} onValueChange={(value: 'daily' | 'weekly' | 'monthly') => setTimePeriod(value)}>
                <SelectTrigger className="h-8 px-3 text-xs w-[120px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="daily">Daily</SelectItem>
                  <SelectItem value="weekly">Weekly</SelectItem>
                  <SelectItem value="monthly">Monthly</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent className="h-[300px] pl-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartDataWithPercentage}
                margin={{
                  top: 5,
                  right: 20,
                  left: 20,
                  bottom: 5,
                }}
                barGap={0}
                barCategoryGap="20%"
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                <YAxis 
                  axisLine={false}
                  tickLine={false}
                  tick={{ 
                    fill: '#6b7280',
                    textAnchor: 'end',
                    dx: -5
                  }}
                  tickFormatter={(value) => `${value}%`}
                  domain={[0, 100]}
                  width={40}
                  tickMargin={0}
                />
                <XAxis 
                  dataKey="name" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ 
                    fill: '#6b7280',
                    dy: 5
                  }}
                  padding={{ left: 15, right: 15 }}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'white',
                    borderRadius: '0.5rem',
                    border: '1px solid #e5e7eb',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    padding: '0.75rem',
                  }}
                  formatter={(value, name) => {
                    const index = chartDataWithPercentage.findIndex(item => 
                      item.presentPercentage === value || item.absentPercentage === value
                    );
                    const data = chartDataWithPercentage[index];
                    const total = data.present + data.absent;
                    const count = name === 'Present' ? data.present : data.absent;
                    return [`${value}% (${count}/${total})`, name];
                  }}
                  labelFormatter={(label) => `Period: ${label}`}
                />
                <Legend 
                  verticalAlign="top"
                  height={36}
                  formatter={(value) => (
                    <span className="text-sm text-muted-foreground">
                      {value} ({value === 'Present' ? '↑' : '↓'})
                    </span>
                  )}
                />
                <Bar 
                  dataKey="presentPercentage" 
                  name="Present"
                  fill="#10b981"
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  dataKey="absentPercentage" 
                  name="Absent"
                  fill="#ef4444"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Recent Attendance Sessions */}
        <Card className="col-span-3 bg-white border border-gray-200 shadow-sm">
          <CardHeader className="pb-3">
            <div className="flex justify-between items-center">
              <div>
                <CardTitle className="text-lg font-semibold text-gray-900">Recent Completed Sessions</CardTitle>
              </div>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => navigate('/schedule')}
                className="h-8 px-3 text-xs"
              >
                View All
              </Button>
            </div>
          </CardHeader>
          <CardContent className="h-[300px] overflow-y-auto overlay-scrollbar-container">
            <div className="space-y-2">
              {recentSessions.length > 0 ? (
                recentSessions.map((session) => (
                  <div 
                    key={session.id}
                    className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer border border-transparent hover:border-gray-200"
                    onClick={() => handleSessionClick(session)}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {session.title}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <p className="text-xs text-gray-500">
                          {session.students} students • {formatDate(new Date(session.date), 'MMM d, yyyy')} • {formatTime(session.time_in)} - {formatTime(session.time_out)}
                        </p>
                      </div>
                    </div>
                    <div className="ml-2">
                      <Badge className={
                        session.type === 'class' 
                          ? 'bg-primary/10 text-primary border-primary/20 text-xs' 
                          : session.type === 'event'
                          ? 'bg-accent/10 text-accent border-accent/20 text-xs'
                          : 'bg-education-navy/10 text-education-navy border-education-navy/20 text-xs'
                      }>
                        {session.type === 'class' ? 'Class' : session.type === 'event' ? 'Event' : 'Activity'}
                      </Badge>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8">
                  <div className="p-3 bg-gray-50 rounded-lg inline-block mb-3">
                    <CalendarDays className="h-6 w-6 text-gray-400" />
                  </div>
                  <p className="text-sm text-gray-500">No recent sessions</p>
                  <p className="text-xs text-gray-400 mt-1">Sessions will appear here</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* SessionStudents Modal */}
      <Dialog open={isStudentsModalOpen} onOpenChange={setIsStudentsModalOpen}>
        <DialogContent className="max-w-[96vw] w-[96vw] max-h-[90vh] p-4">
          {selectedSessionId && (
            <SessionStudents 
              sessionId={selectedSessionId}
              onClose={() => setIsStudentsModalOpen(false)}
              onSessionUpdated={() => {
                // Reload recent sessions
                fetchRecentSessions();
              }}
            />
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Dashboard;