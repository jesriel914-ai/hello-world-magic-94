import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuSeparator, 
  DropdownMenuTrigger 
} from "@/components/ui/dropdown-menu";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { 
  GraduationCap, 
  User, 
  LogOut, 
  Menu,
  Eye,
  EyeOff,
  Edit
} from "lucide-react";
import { UserCircle } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { useNavigate } from "react-router-dom";
import { useSidebar } from "@/contexts/SidebarContext";
import { supabase } from "@/lib/supabase";
import { fetchUserRole } from "@/lib/getUserRole";
import { cn } from "@/lib/utils";

// Extend window object for mobile drawer state
declare global {
  interface Window {
    mobileDrawerState?: {
      isOpen: boolean;
      toggle: () => void;
      close: () => void;
    };
  }
}

// Persistent role cache to prevent refetching and flashing on refresh
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

interface HeaderProps {
  isMobile?: boolean;
}

const Header = ({ isMobile = false }: HeaderProps) => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const { isCollapsed, toggleSidebar } = useSidebar();
  const [userRole, setUserRole] = useState<string>(() => {
    return cachedUserRole || 'user';
  });
  const [userProfile, setUserProfile] = useState<any>(null);
  const [academicYear, setAcademicYear] = useState<{
    year: string;
    semester: string;
  } | null>(null);
  const isInitialMount = useRef(true);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isLogoutConfirmOpen, setIsLogoutConfirmOpen] = useState(false);
  const [profileMode, setProfileMode] = useState<'display' | 'edit' | 'password'>('display');
  const [profileForm, setProfileForm] = useState<{ full_name: string; first_name: string; last_name: string; email: string }>({ full_name: '', first_name: '', last_name: '', email: '' });
  const [passwordForm, setPasswordForm] = useState<{ currentPassword: string; newPassword: string; confirmPassword: string }>({ currentPassword: '', newPassword: '', confirmPassword: '' });
  const [showPasswords, setShowPasswords] = useState<{ current: boolean; new: boolean; confirm: boolean }>({ current: false, new: false, confirm: false });
  const [isUpdatingProfile, setIsUpdatingProfile] = useState(false);
  const [isChangingPassword, setIsChangingPassword] = useState(false);

  useEffect(() => {
    const fetchRole = async () => {
      // If we have cached role for the same user, don't refetch
      if (cachedUserRole && cachedUserId === user?.id) {
        setUserRole(cachedUserRole);
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
        const role = await fetchUserRole(user.id);
        setUserRole(role);
        cachedUserRole = role;
        cachedUserId = user.id;
        setCachedUserRole(role, user.id);
        await fetchProfileData();
      } catch (error) {
        console.error('Error fetching user role:', error);
        const defaultRole = 'user';
        setUserRole(defaultRole);
        cachedUserRole = defaultRole;
        cachedUserId = user?.id || null;
        if (user?.id) {
          setCachedUserRole(defaultRole, user.id);
        }
      }
    };

    // Only fetch role on initial mount or when user changes
    if (isInitialMount.current || cachedUserId !== user?.id) {
      fetchRole();
      isInitialMount.current = false;
    }
  }, [user?.id]);

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

  // Fetch current academic year
  useEffect(() => {
    const fetchAcademicYear = async () => {
      try {
        // Fetch academic year data
        setAcademicYear({
          year: '2025-2026',
          semester: 'First Semester'
        });
      } catch (error) {
        console.error('Error fetching academic year:', error);
      }
    };
    fetchAcademicYear();
  }, []);

  const getPanelLabel = () => {
    if (userRole === 'admin') return 'Admin Panel';
    if (userRole === 'attendance checker') return 'Attendance Checker Panel';
    return 'User Panel';
  };

  const getUserDisplayName = () => {
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

  const handleLogout = async () => {
    try {
      await signOut();
      navigate('/login');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const handleProfileClick = () => {
    // Prefill form from loaded profile
    const fullName = userProfile?.first_name && userProfile?.last_name 
      ? `${userProfile.first_name} ${userProfile.last_name}`
      : userProfile?.first_name || userProfile?.last_name || '';
    
    setProfileForm({
      full_name: fullName,
      first_name: userProfile?.first_name || '',
      last_name: userProfile?.last_name || '',
      email: userProfile?.email || user?.email || ''
    });
    setProfileMode('display');
    setIsProfileOpen(true);
  };

  const handleProfileSave = async () => {
    try {
      if (!user) return;
      setIsUpdatingProfile(true);
      
      // Update either admin or users table depending on where the profile came from
      if (userRole === 'admin') {
        const { error } = await supabase
          .from('admin')
          .update({ first_name: profileForm.first_name, last_name: profileForm.last_name })
          .eq('id', user.id);
        if (error) throw error;
      } else {
        const { error } = await supabase
          .from('attendance_checker')
          .update({ first_name: profileForm.first_name, last_name: profileForm.last_name })
          .eq('id', user.id);
        if (error) throw error;
      }
      setUserProfile((prev: any) => ({ ...prev, first_name: profileForm.first_name, last_name: profileForm.last_name, email: profileForm.email }));
      setProfileMode('display');
    } catch (e) {
      console.error('Failed to save profile:', e);
    } finally {
      setIsUpdatingProfile(false);
    }
  };

  const handlePasswordChange = async () => {
    try {
      if (!user) return;
      
      // Validation
      if (!passwordForm.currentPassword.trim()) {
        console.error('Please enter your current password');
        return;
      }
      if (!passwordForm.newPassword.trim()) {
        console.error('Please enter a new password');
        return;
      }
      if (passwordForm.newPassword !== passwordForm.confirmPassword) {
        console.error('New passwords do not match');
        return;
      }
      if (passwordForm.newPassword.length < 6) {
        console.error('New password must be at least 6 characters long');
        return;
      }
      if (passwordForm.currentPassword === passwordForm.newPassword) {
        console.error('New password must be different from current password');
        return;
      }

      setIsChangingPassword(true);
      
      // First, verify the current password by attempting to sign in
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: user.email!,
        password: passwordForm.currentPassword
      });

      if (signInError) {
        console.error('Current password is incorrect');
        return;
      }

      // If current password is correct, update to new password
      const { error } = await supabase.auth.updateUser({
        password: passwordForm.newPassword
      });

      if (error) throw error;

      // Reset password form
      setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
      setProfileMode('display');
    } catch (error) {
      console.error('Error changing password:', error);
    } finally {
      setIsChangingPassword(false);
    }
  };

  const handleMobileMenuToggle = () => {
    if (window.mobileDrawerState) {
      window.mobileDrawerState.toggle();
    }
  };

  // Render mobile or desktop header
  return (
    <>
      {isMobile ? (
        /* Mobile Header */
        <header className="sticky top-0 z-50 md:hidden bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-sidebar-border h-14">
          <div className="flex items-center justify-between px-4 h-14">
            {/* Left: Menu Icon */}
            <Button variant="ghost" size="icon" onClick={handleMobileMenuToggle}>
              <Menu className="h-5 w-5" />
            </Button>
            
            {/* Right: Profile Icon with Dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-8 w-8 rounded-lg bg-gradient-primary p-0 flex items-center justify-center cursor-pointer transition-all duration-200 hover:scale-105">
                  <UserCircle className="h-5 w-5 text-primary-foreground" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56" align="end" forceMount>
                <div className="flex items-center justify-start gap-2 p-2">
                  <div className="flex flex-col space-y-1 leading-none">
                    <p className="font-medium">{getUserDisplayName()}</p>
                  </div>
                </div>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleProfileClick}>
                  <User className="mr-2 h-4 w-4" />
                  <span>Profile</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setIsLogoutConfirmOpen(true)}>
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>
      ) : (
        /* Desktop Header */
        <header className="sticky top-0 z-40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b border-sidebar-border h-14">
          <div className="flex items-center justify-between pr-6 pl-2 h-14">
            {/* Left side - Logo and toggle */}
            <div className="flex items-center gap-4">
              <div 
                className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center cursor-pointer transition-all duration-200 hover:scale-105"
                onClick={toggleSidebar}
              >
                <GraduationCap className="w-5 h-5 text-primary-foreground" />
              </div>
              <h1 className="text-lg font-bold text-education-navy">AMSUIP</h1>
            </div>

            {/* Right side - Academic first, then Panel label, then User dropdown */}
            <div className="flex items-center gap-4">
              {/* Academic Year and Semester */}
              {academicYear && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span>Current A.Y.:</span>
                  <span>{academicYear.year}</span>
                  <span>{academicYear.semester}</span>
                </div>
              )}

              {/* Vertical Separator */}
              {academicYear && (
                <div className="h-4 w-px bg-gray-300"></div>
              )}

              {/* Panel Label */}
              <div className="text-sm font-medium text-muted-foreground">
                {getPanelLabel()}
              </div>

              {/* User Dropdown */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" className="relative h-8 w-8 rounded-lg bg-gradient-primary p-0 flex items-center justify-center cursor-pointer transition-all duration-200 hover:scale-105">
                    <UserCircle className="h-5 w-5 text-primary-foreground" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56" align="end" forceMount>
                  <div className="flex items-center justify-start gap-2 p-2">
                    <div className="flex flex-col space-y-1 leading-none">
                      <p className="font-medium">{getUserDisplayName()}</p>
                    </div>
                  </div>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={handleProfileClick}>
                    <User className="mr-2 h-4 w-4" />
                    <span>Profile</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setIsLogoutConfirmOpen(true)}>
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Log out</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </header>
      )}
      
      {/* Shared Dialogs for Both Mobile and Desktop */}
      {/* Profile Dialog */}
      <Dialog open={isProfileOpen} onOpenChange={(open) => {
        setIsProfileOpen(open);
        if (!open) {
          // Reset form to original values when closing without saving
          const fullName = userProfile?.first_name && userProfile?.last_name 
            ? `${userProfile.first_name} ${userProfile.last_name}`
            : userProfile?.first_name || userProfile?.last_name || '';
          
          setProfileForm({
            full_name: fullName,
            first_name: userProfile?.first_name || '',
            last_name: userProfile?.last_name || '',
            email: userProfile?.email || user?.email || ''
          });
          setProfileMode('display');
          setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
          setShowPasswords({ current: false, new: false, confirm: false });
        }
      }}>
        <DialogContent 
          className="w-[95vw] max-w-[95vw] lg:max-w-[700px] lg:w-[700px] p-4"
          onOpenAutoFocus={(e) => e.preventDefault()}
        >
          <div className="w-full flex flex-col h-auto lg:h-[460px]">
            {/* Header */}
            <div className="pb-2 mb-3 flex-shrink-0">
              <h2 className="text-education-navy text-xl font-semibold">
                Profile Information
              </h2>
            </div>

            {/* Separator */}
            <div className="border-t border-gray-200 mb-4"></div>

            {/* Form Content */}
            <div className="flex-1 overflow-y-auto max-h-[60vh] lg:max-h-none">
              <div className="space-y-3">
                {/* Role */}
                <div className="flex flex-col space-y-1.5 lg:flex-row lg:items-center lg:space-y-0">
                  <Label className="text-sm text-left w-full lg:w-[200px]">Role:</Label>
                  <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center w-full lg:w-[350px] lg:ml-auto">
                    {getPanelLabel()}
                  </div>
                </div>

                {/* Email */}
                <div className="flex flex-col space-y-1.5 lg:flex-row lg:items-center lg:space-y-0">
                  <Label className="text-sm text-left w-full lg:w-[200px]">Email:</Label>
                  <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center w-full lg:w-[350px] lg:ml-auto">
                    {profileForm.email}
                  </div>
                </div>

                {/* Firstname */}
                <div className="flex flex-col space-y-1.5 lg:flex-row lg:items-center lg:space-y-0">
                  <Label htmlFor="firstname" className="text-sm text-left w-full lg:w-[200px]">Firstname:</Label>
                  <Input
                    id="firstname"
                    value={profileForm.first_name || ''}
                    onChange={(e) => {
                      setProfileForm(p => ({ ...p, first_name: e.target.value }));
                    }}
                    placeholder="Enter your first name"
                    className="h-9 text-sm bg-gray-100 w-full lg:w-[350px] lg:ml-auto"
                    autoFocus={false}
                  />
                </div>

                {/* Lastname */}
                <div className="flex flex-col space-y-1.5 lg:flex-row lg:items-center lg:space-y-0">
                  <Label htmlFor="lastname" className="text-sm text-left w-full lg:w-[200px]">Lastname:</Label>
                  <Input
                    id="lastname"
                    value={profileForm.last_name || ''}
                    onChange={(e) => {
                      setProfileForm(p => ({ ...p, last_name: e.target.value }));
                    }}
                    placeholder="Enter your last name"
                    className="h-9 text-sm bg-gray-100 w-full lg:w-[350px] lg:ml-auto"
                  />
                </div>

                {/* Password */}
                <div className="flex flex-col space-y-1.5 lg:flex-row lg:items-center lg:space-y-0">
                  <Label className="text-sm text-left w-full lg:w-[200px]">Password:</Label>
                  <div className="relative w-full lg:w-[350px] lg:ml-auto">
                    {profileMode === 'password' ? (
                      <>
                        <Input
                          type={showPasswords.current ? "text" : "password"}
                          value={passwordForm.currentPassword}
                          onChange={(e) => setPasswordForm(p => ({ ...p, currentPassword: e.target.value }))}
                          placeholder="Enter current password"
                          className="h-9 text-sm bg-gray-100 pr-10"
                        />
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="absolute right-0 top-0 h-9 w-9 p-0 hover:bg-transparent"
                          onClick={() => setShowPasswords(p => ({ ...p, current: !p.current }))}
                        >
                          {showPasswords.current ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                        </Button>
                      </>
                    ) : (
                      <>
                        <div className="h-9 px-3 py-2 text-sm bg-gray-100 rounded-md border border-input flex items-center pr-10">
                          ••••••••
                        </div>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setProfileMode('password');
                          }}
                          className="absolute right-1 top-1/2 -translate-y-1/2 h-6 w-6 p-0 transition-all duration-200 hover:scale-105 hover:bg-transparent"
                        >
                          <Edit className="h-3 w-3 text-yellow-600 transform hover:scale-125 transition-transform duration-200 ease-in-out" />
                        </Button>
                      </>
                    )}
                  </div>
                </div>

                {/* New Password - shown when edit clicked */}
                {profileMode === 'password' && (
                  <>
                    <div className="flex flex-col space-y-1.5 lg:flex-row lg:items-center lg:space-y-0">
                      <Label htmlFor="newPassword" className="text-sm text-left w-full lg:w-[200px]">New Password:</Label>
                      <div className="relative w-full lg:w-[350px] lg:ml-auto">
                        <Input
                          id="newPassword"
                          type={showPasswords.new ? "text" : "password"}
                          value={passwordForm.newPassword}
                          onChange={(e) => setPasswordForm(p => ({ ...p, newPassword: e.target.value }))}
                          placeholder="Enter new password"
                          className="h-9 text-sm bg-gray-100 pr-10 w-full"
                        />
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="absolute right-0 top-0 h-9 w-9 p-0 hover:bg-transparent"
                          onClick={() => setShowPasswords(p => ({ ...p, new: !p.new }))}
                        >
                          {showPasswords.new ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                        </Button>
                      </div>
                    </div>

                    <div className="flex flex-col space-y-1.5 lg:flex-row lg:items-center lg:space-y-0">
                      <Label htmlFor="confirmPassword" className="text-sm text-left w-full lg:w-[200px]">Confirm New Password:</Label>
                      <div className="relative w-full lg:w-[350px] lg:ml-auto">
                        <Input
                          id="confirmPassword"
                          type={showPasswords.confirm ? "text" : "password"}
                          value={passwordForm.confirmPassword}
                          onChange={(e) => setPasswordForm(p => ({ ...p, confirmPassword: e.target.value }))}
                          placeholder="Confirm new password"
                          className="h-9 text-sm bg-gray-100 pr-10 w-full"
                        />
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="absolute right-0 top-0 h-9 w-9 p-0 hover:bg-transparent"
                          onClick={() => setShowPasswords(p => ({ ...p, confirm: !p.confirm }))}
                        >
                          {showPasswords.confirm ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                        </Button>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Save Button */}
            <div className="pt-3 flex justify-end flex-shrink-0">
              <Button
                onClick={async () => {
                  if (profileMode === 'password' && passwordForm.newPassword) {
                    await handlePasswordChange();
                  } else {
                    await handleProfileSave();
                  }
                  setProfileMode('display');
                  setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' });
                }}
                disabled={
                  isUpdatingProfile || 
                  isChangingPassword || 
                  (profileForm.first_name === (userProfile?.first_name || '') && 
                   profileForm.last_name === (userProfile?.last_name || '') && 
                   !passwordForm.newPassword)
                }
                className="bg-education-blue hover:bg-education-blue/90"
              >
                {isUpdatingProfile || isChangingPassword ? 'Saving...' : 'Save'}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

        {/* Logout Confirmation Dialog - Shared with desktop */}
        <Dialog open={isLogoutConfirmOpen} onOpenChange={setIsLogoutConfirmOpen}>
          <DialogContent className="max-w-sm w-full">
            <DialogHeader>
              <DialogTitle>Confirm Logout</DialogTitle>
            </DialogHeader>
            <p>Are you sure you want to log out?</p>
            <DialogFooter className="flex flex-row gap-2 sm:gap-3">
              <Button variant="outline" onClick={() => setIsLogoutConfirmOpen(false)} className="flex-1">Cancel</Button>
              <Button variant="destructive" onClick={handleLogout} className="flex-1">Log Out</Button>
            </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default Header;
