import { useNavigate, useLocation } from 'react-router-dom';
import { Camera, Calendar, Brain, User } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';

const MobileBottomNav = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  // Get user role from localStorage
  const userRole = localStorage.getItem('userRole') || '';

  // Define navigation items based on role
  const getNavItems = () => {
    const commonItems = [
      { path: '/take-attendance', icon: Camera, label: 'Attendance' },
      { path: '/profile', icon: User, label: 'Profile' },
    ];

    // ROTC Officer only sees Take Attendance
    if (userRole === 'ROTC officer') {
      return [
        { path: '/take-attendance', icon: Camera, label: 'Attendance' },
        { path: '/profile', icon: User, label: 'Profile' },
      ];
    }

    // Admin sees all
    if (userRole === 'admin') {
      return [
        { path: '/take-attendance', icon: Camera, label: 'Attendance' },
        { path: '/schedule', icon: Calendar, label: 'Sessions' },
        { path: '/model-training-signature-classify', icon: Brain, label: 'Training' },
        { path: '/profile', icon: User, label: 'Profile' },
      ];
    }

    // ROTC admin, Instructor, SSG officer see Take Attendance, Sessions, Profile
    return [
      { path: '/take-attendance', icon: Camera, label: 'Attendance' },
      { path: '/schedule', icon: Calendar, label: 'Sessions' },
      { path: '/profile', icon: User, label: 'Profile' },
    ];
  };

  const navItems = getNavItems();

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  return (
    <div className="lg:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 z-50">
      <div className="flex justify-around items-center h-16">
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.path);
          
          return (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              className={`flex flex-col items-center justify-center flex-1 h-full transition-colors ${
                active 
                  ? 'text-blue-600' 
                  : 'text-gray-600'
              }`}
            >
              <Icon className={`w-6 h-6 ${active ? 'text-blue-600' : 'text-gray-600'}`} />
              <span className={`text-xs mt-1 ${active ? 'font-semibold' : ''}`}>
                {item.label}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default MobileBottomNav;
