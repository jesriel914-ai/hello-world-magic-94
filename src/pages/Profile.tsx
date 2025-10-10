import Layout from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { User, LogOut, Mail, Shield } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { useNavigate } from "react-router-dom";
import { useState } from "react";

const Profile = () => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  
  // Get user role from localStorage
  const userRole = localStorage.getItem('userRole') || '';

  // Get user initials from email
  const getInitials = (email: string) => {
    return email
      .split("@")[0]
      .split(".")
      .map((n) => n[0])
      .join("")
      .toUpperCase();
  };

  const handleLogout = async () => {
    setIsLoggingOut(true);
    try {
      await signOut();
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
      setIsLoggingOut(false);
    }
  };

  if (!user) {
    return null;
  }

  return (
    <Layout>
      <div className="w-full space-y-6 lg:px-6 lg:py-4">
        {/* Header */}
        <div className="text-left px-6 lg:px-0 pt-6 lg:pt-0">
          <h1 className="text-3xl font-bold text-education-navy">Profile</h1>
        </div>

        {/* Profile Content */}
        <div className="px-6 lg:px-0">
          <Card className="max-w-2xl mx-auto lg:mx-0">
            <CardHeader>
              <CardTitle className="text-center lg:text-left">User Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Avatar Section */}
              <div className="flex flex-col items-center lg:flex-row lg:items-start gap-4">
                <Avatar className="h-24 w-24">
                  <AvatarImage src={user.user_metadata?.avatar_url} />
                  <AvatarFallback className="bg-primary/10 text-primary text-2xl">
                    {user.email ? getInitials(user.email) : <User className="h-8 w-8" />}
                  </AvatarFallback>
                </Avatar>
                
                <div className="flex-1 text-center lg:text-left space-y-2">
                  <h2 className="text-xl font-semibold text-gray-900">
                    {user.user_metadata?.name || user.email?.split('@')[0]}
                  </h2>
                  <div className="flex items-center justify-center lg:justify-start gap-2 text-sm text-gray-600">
                    <Mail className="w-4 h-4" />
                    {user.email}
                  </div>
                  <div className="flex items-center justify-center lg:justify-start gap-2 text-sm text-gray-600">
                    <Shield className="w-4 h-4" />
                    <span className="capitalize">{userRole || 'User'}</span>
                  </div>
                </div>
              </div>

              {/* Logout Button */}
              <div className="pt-4 border-t">
                <Button
                  onClick={handleLogout}
                  disabled={isLoggingOut}
                  variant="destructive"
                  className="w-full"
                >
                  {isLoggingOut ? (
                    <>
                      <LogOut className="w-4 h-4 mr-2 animate-spin" />
                      Logging out...
                    </>
                  ) : (
                    <>
                      <LogOut className="w-4 h-4 mr-2" />
                      Logout
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
};

export default Profile;
