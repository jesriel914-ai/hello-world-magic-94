import Navigation from "@/components/ui/navigation";
import Header from "@/components/Header";
import MobileBottomNav from "@/components/MobileBottomNav";
import { useMediaQuery } from "../hooks/use-media-query";
import { Skeleton } from "@/components/ui/skeleton";
import { useState, useEffect } from "react";
import { useSidebar } from "@/contexts/SidebarContext";
import { cn } from "@/lib/utils";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  const isDesktop = useMediaQuery("(min-width: 768px)");
  const { isCollapsed } = useSidebar();
  const [isLoading, setIsLoading] = useState(true);



  useEffect(() => {
    // Prevent content flickering during route transitions
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 50);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header Component - Hidden on mobile */}
      <div className="hidden lg:block">
        <Header isMobile={!isDesktop} />
      </div>
      
      {/* Navigation Component - Hidden on mobile */}
      <div className="hidden lg:block">
        <Navigation />
      </div>
      
      {/* Main Content */}
      <main className={cn(
        "min-w-0",
        "transition-[margin-left] duration-250 ease-in-out",
        isDesktop 
          ? (isCollapsed ? 'ml-12' : 'ml-64') 
          : 'ml-0 w-full', // No padding on mobile
        isDesktop ? "pt-2 pb-3 md:pb-4" : "pb-20" // Extra bottom padding for mobile nav
      )}>
        {isLoading ? (
          <div className="space-y-4 px-4">
            <div className="flex items-center justify-between">
              <Skeleton className="h-8 w-64" />
              <Skeleton className="h-9 w-32" />
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              <Skeleton className="h-64 w-full" />
              <Skeleton className="h-64 w-full" />
            </div>
          </div>
        ) : (
          <>{children}</>
        )}
      </main>
      
      {/* Mobile Bottom Navigation */}
      <MobileBottomNav />
    </div>
  );
};

export default Layout;