import { useState, useEffect } from 'react';

const useMobileDetection = () => {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      // Check if user agent indicates mobile device
      const userAgent = typeof navigator !== 'undefined' ? navigator.userAgent : '';
      const mobileRegex = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i;
      const tabletRegex = /iPad|Android(?!.*Mobile)|Tablet/i;
      
      // Check screen width (typical mobile breakpoint)
      const screenWidth = typeof window !== 'undefined' ? window.innerWidth : 0;
      const screenHeight = typeof window !== 'undefined' ? window.innerHeight : 0;
      
      // Check for touch capability (most mobile devices are touch-enabled)
      const isTouchDevice = typeof navigator !== 'undefined' && ('ontouchstart' in window || navigator.maxTouchPoints > 0);
      
      // Consider mobile if:
      // 1. User agent matches mobile pattern (not tablet)
      // 2. Screen width is mobile-sized AND it's a touch device
      // 3. Screen height is larger than width (typical mobile orientation)
      const isMobileDevice = mobileRegex.test(userAgent) || 
                             (screenWidth <= 768 && isTouchDevice) ||
                             (screenWidth <= 768 && screenHeight > screenWidth);
      
      // Debug logging
      console.log('📱 Mobile Detection Debug:', {
        userAgent,
        screenWidth,
        screenHeight,
        isTouchDevice,
        mobileRegexMatch: mobileRegex.test(userAgent),
        tabletRegexMatch: tabletRegex.test(userAgent),
        isMobileDevice
      });
      
      setIsMobile(isMobileDevice);
    };

    // Check on initial load
    checkMobile();

    // Add resize listener to handle orientation changes and window resizing
    if (typeof window !== 'undefined') {
      window.addEventListener('resize', checkMobile);
    }

    // Cleanup
    return () => {
      if (typeof window !== 'undefined') {
        window.removeEventListener('resize', checkMobile);
      }
    };
  }, []);

  return isMobile;
};

export default useMobileDetection;
