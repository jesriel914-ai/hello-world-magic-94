import { format, startOfWeek, endOfWeek, addMinutes } from 'date-fns';

/**
 * Format a time string (HH:mm) to 12-hour format with AM/PM
 */
export const formatTime = (timeString: string): string => {
  if (!timeString) return '--:--';
  
  // Handle both 'HH:mm' and 'HH:mm:ss' formats
  const [hours, minutes] = timeString.split(':');
  const hour = parseInt(hours, 10);
  const mins = minutes?.substring(0, 2) || '00';
  
  const period = hour >= 12 ? 'PM' : 'AM';
  const displayHour = hour % 12 || 12; // Convert 0 to 12 for 12 AM
  
  return `${displayHour}:${mins.padStart(2, '0')} ${period}`;
};

/**
 * Format a date string to YYYY-MM-DD format
 */
export const formatDateString = (date: Date | string): string => {
  try {
    // If input is already a string in YYYY-MM-DD format, return as is
    if (typeof date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(date)) {
      return date;
    }
    
    // Create a new date object to avoid mutating the original
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    
    // Check if date is valid
    if (isNaN(dateObj.getTime())) {
      console.error('Invalid date provided:', date);
      return '';
    }
    
    return format(dateObj, 'yyyy-MM-dd');
  } catch (error) {
    console.error('Error formatting date:', error);
    return '';
  }
};

/**
 * Format a date for display (e.g., "January 1, 2024")
 */
export const formatDateDisplay = (date: Date | string): string => {
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    
    if (isNaN(dateObj.getTime())) {
      console.error('Invalid date provided:', date);
      return '';
    }
    
    return format(dateObj, 'MMMM d, yyyy');
  } catch (error) {
    console.error('Error formatting date for display:', error);
    return '';
  }
};

/**
 * Format a date and time for display (e.g., "January 1, 2024 at 2:30 PM")
 */
export const formatDateTimeDisplay = (date: Date | string): string => {
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    
    if (isNaN(dateObj.getTime())) {
      console.error('Invalid date provided:', date);
      return '';
    }
    
    return format(dateObj, 'MMMM d, yyyy \'at\' h:mm a');
  } catch (error) {
    console.error('Error formatting date time for display:', error);
    return '';
  }
};

/**
 * Parse a time string and add minutes to it
 */
export const addMinutesToTime = (timeString: string, minutesToAdd: number): string => {
  try {
    if (!timeString) return '';
    
    // Parse the time string
    const [hours, minutes] = timeString.split(':').map(Number);
    const baseDate = new Date();
    baseDate.setHours(hours, minutes, 0, 0);
    
    // Add minutes
    const newDate = addMinutes(baseDate, minutesToAdd);
    
    // Format back to HH:mm
    return format(newDate, 'HH:mm');
  } catch (error) {
    console.error('Error adding minutes to time:', error);
    return timeString;
  }
};

/**
 * Check if a time is within a range
 */
export const isTimeInRange = (timeString: string, startTime: string, endTime: string): boolean => {
  try {
    const [timeHours, timeMinutes] = timeString.split(':').map(Number);
    const [startHours, startMinutes] = startTime.split(':').map(Number);
    const [endHours, endMinutes] = endTime.split(':').map(Number);
    
    const timeTotal = timeHours * 60 + timeMinutes;
    const startTotal = startHours * 60 + startMinutes;
    const endTotal = endHours * 60 + endMinutes;
    
    return timeTotal >= startTotal && timeTotal <= endTotal;
  } catch (error) {
    console.error('Error checking time range:', error);
    return false;
  }
};

/**
 * Get the start and end of the week for a given date
 */
export const getWeekRange = (date: Date | string): { start: string; end: string } => {
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    
    if (isNaN(dateObj.getTime())) {
      console.error('Invalid date provided:', date);
      return { start: '', end: '' };
    }
    
    const weekStart = startOfWeek(dateObj, { weekStartsOn: 1 }); // Monday
    const weekEnd = endOfWeek(dateObj, { weekStartsOn: 1 }); // Sunday
    
    return {
      start: formatDateString(weekStart),
      end: formatDateString(weekEnd)
    };
  } catch (error) {
    console.error('Error getting week range:', error);
    return { start: '', end: '' };
  }
};

/**
 * Calculate the difference between two times in minutes
 */
export const getTimeDifferenceInMinutes = (startTime: string, endTime: string): number => {
  try {
    const [startHours, startMinutes] = startTime.split(':').map(Number);
    const [endHours, endMinutes] = endTime.split(':').map(Number);
    
    const startTotal = startHours * 60 + startMinutes;
    const endTotal = endHours * 60 + endMinutes;
    
    return endTotal - startTotal;
  } catch (error) {
    console.error('Error calculating time difference:', error);
    return 0;
  }
};
