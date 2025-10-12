// Cache manager for clearing caches across the application
// This allows different components to clear caches when data changes

export const clearAllSessionCaches = () => {
  console.log('CacheManager: Dispatching clearSessionCaches event');
  // We'll trigger custom events that cache holders will listen to
  window.dispatchEvent(new CustomEvent('clearSessionCaches'));
};

export const clearTakeAttendanceCache = () => {
  console.log('CacheManager: Dispatching clearTakeAttendanceCache event');
  window.dispatchEvent(new CustomEvent('clearTakeAttendanceCache'));
};
