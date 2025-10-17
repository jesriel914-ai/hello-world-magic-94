import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Camera, Play, StopCircle, CheckCircle, XCircle, Users, User, Clock, Calendar, BookOpen, ArrowLeft, RefreshCw, Square, FileImage, Brain, List, Cloud, X, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import Layout from "@/components/Layout";
import { useEffect, useRef, useState, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { supabase } from '@/lib/supabase';
import { toast } from 'sonner';
import { MobileWebcam } from '@/components/model-training-ui/services/mobileWebcam';
import useMobileDetection from '@/hooks/use-mobile-detection';
import * as tf from '@tensorflow/tfjs';
import { getAIModelService } from '@/lib/AIModelService';
import { predictFromCanvas, forceMemoryCleanup } from '@/components/model-training-ui/utils/modelPrediction';
import { fetchSessionStudents } from '@/lib/supabaseService';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { clearAllSessionCaches, clearTakeAttendanceCache } from '@/lib/cacheManager';

interface PredictionResult {
  className: string;
  confidence: number;
}

interface CustomModel {
  featureExtractor: tf.LayersModel | null;
  classifier: tf.LayersModel | null;
  getTotalClasses: () => number;
  getClassLabels: () => string[];
  predict: (image: HTMLCanvasElement | HTMLVideoElement, flipped?: boolean) => Promise<PredictionResult[]>;
}

type Session = {
  id: number;
  title: string;
  description: string;
  program: string;
  year: string;
  date: string;
  time_in: string;
  time_out: string;
  status?: 'not completed' | 'completed';
};

const TakeAttendanceSession = () => {
  const { sessionId, sessionTitle } = useParams<{ sessionId: string; sessionTitle?: string }>();
  const navigate = useNavigate();
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [attendanceStatus, setAttendanceStatus] = useState<'success' | 'error' | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isRequestingCamera, setIsRequestingCamera] = useState(false);
  const [cameraAvailable, setCameraAvailable] = useState<boolean | null>(null);
  const [stats, setStats] = useState({
    totalScanned: 0,
    matched: 0,
    noMatch: 0,
  });
  const [verificationResult, setVerificationResult] = useState<{
    match: boolean;
    student?: {
      id: number;
      student_id: string;
      firstname: string;
      surname: string;
    };
    score: number;
    message: string;
  } | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const permissionGranted = useRef(false);
  
  // Mobile webcam setup (like in Preview.tsx)
  const webcamRef = useRef<HTMLDivElement>(null);
  const mobileWebcam = useRef<MobileWebcam | null>(null);
  const [isCameraStarting, setIsCameraStarting] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const isMobile = useMobileDetection();
  
  // Model and prediction states
  const [model, setModel] = useState<CustomModel | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [modelTrainedAt, setModelTrainedAt] = useState<Date | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [showStudentList, setShowStudentList] = useState(false);
  const [sessionStudents, setSessionStudents] = useState<any[]>([]);
  const [loadingStudents, setLoadingStudents] = useState(false);
  const cameraPredictionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Attendance tracking
  const [attendanceLog, setAttendanceLog] = useState<any[]>([]);
  const [attendanceMap, setAttendanceMap] = useState<Map<number, any>>(new Map());
  const [overlayMessage, setOverlayMessage] = useState<string | null>(null);
  const [overlayStudentName, setOverlayStudentName] = useState<string | null>(null);
  const [overlayType, setOverlayType] = useState<'success' | 'error' | 'warning'>('success');
  const [isPaused, setIsPaused] = useState(false);
  const [showChangeConfirm, setShowChangeConfirm] = useState(false);
  const [pendingChange, setPendingChange] = useState<{student: any, newStatus: string} | null>(null);
  const [isMarkingCompleted, setIsMarkingCompleted] = useState(false);

  useEffect(() => {
    // Fetch session details when component mounts
    const fetchSession = async () => {
      try {
        setLoading(true);
        
        console.log('TakeAttendanceSession: URL params:', { sessionId, sessionTitle });
        
        const { data, error } = await supabase
          .from('sessions')
          .select('*, status')
          .eq('id', parseInt(sessionId))
          .single();

        if (error) throw error;
        
        // Check if session is completed - block access if it is
        if (data.status === 'completed') {
          toast.error('This session has been completed and is no longer accessible.');
          navigate('/take-attendance', { replace: true });
          return;
        }
        
        setSession(data);
        
        // Auto-load students and attendance when session is loaded
        if (data) {
          const response = await fetchSessionStudents(data.id);
          if (response?.students) {
            setSessionStudents(response.students);
          }
          
          // Load attendance records
          const { data: attendanceData } = await supabase
            .from('attendance')
            .select(`
              *,
              students (
                id,
                student_id,
                firstname,
                surname
              )
            `)
            .eq('session_id', data.id)
            .order('created_at', { ascending: false });
          
          const map = new Map();
          (attendanceData || []).forEach((record: any) => {
            map.set(record.student_id, record);
          });
          setAttendanceMap(map);
          setAttendanceLog(attendanceData || []);
        }
      } catch (err) {
        console.error('Error fetching session:', err);
        setError('Failed to load session details');
      } finally {
        setLoading(false);
      }
    };

    if (sessionId) {
      fetchSession();
    } else {
      setLoading(false);
    }

    // Clean up camera stream when component unmounts
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
        });
        streamRef.current = null;
      }
      if (mobileWebcam.current) {
        mobileWebcam.current.stop();
        mobileWebcam.current = null;
      }
      permissionGranted.current = false;
      setCameraActive(false);
    };
  }, [sessionId, navigate]);

  // Update page title when session is loaded
  useEffect(() => {
    if (session?.title) {
      document.title = `${session.title} - AMSUIP`;
    }
    
    // Cleanup on unmount
    return () => {
      document.title = 'Take Attendance - AMSUIP';
    };
  }, [session?.title]);

  // Helper function to format time to AM/PM
  const formatTimeToAMPM = (timeString: string) => {
    if (!timeString) return '';
    
    // Handle both 'HH:mm' and 'HH:mm:ss' formats
    const [hours, minutes] = timeString.split(':');
    const hour = parseInt(hours, 10);
    const mins = minutes || '00';
    
    const period = hour >= 12 ? 'PM' : 'AM';
    const displayHour = hour % 12 || 12; // Convert 0 to 12 for 12 AM
    
    return `${displayHour}:${mins} ${period}`;
  };

  // Handle marking session as completed
  const handleMarkCompleted = async () => {
    if (!session || isMarkingCompleted) return;

    try {
      setIsMarkingCompleted(true);

      // Get all students without attendance records
      const studentsToMarkAbsent = sessionStudents.filter(student => {
        const record = attendanceMap.get(student.id);
        return !record || !record.status || record.status === 'Not recorded';
      });

      // Mark all students without records as absent
      if (studentsToMarkAbsent.length > 0) {
        const absentRecords = studentsToMarkAbsent.map(student => ({
          session_id: session.id,
          student_id: student.id,
          status: 'absent'
        }));

        const { error: attendanceError } = await supabase
          .from('attendance')
          .upsert(absentRecords, {
            onConflict: 'session_id,student_id'
          });

        if (attendanceError) throw attendanceError;
      }

      // Update session status to completed
      const { error: sessionError } = await supabase
        .from('sessions')
        .update({ status: 'completed' })
        .eq('id', session.id);

      if (sessionError) throw sessionError;

      // Clear all caches
      console.log('TakeAttendanceSession: Clearing caches after marking completed...');
      clearAllSessionCaches();
      clearTakeAttendanceCache();

      toast.success('Session marked as completed successfully!');
      
      // Redirect to take attendance page
      setTimeout(() => {
        navigate('/take-attendance');
      }, 1000);

    } catch (error) {
      console.error('Error marking session as completed:', error);
      toast.error('Failed to mark session as completed');
    } finally {
      setIsMarkingCompleted(false);
    }
  };

  // Camera functions from Preview.tsx
  const startCamera = useCallback(async () => {
    console.log('📷 startCamera called - isMobile:', isMobile);
    if (!isMobile || !webcamRef.current) {
      console.log('❌ startCamera blocked');
      return;
    }
    
    console.log('🚀 Starting camera process...');
    setIsCameraStarting(true);
    setCameraError(null);
    setIsCameraReady(false);
    
    try {
      const permissions = await navigator.permissions.query({ name: 'camera' as PermissionName });
      console.log('📋 Camera permission status:', permissions.state);
      
      if (permissions.state === 'denied') {
        throw new Error('Camera permission denied. Please enable camera permissions in your browser settings.');
      }
      
      if (mobileWebcam.current) {
        mobileWebcam.current.stop();
        mobileWebcam.current = null;
      }
      
      webcamRef.current.innerHTML = '<div class="flex flex-col items-center justify-center h-full text-gray-500"><div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-2"></div><div class="text-sm">Starting camera...</div></div>';
      
      mobileWebcam.current = new MobileWebcam({
        width: 300,
        height: 300,
        facingMode: 'environment',
        timeout: 20000,
        zoom: 2.0
      });
      
      console.log('📷 Initializing camera with 2x zoom...');
      const videoElement = await mobileWebcam.current.start();
      
      webcamRef.current.innerHTML = '';
      webcamRef.current.appendChild(videoElement);
      
      setTimeout(() => {
        if (webcamRef.current && mobileWebcam.current) {
          const outerContainer = webcamRef.current.parentElement;
          if (outerContainer) {
            const rect = outerContainer.getBoundingClientRect();
            console.log('📱 Mobile preview container dimensions:', {
              width: rect.width,
              height: rect.height,
              aspectRatio: (rect.width / rect.height).toFixed(2)
            });
            
            mobileWebcam.current.setPreviewDimensions(rect.width, rect.height);
          }
        }
      }, 1000);
      
      setIsCameraReady(true);
      console.log('✅ Camera started successfully with zoom and marked as ready');
      
    } catch (error) {
      console.error('❌ Error starting camera:', error);
      
      let userErrorMessage = 'Failed to start camera';
      
      if (error instanceof Error) {
        if (error.message.includes('permission denied')) {
          userErrorMessage = 'Camera permission denied. Please allow camera access.';
        } else if (error.message.includes('timeout')) {
          userErrorMessage = 'Camera startup timed out. Please try again.';
        } else {
          userErrorMessage = error.message;
        }
      }
      
      setCameraError(userErrorMessage);
      setIsCameraReady(false);
      
      if (webcamRef.current) {
        webcamRef.current.innerHTML = `<div class="flex flex-col items-center justify-center h-full text-red-500 p-4 text-center"><div class="text-lg mb-2">❌ Camera Error</div><div class="text-sm">${userErrorMessage}</div></div>`;
      }
      
      if (mobileWebcam.current) {
        mobileWebcam.current.stop();
        mobileWebcam.current = null;
      }
    } finally {
      setIsCameraStarting(false);
    }
  }, [isMobile]);

  const stopCamera = useCallback(() => {
    console.log('🛑 Stopping camera...');
    
    if (mobileWebcam.current) {
      mobileWebcam.current.stop();
      mobileWebcam.current = null;
    }
    
    setIsCameraReady(false);
    
    if (webcamRef.current) {
      webcamRef.current.innerHTML = '';
    }
    
    console.log('✅ Camera stopped successfully');
  }, []);

  // Fetch students registered for this session
  const loadSessionStudents = useCallback(async () => {
    if (!session) return;
    
    try {
      setLoadingStudents(true);
      console.log('Fetching students for session:', session.id);
      
      const response = await fetchSessionStudents(session.id);
      
      if (!response || !response.students) {
        console.log('No students found in response');
        setSessionStudents([]);
        return;
      }
      
      console.log('Students loaded:', response.students.length);
      setSessionStudents(response.students);
      
      // Also load existing attendance records
      await loadAttendanceRecords();
    } catch (error) {
      console.error('Error fetching session students:', error);
      toast.error('Failed to load students');
      setSessionStudents([]);
    } finally {
      setLoadingStudents(false);
    }
  }, [session]);

  // Load attendance records for this session
  const loadAttendanceRecords = useCallback(async () => {
    if (!session) return;
    
    try {
      const { data, error } = await supabase
        .from('attendance')
        .select(`
          *,
          students (
            id,
            student_id,
            firstname,
            surname
          )
        `)
        .eq('session_id', session.id)
        .order('created_at', { ascending: false });
      
      if (error) throw error;
      
      // Create map for quick lookup
      const map = new Map();
      (data || []).forEach((record: any) => {
        map.set(record.student_id, record);
      });
      setAttendanceMap(map);
      setAttendanceLog(data || []);
    } catch (error) {
      console.error('Error loading attendance records:', error);
    }
  }, [session]);

  // Mark attendance function with optimistic UI update
  const markAttendance = async (status: 'present' | 'absent') => {
    console.log('🎯 Mark attendance clicked:', status);
    console.log('Predictions:', predictions);
    console.log('Session:', session);
    console.log('Session students:', sessionStudents);
    
    if (!predictions.length || !session) {
      console.log('❌ No predictions or session');
      toast.error('No student detected');
      return;
    }
    
    const predictedName = predictions[0].className;
    console.log('🔍 Looking for student:', predictedName);
    
    // Find student in session students list - try multiple formats
    const student = sessionStudents.find(s => {
      const format1 = `${s.student_id} - ${s.firstname} ${s.surname}`;
      const format2 = `${s.firstname} ${s.surname}`;
      const format3 = s.full_name;
      
      console.log('Checking:', { format1, format2, format3, predictedName });
      
      return format1 === predictedName || 
             format2 === predictedName || 
             format3 === predictedName;
    });
    
    console.log('Found student:', student);
    
    if (!student) {
      // Student not in required attendees
      console.log('⚠️ Student not in session');
      showOverlay(predictedName, 'Student not included in this session', 'warning');
      return;
    }
    
    // Check if already marked (using local state for instant check)
    const existingRecord = attendanceMap.get(student.id);
    console.log('Existing record:', existingRecord);
    
    if (existingRecord) {
      if (existingRecord.status === status) {
        // Same status - just show message
        const studentName = `${student.firstname} ${student.surname}`;
        console.log('⚠️ Already marked:', status);
        showOverlay(studentName, `Already marked ${status}`, 'warning');
        return;
      } else {
        // Different status - ask for confirmation
        console.log('🔄 Status change requested');
        setPendingChange({ student, newStatus: status });
        setShowChangeConfirm(true);
        return;
      }
    }
    
    // New record - optimistic UI update
    console.log('✅ Optimistic UI update for new attendance record');
    const studentName = `${student.firstname} ${student.surname}`;
    const overlayType = status === 'present' ? 'success' : 'error';
    
    // 1. Immediately update local state
    const newRecord = {
      session_id: session.id,
      student_id: student.id,
      status: status,
      time_in: status === 'present' ? new Date().toISOString() : null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
    
    setAttendanceMap(prev => new Map(prev).set(student.id, newRecord));
    setAttendanceLog(prev => [newRecord, ...prev]);
    
    // 2. Immediately show overlay
    showOverlay(studentName, `Marked ${status === 'present' ? 'Present' : 'Absent'}`, overlayType);
    
    // 3. Save to database in background (don't await)
    saveAttendanceToDatabase(student, status, newRecord);
  };

  // Save attendance to database (background operation)
  const saveAttendanceToDatabase = async (student: any, status: string, optimisticRecord: any) => {
    try {
      console.log('💾 Saving to database in background:', {
        session_id: session!.id,
        student_id: student.id,
        status: status
      });
      
      const { data, error } = await supabase
        .from('attendance')
        .upsert({
          session_id: session!.id,
          student_id: student.id,
          status: status,
          time_in: status === 'present' ? new Date().toISOString() : null,
          updated_at: new Date().toISOString()
        }, {
          onConflict: 'session_id,student_id'
        })
        .select();
      
      if (error) {
        console.error('❌ Database error:', error);
        
        // Revert optimistic update on error
        setAttendanceMap(prev => {
          const newMap = new Map(prev);
          newMap.delete(student.id);
          return newMap;
        });
        setAttendanceLog(prev => prev.filter(record => record.student_id !== student.id));
        
        toast.error('Failed to save attendance. Please try again.');
        throw error;
      }
      
      console.log('✅ Saved successfully to database:', data);
      
      // Update with actual database record
      if (data && data[0]) {
        setAttendanceMap(prev => new Map(prev).set(student.id, data[0]));
        setAttendanceLog(prev => {
          const filtered = prev.filter(record => record.student_id !== student.id);
          return [data[0], ...filtered];
        });
      }
    } catch (error) {
      console.error('❌ Error saving attendance:', error);
    }
  };
  
  // Save attendance with optimistic update (for confirmation dialog)
  const saveAttendance = async (student: any, status: string) => {
    const studentName = `${student.firstname} ${student.surname}`;
    const overlayType = status === 'present' ? 'success' : 'error';
    
    // 1. Immediately update local state
    const newRecord = {
      session_id: session!.id,
      student_id: student.id,
      status: status,
      time_in: status === 'present' ? new Date().toISOString() : null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
    
    setAttendanceMap(prev => new Map(prev).set(student.id, newRecord));
    setAttendanceLog(prev => {
      const filtered = prev.filter(record => record.student_id !== student.id);
      return [newRecord, ...filtered];
    });
    
    // 2. Immediately show overlay
    showOverlay(studentName, `Marked ${status === 'present' ? 'Present' : 'Absent'}`, overlayType);
    
    // 3. Save to database in background
    await saveAttendanceToDatabase(student, status, newRecord);
  };

  // Show overlay and pause camera
  const showOverlay = (studentName: string, message: string, type: 'success' | 'error' | 'warning') => {
    setOverlayStudentName(studentName);
    setOverlayMessage(message);
    setOverlayType(type);
    setIsPaused(true);
    
    // Hide overlay and resume after 1 second
    setTimeout(() => {
      setOverlayMessage(null);
      setOverlayStudentName(null);
      setIsPaused(false);
    }, 1000);
  };

  // Confirm status change
  const confirmStatusChange = async () => {
    if (!pendingChange) return;
    
    // Store the pending change before clearing
    const student = pendingChange.student;
    const newStatus = pendingChange.newStatus;
    
    // 1. Immediately close dialog and clear state
    setShowChangeConfirm(false);
    setPendingChange(null);
    
    // 2. Then update attendance in background
    await saveAttendance(student, newStatus);
  };

  // Auto-load latest model on page open
  useEffect(() => {
    const loadLatestModel = async () => {
      try {
        setIsLoadingModel(true);
        console.log('🤖 Auto-loading latest model...');
        
        const aiService = getAIModelService();
        const models = await aiService.getTrainedModels();
        
        if (!models || models.length === 0) {
          console.log('No models available');
          toast.error('No trained models available');
          return;
        }
        
        // Get the latest model (sorted by training_date descending)
        const latestModel = models.sort((a, b) => 
          new Date(b.training_date).getTime() - new Date(a.training_date).getTime()
        )[0];
        
        console.log('📥 Loading latest model:', latestModel.id);
        
        const loadResult = await aiService.loadModel(latestModel.id);
        
        if (!loadResult.success || !loadResult.model) {
          throw new Error(loadResult.error || 'Failed to load model');
        }
        
        console.log('✅ Model loaded successfully:', loadResult.model);
        
        const loadedModelData = loadResult.model;
        const customModel: CustomModel = {
          featureExtractor: null,
          classifier: null,
          getTotalClasses: () => loadedModelData.getTotalClasses(),
          getClassLabels: () => loadedModelData.getClassLabels(),
          predict: async (image: HTMLCanvasElement | HTMLVideoElement, flipped?: boolean) => {
            const results = await loadedModelData.predict(image, flipped);
            return results;
          }
        };
        
        setModel(customModel);
        setModelTrainedAt(new Date(latestModel.training_date));
        
        console.log('✅ Latest model loaded successfully');
      } catch (error) {
        console.error('❌ Error loading latest model:', error);
        toast.error('Failed to load model: ' + (error instanceof Error ? error.message : 'Unknown error'));
      } finally {
        setIsLoadingModel(false);
      }
    };
    
    loadLatestModel();
  }, []);

  // Camera prediction loop (only top 1 prediction) - Using improved version from master with requestAnimationFrame
  useEffect(() => {
    if (!isMobile || !isCameraReady || !mobileWebcam.current || !model) {
      setPredictions([]);
      console.log('🛑 Camera prediction loop stopped');
      return;
    }
    
    let isRunning = false;
    let consecutiveErrors = 0;
    const MAX_ERRORS = 3;
    let successCount = 0;
    let lastCleanupTime = Date.now();
    let lastPredictionTime = 0;
    const CLEANUP_INTERVAL = 5000;
    const MIN_PREDICTION_INTERVAL = 500; // ✅ Adjust this for your hardware (300-700ms)
    
    const runCameraPrediction = async () => {
      if (isRunning) return;
      if (!mobileWebcam.current || !model) return;
      
      isRunning = true;
      
      try {
        const now = Date.now();
        if (now - lastCleanupTime > CLEANUP_INTERVAL) {
          forceMemoryCleanup();
          lastCleanupTime = now;
        }
        
        const videoElement = mobileWebcam.current.getVideo();
        if (!videoElement) throw new Error('Video element is null');
        if (videoElement.paused || videoElement.ended) throw new Error('Video is paused or ended');
        if (videoElement.readyState < 2) throw new Error(`Video not ready (readyState: ${videoElement.readyState})`);
        
        const canvas = mobileWebcam.current.captureFrame();
        if (!canvas) throw new Error('captureFrame() returned null');
        if (canvas.width !== 224 || canvas.height !== 224) throw new Error(`Invalid canvas size: ${canvas.width}x${canvas.height}`);
        
        const predictions = await predictFromCanvas(model, canvas, false);
        if (!predictions || predictions.length === 0) throw new Error('Model returned empty predictions');
        
        consecutiveErrors = 0;
        successCount++;
        
        const sortedPredictions = predictions.sort((a, b) => b.confidence - a.confidence);
        
        if (successCount % 30 === 0) {
          const memory = tf.memory();
          console.log(`📊 Memory stats after ${successCount} predictions:`, {
            numTensors: memory.numTensors,
            numBytes: (memory.numBytes / 1024 / 1024).toFixed(2) + ' MB'
          });
        }
        
        setPredictions(sortedPredictions);
        
      } catch (error) {
        consecutiveErrors++;
        console.error(`❌ Camera prediction error #${consecutiveErrors}:`, error);
        
        if (consecutiveErrors >= MAX_ERRORS) {
          console.error('🛑 Too many consecutive errors, stopping predictions');
          toast({
            title: 'Camera Prediction Failed',
            description: 'Unable to process camera feed. Please try restarting the camera.',
            variant: 'destructive',
          });
          setPredictions([]);
        }
      } finally {
        isRunning = false;
      }
    };
    
    // ✅ Use requestAnimationFrame for smooth predictions
    let animationFrameId: number;
    
    const predictionLoop = async () => {
      const now = Date.now();
      
      // Throttle predictions to MIN_PREDICTION_INTERVAL
      if (now - lastPredictionTime >= MIN_PREDICTION_INTERVAL) {
        await runCameraPrediction();
        lastPredictionTime = now;
      }
      
      // Continue loop if camera is still active
      if (isCameraReady && model) {
        animationFrameId = requestAnimationFrame(predictionLoop);
      }
    };
    
    console.log('🚀 Starting camera prediction loop with requestAnimationFrame');
    predictionLoop();
    
    return () => {
      console.log('🛑 Stopping camera prediction loop');
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      console.log('🧹 Final memory cleanup');
      forceMemoryCleanup();
    };
  }, [isMobile, isCameraReady, model, isPaused, toast]);

  // Check for basic MediaDevices API support on component mount
  useEffect(() => {
    const checkBasicSupport = () => {
      // Define type for extended Navigator
      interface NavigatorExtended extends Navigator {
        webkitGetUserMedia?: typeof navigator.mediaDevices.getUserMedia;
        mozGetUserMedia?: typeof navigator.mediaDevices.getUserMedia;
        msGetUserMedia?: typeof navigator.mediaDevices.getUserMedia;
      }
      
      const nav = navigator as NavigatorExtended;
      const hasMediaDevices = !!(nav.mediaDevices && nav.mediaDevices.getUserMedia);
      const hasGetUserMedia = !!(nav.mediaDevices?.getUserMedia || 
                               nav.webkitGetUserMedia || 
                               nav.mozGetUserMedia ||
                               nav.msGetUserMedia);
      
      // Check if we're on a mobile device
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const isLocalNetwork = ['localhost', '127.0.0.1', '192.168.254.100'].includes(window.location.hostname);
      const isSecureContext = window.isSecureContext;
      
      console.log('Basic media devices check:', {
        hasMediaDevices,
        hasGetUserMedia,
        isSecureContext,
        isMobile,
        isLocalNetwork,
        location: window.location.href
      });
      
      // For mobile devices, we need HTTPS unless on localhost
      if (isMobile && !isSecureContext && !isLocalNetwork) {
        setCameraAvailable(false);
        return;
      }
      
      // Just check for basic API support, don't enumerate devices yet
      setCameraAvailable(hasMediaDevices || hasGetUserMedia);
    };
    
    checkBasicSupport();
  }, []);

  const handleStartCamera = async () => {
    console.log('Starting camera...');
    
    // Reset any previous error
    setError(null);
    
    if (cameraActive || !videoRef.current) {
      console.log('Camera already active or video ref not available');
      return;
    }
    
    setIsRequestingCamera(true);
    
    try {
      // First, check if we have permission to access media devices
      let devices: MediaDeviceInfo[] = [];
      try {
        devices = await navigator.mediaDevices.enumerateDevices();
        console.log('Available devices:', devices);
        
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        console.log('Video devices found:', videoDevices);
        
        if (videoDevices.length === 0) {
          throw new Error('No video input devices found');
        }
      } catch (err) {
        console.warn('Could not enumerate devices:', err);
        // Continue anyway, as some browsers might not support enumerateDevices
        // or might require getUserMedia first to get device labels
      }
      // Log environment info for debugging
      console.log('Environment info:', {
        userAgent: navigator.userAgent,
        isSecureContext: window.isSecureContext,
        location: window.location.href,
        mediaDevices: !!navigator.mediaDevices,
        getUserMedia: !!(navigator.mediaDevices?.getUserMedia)
      });
      
      // Check if we're on a mobile device and local network
      const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      const isLocalNetwork = ['localhost', '127.0.0.1', '192.168.254.100'].includes(window.location.hostname);
      const isSecureContext = window.isSecureContext;
      
      // Check HTTPS requirement for mobile devices
      if (isMobile && !isSecureContext && !isLocalNetwork) {
        throw new Error('Camera access requires HTTPS on mobile devices. Please use HTTPS or access from localhost.');
      }
      
      if (isMobile && isLocalNetwork) {
        console.log('Mobile device on local network detected, using legacy API if needed');
      }
      console.log('Requesting camera access...');
      
      // Define the type for getUserMedia function
      type GetUserMediaFunction = (constraints: MediaStreamConstraints) => Promise<MediaStream>;
      let getUserMedia: GetUserMediaFunction;

      // Define types for legacy browser support
      interface NavigatorWithLegacyGetUserMedia extends Navigator {
        webkitGetUserMedia?: (
          constraints: MediaStreamConstraints,
          success: (stream: MediaStream) => void,
          error: (error: Error) => void
        ) => void;
        mozGetUserMedia?: (
          constraints: MediaStreamConstraints,
          success: (stream: MediaStream) => void,
          error: (error: Error) => void
        ) => void;
      }

      const navWithLegacy = navigator as NavigatorWithLegacyGetUserMedia;
      let stream: MediaStream | null = null;

      // Function to try getting media with specific constraints
      const tryGetMedia = async (constraints: MediaStreamConstraints): Promise<MediaStream> => {
        console.log('Trying constraints:', JSON.stringify(constraints));
        
        try {
          if (navWithLegacy.mediaDevices?.getUserMedia) {
            // Standard way (modern browsers)
            return await navWithLegacy.mediaDevices.getUserMedia(constraints);
          } else if (navWithLegacy.webkitGetUserMedia) {
            // Old Chrome/WebKit way
            return await new Promise<MediaStream>((resolve, reject) => 
              navWithLegacy.webkitGetUserMedia!(
                constraints,
                (s) => resolve(s),
                (e) => reject(e)
              )
            );
          } else if (navWithLegacy.mozGetUserMedia) {
            // Old Firefox way
            return await new Promise<MediaStream>((resolve, reject) => 
              navWithLegacy.mozGetUserMedia!(
                constraints,
                (s) => resolve(s),
                (e) => reject(e)
              )
            );
          }
          throw new Error('No supported camera API found');
        } catch (err) {
          console.error('Error in tryGetMedia:', err);
          throw err;
        }
      };

      // Try different constraint sets in sequence, prioritizing rear camera
      const constraintSets: MediaStreamConstraints[] = [
        // First try with rear camera (environment) and high resolution
        { 
          video: { 
            facingMode: { exact: 'environment' },
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          },
          audio: false 
        },
        // Fallback to any rear camera with default resolution
        { 
          video: { 
            facingMode: { exact: 'environment' }
          },
          audio: false 
        },
        // If no rear camera is found, try any camera
        { 
          video: true,
          audio: false 
        }
      ];
      
      console.log('Trying constraint sets:', constraintSets);

      // Try each constraint set until one works
      for (const constraints of constraintSets) {
        try {
          stream = await tryGetMedia(constraints);
          if (stream) {
            console.log('Successfully obtained stream with constraints:', constraints);
            break;
          }
        } catch (err) {
          console.warn(`Failed with constraints ${JSON.stringify(constraints)}:`, err);
        }
      }
      
      if (!stream) {
        // If we have video devices but couldn't access any, show a more specific error
        const hasVideoDevices = devices.some(d => d.kind === 'videoinput');
        if (hasVideoDevices) {
          throw new Error('Could not access camera. Please check browser permissions and ensure no other app is using the camera.');
        } else {
          throw new Error('No camera found. Please ensure a camera is connected and not in use by another application.');
        }
      }
      console.log('Successfully got media stream');
      
      // Set the video element's source
      if (videoRef.current) {
        // Stop any existing stream first
        if (streamRef.current) {
          console.log('Stopping existing stream');
          streamRef.current.getTracks().forEach(track => track.stop());
        }
        
        // Set the new stream
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        
        // Wait for the video to be ready to play
        return new Promise<void>((resolve) => {
          if (!videoRef.current) {
            setError('Video element not available');
            return resolve();
          }
          
          const onCanPlay = () => {
            console.log('Video can play, starting...');
            videoRef.current?.play()
              .then(() => {
                console.log('Video playback started');
                setCameraActive(true);
                setCapturedImage(null);
                setAttendanceStatus(null);
                resolve();
              })
              .catch(playErr => {
                console.error('Error playing video:', playErr);
                setError('Could not start the camera. Please try again.');
                handleStopCamera();
                resolve();
              });
          };
          
          // Set up event listeners
          videoRef.current.oncanplay = onCanPlay;
          videoRef.current.onerror = (err) => {
            console.error('Video element error:', err);
            setError('Error initializing video stream');
            resolve();
          };
          
          // If the video is already ready, call onCanPlay directly
          if (videoRef.current.readyState >= 2) {
            onCanPlay();
          }
        });
      }
      
    } catch (err) {
      console.error('Camera error:', err);
      
      if (err instanceof Error) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
          setError('Camera access was denied. Please check your browser settings and allow camera access.');
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
          setError('No camera found on this device.');
        } else if (err.name === 'NotReadableError') {
          setError('Camera is already in use by another application.');
        } else if (err.name === 'TypeError' && err.message.includes('navigator.mediaDevices')) {
          setError('Your browser does not support camera access or is not in a secure context (try using HTTPS or localhost).');
        } else if (err.message.includes('HTTPS')) {
          setError(err.message);
        } else {
          setError(`Camera error: ${err.message}`);
        }
      } else {
        setError('Failed to access camera. Please check your browser settings and try again.');
      }
    } finally {
      setIsRequestingCamera(false);
    }
  };

  const handleStopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop();
      });
      streamRef.current = null;
    }
    setCameraActive(false);
    permissionGranted.current = false;
  };

  const handleCaptureSignature = async () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const imageUrl = canvas.toDataURL('image/png');
        setCapturedImage(imageUrl);
        
        // Stop the camera after capturing
        handleStopCamera();
        
        // Verify signature with AI
        await verifySignatureWithAI(imageUrl);
      }
    }
  };

  const verifySignatureWithAI = async (imageDataUrl: string) => {
    setIsVerifying(true);
    setVerificationResult(null);
    setAttendanceStatus(null);
    
    try {
      console.log('AI signature verification disabled - using placeholder');
      
      // Placeholder implementation since aiService.ts was removed
      // This simulates a failed verification
      const result = {
        success: false,
        match: false,
        predicted_student: null,
        score: 0,
        message: 'AI service temporarily disabled. Please use manual attendance marking.'
      };
      
      console.log('Placeholder verification result:', result);
      
      // Update stats
      setStats(prev => ({
        ...prev,
        totalScanned: prev.totalScanned + 1,
        matched: prev.matched + (result.match ? 1 : 0),
        noMatch: prev.noMatch + (!result.match ? 1 : 0),
      }));
      
      if (result.success) {
        setVerificationResult({
          match: result.match,
          student: result.predicted_student,
          score: result.score,
          message: result.message,
        });
        
        if (result.match && result.predicted_student) {
          setAttendanceStatus('success');
          toast.success(`Signature matched: ${result.predicted_student.firstname} ${result.predicted_student.surname} (${result.predicted_student.student_id})`);
        } else {
          setAttendanceStatus('error');
          toast.warning('Signature not recognized. Please try again or mark attendance manually.');
        }
      } else {
        setVerificationResult({
          match: false,
          score: 0,
          message: result.message || 'Verification failed',
        });
        setAttendanceStatus('error');
        toast.error(result.message || 'Signature verification failed');
      }
      
    } catch (error) {
      console.error('Error during signature verification:', error);
      setVerificationResult({
        match: false,
        score: 0,
        message: 'Verification error occurred',
      });
      setAttendanceStatus('error');
      toast.error('Failed to verify signature. Please try again.');
      
      // Update stats for error
      setStats(prev => ({
        ...prev,
        totalScanned: prev.totalScanned + 1,
      }));
    } finally {
      setIsVerifying(false);
    }
  };

  const handleRetakeSignature = () => {
    setCapturedImage(null);
    setVerificationResult(null);
    setAttendanceStatus(null);
    // Don't restart camera automatically, let user click to start
  };

  if (loading) {
    return (
      <Layout>
        <div className="p-8">
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary mx-auto mb-4"></div>
            <h2 className="text-xl font-medium text-education-navy">
              Loading session details...
            </h2>
          </div>
        </div>
      </Layout>
    );
  }

  if (error || !session) {
    return (
      <Layout>
        <div className="p-8">
          <div className="text-center py-12">
            <h2 className="text-2xl font-bold text-education-navy mb-4">
              {error || 'Session not found'}
            </h2>
            <Button 
              onClick={() => navigate('/take-attendance')}
              className="mt-4"
            >
              Back to Sessions
            </Button>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="w-full space-y-6 lg:px-6 lg:py-4">
        {/* Session Header - Left Aligned */}
        <div className="text-left">
          <h1 className="text-3xl font-bold text-education-navy">{session.title}</h1>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Section: Scan Signatures - Card on desktop only */}
          <div className="lg:bg-white lg:rounded-lg lg:border lg:border-gray-200 lg:shadow-sm lg:p-4">
            <div className="space-y-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-base font-semibold">
                  {showStudentList ? 'Details' : 'Scan Signatures'}
                </span>
                <Button 
                  variant="ghost" 
                  size="default"
                  onClick={() => {
                    const newValue = !showStudentList;
                    
                    // Stop camera when going to Details
                    if (newValue && isCameraReady) {
                      stopCamera();
                    }
                    
                    setShowStudentList(newValue);
                    
                    if (newValue && sessionStudents.length === 0) {
                      loadSessionStudents();
                    }
                  }}
                  className="h-10 w-10 p-0"
                  title={showStudentList ? "Back to Scanner" : "View Details"}
                >
                  {showStudentList ? <X className="w-5 h-5" /> : <List className="w-5 h-5" />}
                </Button>
              </div>
              
              {/* Camera On/Off Switch - Only in Scan Signatures view */}
              {!showStudentList && (
                <div className="flex items-center gap-2">
                  <Switch 
                    id="camera-switch"
                    checked={isCameraReady}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        startCamera();
                      } else {
                        stopCamera();
                      }
                    }}
                    disabled={isCameraStarting}
                  />
                  <Label htmlFor="camera-switch" className="text-sm font-medium cursor-pointer">
                    {isCameraStarting ? 'Starting...' : (isCameraReady ? 'On' : 'Off')}
                  </Label>
                </div>
              )}
            </div>
            
            {showStudentList ? (
              <div className="space-y-4 max-h-[500px] overflow-y-auto">
                {/* Session Details */}
                {session && (
                  <div>
                    <h3 className="text-sm font-semibold text-gray-700 mb-3">Session Information</h3>
                    <div className="p-4 bg-white rounded-lg border shadow-sm">
                      <div className="space-y-1.5">
                        <p className="text-sm">
                          <span className="text-gray-500">Title:</span> <span className="font-medium text-gray-900">{session.title}</span>
                        </p>
                        <p className="text-sm">
                          <span className="text-gray-500">Program:</span> <span className="font-medium text-gray-900">{session.program}</span>
                        </p>
                        <p className="text-sm">
                          <span className="text-gray-500">Year:</span> <span className="font-medium text-gray-900">{session.year}</span>
                        </p>
                        <p className="text-sm">
                          <span className="text-gray-500">Date:</span> <span className="font-medium text-gray-900">
                            {new Date(session.date).toLocaleDateString('en-US', {
                              month: 'short',
                              day: 'numeric',
                              year: 'numeric'
                            })}
                          </span>
                        </p>
                        <p className="text-sm">
                          <span className="text-gray-500">Time:</span> <span className="font-medium text-gray-900">
                            {session.time_in && formatTimeToAMPM(session.time_in)} - {session.time_out && formatTimeToAMPM(session.time_out)}
                          </span>
                        </p>
                        <p className="text-sm">
                          <span className="text-gray-500">Students:</span> <span className="font-medium text-gray-900">{sessionStudents.length} total</span>
                        </p>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Students List - Hidden on desktop (lg:hidden), shown on mobile only */}
                <div className="lg:hidden">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-gray-700">Students</h3>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-green-600 font-medium">
                        {sessionStudents.filter(s => attendanceMap.get(s.id)?.status === 'present').length} Present
                      </span>
                      <span className="text-xs text-red-600 font-medium">
                        {sessionStudents.filter(s => attendanceMap.get(s.id)?.status === 'absent').length} Absent
                      </span>
                    </div>
                  </div>
                  {sessionStudents.length > 0 ? (
                    <div className="space-y-1.5">
                      {[...sessionStudents].sort((a, b) => {
                        const aRecord = attendanceMap.get(a.id);
                        const bRecord = attendanceMap.get(b.id);
                        
                        // Sort by: 1) Has attendance record, 2) Most recent first
                        if (!aRecord && !bRecord) return 0;
                        if (!aRecord) return 1;
                        if (!bRecord) return -1;
                        
                        return new Date(bRecord.created_at || bRecord.updated_at || 0).getTime() - 
                               new Date(aRecord.created_at || aRecord.updated_at || 0).getTime();
                      }).map((student) => {
                        const attendanceRecord = attendanceMap.get(student.id);
                        return (
                          <div key={student.id} className="p-2 bg-white rounded-lg border shadow-sm">
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="text-sm font-medium text-gray-900">
                                  {student.full_name || `${student.firstname} ${student.surname}`}
                                </p>
                                <p className="text-xs text-gray-500">ID: {student.student_id}</p>
                              </div>
                              <div className="text-right">
                                {attendanceRecord ? (
                                  <>
                                    <span className={`text-xs font-medium ${
                                      attendanceRecord.status === 'present' 
                                        ? 'text-green-600' 
                                        : 'text-red-600'
                                    }`}>
                                      {attendanceRecord.status === 'present' ? 'Present' : 'Absent'}
                                    </span>
                                    {attendanceRecord.time_in && (
                                      <p className="text-xs text-gray-500 mt-1">
                                        {new Date(attendanceRecord.time_in).toLocaleTimeString('en-US', { 
                                          hour: 'numeric',
                                          minute: '2-digit',
                                          hour12: true 
                                        })}
                                      </p>
                                    )}
                                  </>
                                ) : (
                                  <span className="text-xs text-gray-400">Not marked</span>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <Users className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                      <p className="text-sm text-gray-400">No students registered for this session</p>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <>
            
            {/* Preview Box for Camera Feed */}
            <div className="relative border-2 border-dashed border-gray-300 rounded-lg aspect-video flex items-center justify-center bg-gray-50">
              <div 
                ref={webcamRef} 
                className="absolute inset-[2px] flex items-center justify-center z-0 rounded-lg overflow-hidden"
              />
              
              {cameraError && (
                <div className="absolute inset-[2px] flex items-center justify-center bg-red-50 z-20 rounded-lg">
                  <div className="text-center p-4">
                    <div className="text-red-600 font-medium mb-2">Camera Error</div>
                    <div className="text-red-500 text-sm">{cameraError}</div>
                  </div>
                </div>
              )}
              
              {!isCameraReady && !isCameraStarting && !cameraError && (
                <div className="text-gray-500 text-center">
                  <Camera className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                  Camera not active
                </div>
              )}
              
              {/* Model Status Overlay - Shows when camera is NOT active */}
              {!isCameraReady && !isCameraStarting && !cameraError && (
                <div className="absolute bottom-2 left-2 bg-black/70 text-white px-2 py-1 rounded text-xs font-medium z-30">
                  {isLoadingModel ? 'Not ready for attendance' : (model ? 'Ready for attendance' : 'No model loaded')}
                </div>
              )}
              
              {/* Prediction Overlay - Lower-left when camera is active */}
              {isCameraReady && predictions.length > 0 && !overlayMessage && (
                <>
                  {/* Lower-left: Student Name */}
                  <div className="absolute bottom-2 left-2 bg-black/70 text-white px-2 py-1 rounded text-xs font-medium z-30">
                    {predictions[0].className}
                  </div>
                  {/* Lower-right: Confidence */}
                  <div className="absolute bottom-2 right-2 bg-black/70 text-white px-2 py-1 rounded text-xs font-medium z-30">
                    {(predictions[0].confidence * 100).toFixed(0)}%
                  </div>
                </>
              )}
              
              {/* Status Overlay - Shows after marking */}
              {overlayMessage && (
                <div className={`absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-40 rounded-lg px-4 py-3 shadow-lg ${
                  overlayType === 'success' ? 'bg-green-600' : 
                  overlayType === 'error' ? 'bg-red-600' :
                  overlayType === 'warning' ? 'bg-yellow-600' : 
                  'bg-red-600'
                }`}>
                  <div className="text-center text-white">
                    <p className="font-bold text-sm">{overlayStudentName}</p>
                    <p className="text-xs mt-1">{overlayMessage}</p>
                  </div>
                </div>
              )}
            </div>
            
            {/* Attendance Buttons - Only show when camera is active */}
            {isCameraReady && (
              <div className="space-y-2">
                <Button 
                  onClick={(e) => {
                    e.currentTarget.blur();
                    markAttendance('present');
                  }}
                  disabled={!predictions.length || isPaused}
                  className="w-full h-12 text-sm bg-green-600 text-white focus-visible:ring-0 focus-visible:ring-offset-0 disabled:opacity-50 disabled:cursor-not-allowed pointer-events-auto"
                  style={{ backgroundColor: !predictions.length || isPaused ? undefined : '#16a34a' }}
                  onMouseDown={(e) => e.preventDefault()}
                >
                  Mark Present
                </Button>
                <Button 
                  onClick={(e) => {
                    e.currentTarget.blur();
                    markAttendance('absent');
                  }}
                  disabled={!predictions.length || isPaused}
                  className="w-full h-12 text-sm bg-red-600 text-white focus-visible:ring-0 focus-visible:ring-offset-0 disabled:opacity-50 disabled:cursor-not-allowed pointer-events-auto"
                  style={{ backgroundColor: !predictions.length || isPaused ? undefined : '#dc2626' }}
                  onMouseDown={(e) => e.preventDefault()}
                >
                  Mark Absent
                </Button>
              </div>
            )}
            
            {/* Mark Completed Button - Only show when camera is NOT active */}
            {!isCameraReady && session && session.status !== 'completed' && (
              <Button 
                className="w-full h-12 text-sm bg-blue-600 text-white focus-visible:ring-0 focus-visible:ring-offset-0"
                style={{ backgroundColor: '#2563eb' }}
                onMouseDown={(e) => e.preventDefault()}
                onClick={handleMarkCompleted}
                disabled={isMarkingCompleted}
              >
                {isMarkingCompleted ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    Marking...
                  </>
                ) : (
                  'Mark Completed'
                )}
              </Button>
            )}
            </>
            )}
            </div>
          </div>

          {/* Right Section: Students - Card on desktop only, hidden on mobile */}
          <div className="hidden lg:block lg:bg-white lg:rounded-lg lg:border lg:border-gray-200 lg:shadow-sm lg:p-4">
            <div className="space-y-4">
              {/* Header */}
              <div className="flex items-center justify-between">
                <h3 className="text-base font-semibold">Students</h3>
                <div className="flex items-center gap-3">
                  <span className="text-sm text-green-600 font-medium">
                    {sessionStudents.filter(s => attendanceMap.get(s.id)?.status === 'present').length} Present
                  </span>
                  <span className="text-sm text-red-600 font-medium">
                    {sessionStudents.filter(s => attendanceMap.get(s.id)?.status === 'absent').length} Absent
                  </span>
                </div>
              </div>
              
              {/* Content - All Students */}
              {sessionStudents.length > 0 ? (
                <div className="space-y-1.5 max-h-[400px] overflow-y-auto">
                  {[...sessionStudents].sort((a, b) => {
                    const aRecord = attendanceMap.get(a.id);
                    const bRecord = attendanceMap.get(b.id);
                    
                    // Sort by: 1) Has attendance record, 2) Most recent first
                    if (!aRecord && !bRecord) return 0;
                    if (!aRecord) return 1;
                    if (!bRecord) return -1;
                    
                    return new Date(bRecord.created_at || bRecord.updated_at || 0).getTime() - 
                           new Date(aRecord.created_at || aRecord.updated_at || 0).getTime();
                  }).map((student) => {
                    const attendanceRecord = attendanceMap.get(student.id);
                    return (
                      <div key={student.id} className="p-2 rounded-lg border border-gray-200">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium text-gray-900">
                              {student.full_name || `${student.firstname} ${student.surname}`}
                            </p>
                            <p className="text-xs text-gray-500">ID: {student.student_id}</p>
                          </div>
                          <div className="text-right">
                            {attendanceRecord ? (
                              <>
                                <span className={`text-xs font-medium ${
                                  attendanceRecord.status === 'present' 
                                    ? 'text-green-600' 
                                    : 'text-red-600'
                                }`}>
                                  {attendanceRecord.status === 'present' ? 'Present' : 'Absent'}
                                </span>
                                {attendanceRecord.time_in && (
                                  <p className="text-xs text-gray-500 mt-1">
                                    {new Date(attendanceRecord.time_in).toLocaleTimeString('en-US', { 
                                      hour: 'numeric',
                                      minute: '2-digit',
                                      hour12: true 
                                    })}
                                  </p>
                                )}
                              </>
                            ) : (
                              <span className="text-xs text-gray-400">Not marked</span>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-8">
                  <Users className="w-12 h-12 text-gray-300 mb-2" />
                  <p className="text-gray-500 text-sm">No students registered for this session</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
        
      {/* Status Change Confirmation Dialog */}
      <Dialog open={showChangeConfirm} onOpenChange={setShowChangeConfirm}>
          <DialogContent className="max-w-sm w-full">
            <DialogHeader>
              <DialogTitle>Confirm Status Change</DialogTitle>
            </DialogHeader>
            <p>
              <strong>{pendingChange?.student && `${pendingChange.student.firstname} ${pendingChange.student.surname}`}</strong> is already marked as{' '}
              <strong>{pendingChange?.student && attendanceMap.get(pendingChange.student.id)?.status}</strong>. 
              Do you want to change it to <strong>{pendingChange?.newStatus}</strong>?
            </p>
            <DialogFooter className="flex flex-col gap-2">
              <Button 
                onClick={confirmStatusChange}
                className="w-full h-12"
              >
                Yes
              </Button>
              <Button 
                variant="outline" 
                onClick={() => {
                  setShowChangeConfirm(false);
                  setPendingChange(null);
                }}
                className="w-full h-12"
              >
                No
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
    </Layout>
  );
};

export default TakeAttendanceSession;
