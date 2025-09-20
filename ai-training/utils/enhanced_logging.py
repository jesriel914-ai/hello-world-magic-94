"""
Enhanced Logging System for GPU Training
Provides structured logging with real-time progress updates
"""

import logging
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import os
from pathlib import Path

class TrainingLogger:
    """Enhanced logger for training jobs with real-time updates"""
    
    def __init__(self, job_id: str, log_dir: str = "/var/log/ai-training"):
        self.job_id = job_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self._setup_loggers()
        
        # Progress tracking
        self.current_progress = 0.0
        self.current_stage = "initializing"
        self.start_time = time.time()
        
    def _setup_loggers(self):
        """Setup different loggers for different purposes"""
        
        # Main training logger
        self.logger = logging.getLogger(f"training_{self.job_id}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"training_{self.job_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Progress logger (for real-time updates)
        self.progress_logger = logging.getLogger(f"progress_{self.job_id}")
        self.progress_logger.setLevel(logging.INFO)
        self.progress_logger.handlers.clear()
        
        progress_file = self.log_dir / f"progress_{self.job_id}.jsonl"
        progress_handler = logging.FileHandler(progress_file)
        progress_handler.setLevel(logging.INFO)
        self.progress_logger.addHandler(progress_handler)
        
        # Error logger
        self.error_logger = logging.getLogger(f"error_{self.job_id}")
        self.error_logger.setLevel(logging.ERROR)
        self.error_logger.handlers.clear()
        
        error_file = self.log_dir / f"errors_{self.job_id}.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message)
        self._log_progress("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message)
        self._log_progress("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message)
        self.error_logger.error(message)
        self._log_progress("error", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message)
        self._log_progress("debug", message, **kwargs)
    
    def update_progress(self, progress: float, stage: str, message: str = "", **kwargs):
        """Update training progress with real-time logging"""
        self.current_progress = progress
        self.current_stage = stage
        
        progress_message = f"[{progress:.1f}%] {stage}: {message}" if message else f"[{progress:.1f}%] {stage}"
        self.logger.info(progress_message)
        
        # Log structured progress data
        progress_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": self.job_id,
            "progress": progress,
            "stage": stage,
            "message": message,
            "elapsed_time": time.time() - self.start_time,
            **kwargs
        }
        
        self.progress_logger.info(json.dumps(progress_data))
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics"""
        self.logger.info(f"Training metrics: {json.dumps(metrics, indent=2)}")
        
        metrics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": self.job_id,
            "type": "metrics",
            "metrics": metrics,
            "elapsed_time": time.time() - self.start_time
        }
        
        self.progress_logger.info(json.dumps(metrics_data))
    
    def log_gpu_info(self):
        """Log GPU information"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            gpu_info = {
                "gpu_count": len(gpus),
                "gpu_devices": [str(gpu) for gpu in gpus],
                "cuda_available": tf.test.is_built_with_cuda(),
                "tensorflow_version": tf.__version__
            }
            
            self.logger.info(f"GPU Information: {json.dumps(gpu_info, indent=2)}")
            self._log_progress("gpu_info", "GPU information logged", **gpu_info)
            
        except Exception as e:
            self.warning(f"Failed to get GPU info: {e}")
    
    def log_system_info(self):
        """Log system information"""
        try:
            import psutil
            
            system_info = {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_total": psutil.disk_usage('/').total,
                "disk_free": psutil.disk_usage('/').free,
                "load_average": psutil.getloadavg()
            }
            
            self.logger.info(f"System Information: {json.dumps(system_info, indent=2)}")
            self._log_progress("system_info", "System information logged", **system_info)
            
        except Exception as e:
            self.warning(f"Failed to get system info: {e}")
    
    def _log_progress(self, level: str, message: str, **kwargs):
        """Internal method to log progress data"""
        progress_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "job_id": self.job_id,
            "level": level,
            "message": message,
            "current_progress": self.current_progress,
            "current_stage": self.current_stage,
            "elapsed_time": time.time() - self.start_time,
            **kwargs
        }
        
        self.progress_logger.info(json.dumps(progress_data))
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        return {
            "job_id": self.job_id,
            "current_progress": self.current_progress,
            "current_stage": self.current_stage,
            "elapsed_time": time.time() - self.start_time,
            "start_time": self.start_time,
            "status": "running" if self.current_progress < 100 else "completed"
        }
    
    def cleanup(self):
        """Cleanup loggers"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        for handler in self.progress_logger.handlers[:]:
            handler.close()
            self.progress_logger.removeHandler(handler)
        
        for handler in self.error_logger.handlers[:]:
            handler.close()
            self.error_logger.removeHandler(handler)

class ProgressTracker:
    """Real-time progress tracker for training jobs"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.progress_file = Path("/var/log/ai-training") / f"progress_{job_id}.jsonl"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_latest_progress(self) -> Optional[Dict[str, Any]]:
        """Get the latest progress update"""
        try:
            if not self.progress_file.exists():
                return None
            
            with open(self.progress_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return None
                
                # Get the last line
                last_line = lines[-1].strip()
                if last_line:
                    return json.loads(last_line)
        
        except Exception as e:
            print(f"Error reading progress: {e}")
        
        return None
    
    def get_all_progress(self) -> list:
        """Get all progress updates"""
        try:
            if not self.progress_file.exists():
                return []
            
            progress_updates = []
            with open(self.progress_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            progress_updates.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            return progress_updates
        
        except Exception as e:
            print(f"Error reading all progress: {e}")
            return []
    
    def stream_progress(self):
        """Stream progress updates in real-time"""
        try:
            if not self.progress_file.exists():
                yield {"error": "Progress file not found"}
                return
            
            with open(self.progress_file, 'r') as f:
                f.seek(0, 2)  # Go to end of file
                
                while True:
                    line = f.readline()
                    if line:
                        try:
                            yield json.loads(line.strip())
                        except json.JSONDecodeError:
                            continue
                    else:
                        time.sleep(0.1)  # Wait for new content
        
        except Exception as e:
            yield {"error": str(e)}

# Global logger instance
_training_loggers = {}

def get_training_logger(job_id: str) -> TrainingLogger:
    """Get or create a training logger for a job"""
    if job_id not in _training_loggers:
        _training_loggers[job_id] = TrainingLogger(job_id)
    
    return _training_loggers[job_id]

def cleanup_training_logger(job_id: str):
    """Cleanup a training logger"""
    if job_id in _training_loggers:
        _training_loggers[job_id].cleanup()
        del _training_loggers[job_id]