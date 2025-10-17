"""
google drive filepath: siamese_training/smart_orchestrator.py
Smart Training Orchestrator - FIXED for proper incremental learning
Automatically detects new vs existing students and handles classification DB correctly
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import shutil
import tempfile
import cv2
import numpy as np
import base64
import gc
import tensorflow as tf

class SmartTrainingOrchestrator:
    """
    FIXED: Automatically determines whether to:
    1. Train a new student from scratch
    2. Add samples incrementally to existing student
    3. Skip student (no new data)
    """
    
    def __init__(self, base_dir='models'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Track processed samples to avoid duplicates
        self.sample_tracking_file = self.base_dir / 'sample_tracking.json'
        self.sample_hashes = self._load_sample_tracking()
        
        print("🧠 Smart Training Orchestrator initialized (FIXED)")
    
    def _load_sample_tracking(self) -> Dict[str, List[str]]:
        """Load tracking of previously processed samples"""
        if self.sample_tracking_file.exists():
            try:
                with open(self.sample_tracking_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load sample tracking: {e}")
                return {}
        return {}
    
    def _save_sample_tracking(self):
        """Save sample tracking"""
        try:
            with open(self.sample_tracking_file, 'w') as f:
                json.dump(self.sample_hashes, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save sample tracking: {e}")
    
    def _hash_sample(self, sample_data: str) -> str:
        """Create hash of sample to detect duplicates"""
        # Handle base64 with data URI prefix
        if ',' in sample_data:
            sample_data = sample_data.split(',')[1]
        return hashlib.sha256(sample_data.encode()).hexdigest()
    
    def _is_student_trained(self, student_id: str) -> bool:
        """Check if student has a trained model"""
        student_dir = self.base_dir / student_id
        
        # Check for model files
        model_exists = (
            (student_dir / 'siamese_model.keras').exists() or
            (student_dir / 'feature_extractor.keras').exists()
        )
        
        # Check for reference embeddings
        embeddings_exist = (student_dir / 'reference_genuine.npy').exists()
        
        return model_exists and embeddings_exist
    
    def _get_new_samples(
        self, 
        student_id: str, 
        samples: List[str]
    ) -> List[str]:
        """Filter out samples that were already processed"""
        if student_id not in self.sample_hashes:
            # All samples are new
            return samples
        
        existing_hashes = set(self.sample_hashes[student_id])
        new_samples = []
        
        for sample in samples:
            sample_hash = self._hash_sample(sample)
            if sample_hash not in existing_hashes:
                new_samples.append(sample)
        
        return new_samples
    
    def _update_sample_tracking(
        self, 
        student_id: str, 
        samples: List[str]
    ):
        """Update tracking with newly processed samples"""
        if student_id not in self.sample_hashes:
            self.sample_hashes[student_id] = []
        
        for sample in samples:
            sample_hash = self._hash_sample(sample)
            if sample_hash not in self.sample_hashes[student_id]:
                self.sample_hashes[student_id].append(sample_hash)
        
        self._save_sample_tracking()
    
    def analyze_training_batch(
        self, 
        students: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Analyze a batch of students and categorize them
        
        Args:
            students: List of {
                'student_id': str,
                'genuine_samples': List[str],
                'forged_samples': List[str]
            }
        
        Returns:
            {
                'new_students': [...],      # Need full training
                'incremental': [...],        # Need incremental update
                'no_change': [...],          # No new samples
                'needs_retraining': [...]    # Too many updates, need full retrain
            }
        """
        result = {
            'new_students': [],
            'incremental': [],
            'no_change': [],
            'needs_retraining': []
        }
        
        for student_data in students:
            student_id = student_data['student_id']
            genuine_samples = student_data.get('genuine_samples', [])
            forged_samples = student_data.get('forged_samples', [])
            
            # Check if student exists
            is_trained = self._is_student_trained(student_id)
            
            if not is_trained:
                # New student - needs full training
                result['new_students'].append(student_data)
                print(f"🆕 {student_id}: New student (full training)")
            else:
                # Existing student - check for new samples
                new_genuine = self._get_new_samples(student_id, genuine_samples)
                new_forged = self._get_new_samples(student_id, forged_samples)
                
                if len(new_genuine) == 0 and len(new_forged) == 0:
                    # No new samples
                    result['no_change'].append(student_data)
                    print(f"⏭️  {student_id}: No new samples (skipping)")
                else:
                    # Has new samples - check if retraining needed
                    from siamese_incremental_trainer import SiameseIncrementalTrainer
                    
                    trainer = SiameseIncrementalTrainer(base_dir=self.base_dir)
                    check = trainer.check_if_retraining_needed(
                        student_id, 
                        len(new_genuine) + len(new_forged)
                    )
                    
                    if check['needs_retraining']:
                        # Too many updates - need full retraining
                        result['needs_retraining'].append({
                            **student_data,
                            'new_genuine': new_genuine,
                            'new_forged': new_forged,
                            'reason': check['reason']
                        })
                        print(f"🔄 {student_id}: Needs retraining ({check['reason']})")
                    else:
                        # Can do incremental update
                        result['incremental'].append({
                            **student_data,
                            'new_genuine': new_genuine,
                            'new_forged': new_forged
                        })
                        print(f"➕ {student_id}: Incremental update ({len(new_genuine)} genuine, {len(new_forged)} forged)")
        
        return result
    
    def execute_smart_training(
        self,
        students: List[Dict],
        on_progress=None
    ) -> Dict:
        """
        FIXED: Execute smart training with proper classification DB updates
        
        Args:
            students: List of student data
            on_progress: Optional callback(student_id, status, progress)
        
        Returns:
            {
                'total': int,
                'new_trained': int,
                'incremental_updated': int,
                'retrained': int,
                'skipped': int,
                'failed': int,
                'results': List[Dict]
            }
        """
        print("\n" + "="*60)
        print("SMART TRAINING ORCHESTRATOR (FIXED)")
        print("="*60)
        
        # Analyze batch
        analysis = self.analyze_training_batch(students)
        
        print(f"\n📊 Analysis:")
        print(f"   New students: {len(analysis['new_students'])}")
        print(f"   Incremental updates: {len(analysis['incremental'])}")
        print(f"   Needs retraining: {len(analysis['needs_retraining'])}")
        print(f"   No changes: {len(analysis['no_change'])}")
        print("="*60 + "\n")
        
        results = {
            'total': len(students),
            'new_trained': 0,
            'incremental_updated': 0,
            'retrained': 0,
            'skipped': len(analysis['no_change']),
            'failed': 0,
            'results': []
        }
        
        # Import trainers
        from siamese_trainer import SiameseSignatureTrainer
        from siamese_incremental_trainer import SiameseIncrementalTrainer
        
        trainer = SiameseSignatureTrainer(base_dir=self.base_dir)
        incremental_trainer = SiameseIncrementalTrainer(base_dir=self.base_dir)
        
        # Track students that need classifier update
        students_to_update_in_classifier = []
        
        # 1. Train new students
        for i, student_data in enumerate(analysis['new_students']):
            student_id = student_data['student_id']
            
            if on_progress:
                on_progress(student_id, 'training_new', i / len(students))
            
            try:
                print(f"\n🆕 Training new student: {student_id}")
                
                # Save samples temporarily
                genuine_paths = self._save_temp_samples(
                    student_id, 
                    student_data['genuine_samples'],
                    'genuine'
                )
                forged_paths = self._save_temp_samples(
                    student_id,
                    student_data['forged_samples'],
                    'forged'
                ) if student_data.get('forged_samples') else None
                
                # Full training
                metadata = trainer.train_student_model(
                    student_id=student_id,
                    genuine_samples=genuine_paths,
                    forged_samples=forged_paths,
                    epochs=50
                )
                
                trainer.save_reference_embeddings(student_id, genuine_paths)
                
                # Update tracking
                self._update_sample_tracking(
                    student_id,
                    student_data['genuine_samples'] + student_data.get('forged_samples', [])
                )
                
                # Mark for classifier update
                students_to_update_in_classifier.append(student_id)
                
                results['new_trained'] += 1
                results['results'].append({
                    'student_id': student_id,
                    'action': 'new_training',
                    'status': 'success',
                    'metadata': metadata
                })
                
                print(f"✅ {student_id}: Training completed")
                
                # Cleanup temp files
                for path in genuine_paths:
                    if os.path.exists(path):
                        os.remove(path)
                if forged_paths:
                    for path in forged_paths:
                        if os.path.exists(path):
                            os.remove(path)
                
            except Exception as e:
                print(f"❌ {student_id}: Training failed - {e}")
                results['failed'] += 1
                results['results'].append({
                    'student_id': student_id,
                    'action': 'new_training',
                    'status': 'failed',
                    'error': str(e)
                })
            finally:
                gc.collect()
                tf.keras.backend.clear_session()
        
        # 2. Incremental updates
        for i, student_data in enumerate(analysis['incremental']):
            student_id = student_data['student_id']
            
            if on_progress:
                on_progress(student_id, 'incremental', 
                           (len(analysis['new_students']) + i) / len(students))
            
            try:
                print(f"\n➕ Incremental update for: {student_id}")
                
                # Add new genuine samples
                if len(student_data['new_genuine']) > 0:
                    genuine_paths = self._save_temp_samples(
                        student_id,
                        student_data['new_genuine'],
                        'new_genuine'
                    )
                    
                    metadata = incremental_trainer.add_new_genuine_samples(
                        student_id=student_id,
                        new_genuine_samples=genuine_paths,
                        update_threshold=True
                    )
                    
                    self._update_sample_tracking(student_id, student_data['new_genuine'])
                    
                    # Cleanup
                    for path in genuine_paths:
                        if os.path.exists(path):
                            os.remove(path)
                
                # Add new forged samples
                if len(student_data.get('new_forged', [])) > 0:
                    forged_paths = self._save_temp_samples(
                        student_id,
                        student_data['new_forged'],
                        'new_forged'
                    )
                    
                    incremental_trainer.add_new_forged_samples(
                        student_id=student_id,
                        new_forged_samples=forged_paths
                    )
                    
                    self._update_sample_tracking(student_id, student_data['new_forged'])
                    
                    # Cleanup
                    for path in forged_paths:
                        if os.path.exists(path):
                            os.remove(path)
                
                # Mark for classifier update
                students_to_update_in_classifier.append(student_id)
                
                results['incremental_updated'] += 1
                results['results'].append({
                    'student_id': student_id,
                    'action': 'incremental',
                    'status': 'success'
                })
                
                print(f"✅ {student_id}: Incremental update completed")
                
            except Exception as e:
                print(f"❌ {student_id}: Incremental update failed - {e}")
                results['failed'] += 1
                results['results'].append({
                    'student_id': student_id,
                    'action': 'incremental',
                    'status': 'failed',
                    'error': str(e)
                })
            finally:
                gc.collect()
        
        # 3. Retrain students (if needed)
        for i, student_data in enumerate(analysis['needs_retraining']):
            student_id = student_data['student_id']
            
            if on_progress:
                on_progress(student_id, 'retraining',
                           (len(analysis['new_students']) + len(analysis['incremental']) + i) / len(students))
            
            try:
                print(f"\n🔄 Retraining: {student_id} (Reason: {student_data['reason']})")
                
                # Combine ALL samples (old + new)
                all_genuine = student_data['genuine_samples']
                all_forged = student_data.get('forged_samples', [])
                
                genuine_paths = self._save_temp_samples(student_id, all_genuine, 'genuine')
                forged_paths = self._save_temp_samples(student_id, all_forged, 'forged') if all_forged else None
                
                # Full retraining
                metadata = trainer.train_student_model(
                    student_id=student_id,
                    genuine_samples=genuine_paths,
                    forged_samples=forged_paths,
                    epochs=50
                )
                
                trainer.save_reference_embeddings(student_id, genuine_paths)
                
                # Update tracking
                self._update_sample_tracking(student_id, all_genuine + all_forged)
                
                # Reset incremental update counter in metadata
                metadata_file = self.base_dir / student_id / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        meta = json.load(f)
                    meta['incremental_updates'] = 0
                    meta['last_full_retrain'] = datetime.now().isoformat()
                    with open(metadata_file, 'w') as f:
                        json.dump(meta, f, indent=2)
                
                # Mark for classifier update
                students_to_update_in_classifier.append(student_id)
                
                results['retrained'] += 1
                results['results'].append({
                    'student_id': student_id,
                    'action': 'retrain',
                    'status': 'success',
                    'metadata': metadata
                })
                
                print(f"✅ {student_id}: Retraining completed")
                
                # Cleanup
                for path in genuine_paths:
                    if os.path.exists(path):
                        os.remove(path)
                if forged_paths:
                    for path in forged_paths:
                        if os.path.exists(path):
                            os.remove(path)
                
            except Exception as e:
                print(f"❌ {student_id}: Retraining failed - {e}")
                results['failed'] += 1
                results['results'].append({
                    'student_id': student_id,
                    'action': 'retrain',
                    'status': 'failed',
                    'error': str(e)
                })
            finally:
                gc.collect()
                tf.keras.backend.clear_session()
        
        # 4. CRITICAL FIX: Rebuild classification database if ANY student was updated
        if len(students_to_update_in_classifier) > 0:
            print(f"\n🔧 Rebuilding classification database for {len(students_to_update_in_classifier)} students...")
            print(f"   Students: {', '.join(students_to_update_in_classifier)}")
            
            try:
                from siamese_classifier import SiameseSignatureClassifier
                classifier = SiameseSignatureClassifier(base_dir=self.base_dir)
                
                # ALWAYS rebuild entire database to ensure consistency
                classifier.build_classification_database(rebuild=True)
                
                print("✅ Classification database rebuilt successfully")
                
            except Exception as e:
                print(f"⚠️  Failed to rebuild classification database: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("SMART TRAINING COMPLETE")
        print(f"  Total students: {results['total']}")
        print(f"  New trained: {results['new_trained']}")
        print(f"  Incremental: {results['incremental_updated']}")
        print(f"  Retrained: {results['retrained']}")
        print(f"  Skipped: {results['skipped']}")
        print(f"  Failed: {results['failed']}")
        print("="*60 + "\n")
        
        return results
    
    def _save_temp_samples(
        self,
        student_id: str,
        samples: List[str],
        prefix: str
    ) -> List[str]:
        """Save samples temporarily for training"""
        temp_dir = Path(tempfile.mkdtemp())
        paths = []
        
        for i, sample_base64 in enumerate(samples):
            try:
                # Decode base64
                if ',' in sample_base64:
                    sample_base64 = sample_base64.split(',')[1]
                
                img_bytes = base64.b64decode(sample_base64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"⚠️  Failed to decode sample {i} for {student_id}")
                    continue
                
                # Save
                path = temp_dir / f"{prefix}_{i}.jpg"
                cv2.imwrite(str(path), img)
                paths.append(str(path))
                
            except Exception as e:
                print(f"⚠️  Error saving sample {i} for {student_id}: {e}")
                continue
        
        return paths


# Standalone testing
if __name__ == "__main__":
    orchestrator = SmartTrainingOrchestrator(base_dir='models')
    
    # Example: Day 2 training
    day2_students = [
        {
            'student_id': '2025001',  # Existing student
            'genuine_samples': ['new_sig1', 'new_sig2'],
            'forged_samples': []
        },
        {
            'student_id': '2025010',  # New student
            'genuine_samples': ['sig1', 'sig2', 'sig3'],
            'forged_samples': ['forge1', 'forge2']
        }
    ]
    
    # Analyze
    analysis = orchestrator.analyze_training_batch(day2_students)
    print(json.dumps(analysis, indent=2))