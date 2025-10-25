# filepath: siamese_training/classification.py
"""
Classification and Verification Logic
FIXED: Stricter thresholds, better non-signature detection
"""

import numpy as np
from typing import Dict, Tuple, Optional

class SignatureClassifier:
    """
    Handles signature identification and verification.
    FIXED: Much stricter thresholds to prevent false positives.
    """
    
    def __init__(self, model_wrapper, trainer):
        self.model = model_wrapper
        self.trainer = trainer
        
        # RELAXED THRESHOLDS - Signatures are naturally similar
        # Based on L2 normalized embeddings (distances typically 0-1 for signatures)
        self.ACCEPT_THRESHOLD = 0.35     # Distance <= 0.35: Accept (relaxed from 0.25)
        self.UNCERTAIN_THRESHOLD = 0.55  # 0.35 < d <= 0.55: Uncertain  
        self.REJECT_THRESHOLD = 0.75     # 0.55 < d <= 0.75: Unknown student (relaxed from 0.65)
        self.NONSIG_THRESHOLD = 1.0      # Distance > 1.0: Not a signature (relaxed from 0.85)
        
        # RELAXED confidence requirement
        self.MIN_CONFIDENCE = 0.60       # Minimum confidence to accept (reduced from 70%)
        
    def classify(self, image_base64: str) -> Dict:
        """
        Identify the owner of a signature (1:N identification).
        
        FIXED: Stricter decision logic with multi-sample verification
        """
        # Check if model is trained
        if not self.trainer.embeddings_db:
            return {
                'identified': False,
                'student_id': None,
                'confidence': 0.0,
                'distance': 999.0,
                'decision': 'REJECT',
                'message': 'Model not trained yet. Please train with student signatures first.',
                'threshold_info': self._get_threshold_info()
            }
        
        # Preprocess image
        try:
            image = self.trainer.preprocess_image(image_base64)
        except Exception as e:
            return {
                'identified': False,
                'student_id': None,
                'confidence': 0.0,
                'distance': 999.0,
                'decision': 'REJECT',
                'message': f'Failed to process image: {str(e)}',
                'threshold_info': self._get_threshold_info()
            }
        
        # Get embedding
        query_embedding = self.model.get_embedding(image)
        
        # MULTI-SAMPLE VERIFICATION: Compare with ALL samples, use statistics
        best_student = None
        best_min_distance = float('inf')
        best_avg_distance = float('inf')
        all_student_stats = []
        
        for student_id, embeddings_list in self.trainer.embeddings_db.items():
            embeddings = np.array(embeddings_list)
            distances = np.linalg.norm(embeddings - query_embedding, axis=1)
            
            min_dist = np.min(distances)
            avg_dist = np.mean(distances)
            median_dist = np.median(distances)
            
            all_student_stats.append({
                'student_id': student_id,
                'min_distance': min_dist,
                'avg_distance': avg_dist,
                'median_distance': median_dist,
                'num_samples': len(distances)
            })
            
            # Use AVERAGE of top 3 closest samples for more robust matching
            top_k = min(3, len(distances))
            top_k_avg = np.mean(np.sort(distances)[:top_k])
            
            if top_k_avg < best_avg_distance:
                best_avg_distance = top_k_avg
                best_min_distance = min_dist
                best_student = student_id
        
        # REMOVED MARGIN CHECK - Signatures are too similar across students
        # Using ONLY distance threshold is more reliable for signature recognition
        
        # Make decision using average distance (more robust than min)
        return self._make_decision(best_student, best_avg_distance, min_distance=best_min_distance)
    
    def verify(self, image_base64: str, claimed_student_id: str) -> Dict:
        """
        Verify if signature belongs to claimed student (1:1 verification).
        """
        # Check if student exists in database
        if claimed_student_id not in self.trainer.embeddings_db:
            return {
                'verified': False,
                'student_id': claimed_student_id,
                'distance': 999.0,
                'confidence': 0.0,
                'message': f'Student {claimed_student_id} not found in training database.'
            }
        
        # Preprocess image
        try:
            image = self.trainer.preprocess_image(image_base64)
        except Exception as e:
            return {
                'verified': False,
                'student_id': claimed_student_id,
                'distance': 999.0,
                'confidence': 0.0,
                'message': f'Failed to process image: {str(e)}'
            }
        
        # Get embedding
        query_embedding = self.model.get_embedding(image)
        
        # Compare with claimed student's embeddings (use top-3 average)
        embeddings = np.array(self.trainer.embeddings_db[claimed_student_id])
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        
        top_k = min(3, len(distances))
        avg_distance = np.mean(np.sort(distances)[:top_k])
        min_distance = np.min(distances)
        
        # Verify using stricter threshold
        verified = avg_distance <= self.ACCEPT_THRESHOLD
        confidence = self._distance_to_confidence(avg_distance)
        
        if verified and confidence >= self.MIN_CONFIDENCE:
            message = f'✅ Signature verified as {claimed_student_id} with {confidence*100:.1f}% confidence.'
        elif verified:
            message = f'⚠️ Low confidence match ({confidence*100:.1f}%). Distance: {avg_distance:.3f}'
            verified = False
        else:
            message = f'❌ Signature does not match {claimed_student_id}. Distance: {avg_distance:.3f}'
        
        return {
            'verified': verified,
            'student_id': claimed_student_id,
            'distance': float(avg_distance),
            'min_distance': float(min_distance),
            'confidence': float(confidence),
            'message': message
        }
    
    def _make_decision(self, student_id: Optional[str], distance: float, 
                       min_distance: float = None, force_uncertain: bool = False) -> Dict:
        """
        Make classification decision based on distance and thresholds.
        
        FIXED: Much stricter criteria with confidence requirements
        """
        if min_distance is None:
            min_distance = distance
        
        confidence = self._distance_to_confidence(distance)
        
        # FORCE UNCERTAIN if margin too small
        if force_uncertain and distance <= self.UNCERTAIN_THRESHOLD:
            decision = 'UNCERTAIN'
            identified = False
            message = (
                f'❓ Multiple students have similar signatures. '
                f'Best match: {student_id} (confidence: {confidence*100:.1f}%), '
                f'but margin to next match is too small. '
                'Cannot confidently identify owner.'
            )
            student_id = None
        
        # NON-SIGNATURE: Very high distance
        elif distance > self.NONSIG_THRESHOLD:
            decision = 'NON_SIGNATURE'
            identified = False
            message = (
                '🚫 This does not appear to be a signature. '
                'The system detected a random photo, blank page, or non-signature content. '
                f'Distance: {distance:.3f} (threshold: {self.NONSIG_THRESHOLD:.3f}). '
                'Please upload a clear image of a signature.'
            )
            student_id = None
        
        # UNKNOWN STUDENT: High distance
        elif distance > self.REJECT_THRESHOLD:
            decision = 'UNKNOWN'
            identified = False
            message = (
                f'❌ Signature does not match any trained student. '
                f'Best match was {student_id} with distance {distance:.3f}, '
                f'which exceeds the rejection threshold ({self.REJECT_THRESHOLD:.3f}). '
                'The student may not be enrolled yet.'
            )
            student_id = None
        
        # UNCERTAIN: Medium distance
        elif distance > self.UNCERTAIN_THRESHOLD:
            decision = 'UNCERTAIN'
            identified = False
            message = (
                f'❓ Uncertain match with {student_id} (confidence: {confidence*100:.1f}%). '
                f'Distance {distance:.3f} is between thresholds '
                f'({self.UNCERTAIN_THRESHOLD:.3f} - {self.REJECT_THRESHOLD:.3f}). '
                'This might indicate poor image quality, insufficient training samples, '
                'or the student is not enrolled.'
            )
            student_id = None
        
        # WEAK ACCEPT: Below accept threshold but low confidence
        elif distance <= self.ACCEPT_THRESHOLD and confidence < self.MIN_CONFIDENCE:
            decision = 'UNCERTAIN'
            identified = False
            message = (
                f'❓ Match found ({student_id}) but confidence is low ({confidence*100:.1f}%). '
                f'Distance: {distance:.3f}. '
                'Please upload a clearer signature or add more training samples.'
            )
            student_id = None
        
        # STRONG ACCEPT: Low distance + high confidence
        elif distance <= self.ACCEPT_THRESHOLD:
            decision = 'ACCEPT'
            identified = True
            message = (
                f'✅ Signature identified as {student_id} with {confidence*100:.1f}% confidence. '
                f'This is a strong match (distance: {distance:.3f}, min: {min_distance:.3f}).'
            )
        
        else:
            # Fallback
            decision = 'UNCERTAIN'
            identified = False
            message = f'Unable to classify signature. Distance: {distance:.3f}'
            student_id = None
        
        return {
            'identified': identified,
            'student_id': student_id,
            'confidence': float(confidence),
            'distance': float(distance),
            'min_distance': float(min_distance),
            'decision': decision,
            'message': message,
            'threshold_info': self._get_threshold_info()
        }
    
    def _distance_to_confidence(self, distance: float) -> float:
        """
        Convert distance to confidence score (0-1).
        
        FIXED: More aggressive decay for stricter confidence
        """
        k = 3.5  # Increased decay factor (was 2.0)
        confidence = np.exp(-k * distance)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _get_threshold_info(self) -> Dict:
        """Get threshold configuration."""
        return {
            'accept_threshold': self.ACCEPT_THRESHOLD,
            'uncertain_threshold': self.UNCERTAIN_THRESHOLD,
            'reject_threshold': self.REJECT_THRESHOLD,
            'nonsig_threshold': self.NONSIG_THRESHOLD,
            'min_confidence': self.MIN_CONFIDENCE
        }
    
    def get_all_students(self) -> Dict:
        """
        Get list of all trained students.
        
        Returns:
            Dict with student list and metadata
        """
        students = list(self.trainer.embeddings_db.keys())
        
        return {
            'students': students,
            'total': len(students),
            'last_updated': self.trainer.metadata.get('last_updated', 'Unknown')
        }