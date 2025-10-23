# filepath: siamese_training/identification.py
"""
Signature Identification and Verification
Returns single result (owner) with confidence scores
"""

import numpy as np
from typing import Dict, Optional, Tuple
from preprocessing import preprocess_signature, is_likely_signature
from training import get_trainer

# Thresholds for decision making (STRICTER for better accuracy)
ACCEPT_THRESHOLD = 0.35      # Distance < 0.35 → ACCEPT (high confidence match)
REJECT_THRESHOLD = 0.7       # Distance > 0.7 → REJECT/UNKNOWN (too far)
NONSIG_THRESHOLD = 1.0       # Distance > 1.0 → NON_SIGNATURE (definitely not a signature)

# Confidence mapping (updated for stricter thresholds)
CONFIDENCE_HIGH = 0.9        # Distance < 0.2
CONFIDENCE_MEDIUM = 0.7      # Distance < 0.35
CONFIDENCE_LOW = 0.5         # Distance < 0.7


class SignatureIdentifier:
    """
    Signature Identification System
    Identifies the owner of a signature or returns "unknown"
    """
    
    def __init__(self):
        """Initialize the identifier"""
        self.trainer = get_trainer()
    
    def identify_signature(self, signature_image: str, is_base64: bool = True) -> Dict:
        """
        Identify the owner of a signature
        
        Args:
            signature_image: Base64 encoded signature image or numpy array
            is_base64: Whether input is base64 encoded
        
        Returns:
            Dict with identification result:
            {
                'identified': bool,
                'student_id': str or None,
                'confidence': float,
                'distance': float,
                'decision': str (ACCEPT, UNCERTAIN, UNKNOWN, NON_SIGNATURE, REJECT),
                'message': str,
                'threshold_info': dict
            }
        """
        # Check if model is trained
        if len(self.trainer.embeddings_db) == 0:
            return self._create_result(
                identified=False,
                student_id=None,
                confidence=0.0,
                distance=999.0,
                decision='UNKNOWN',
                message='❌ No students trained yet. Please train the model first.'
            )
        
        # Step 1: Check if it's likely a signature
        if not is_likely_signature(signature_image, is_base64):
            return self._create_result(
                identified=False,
                student_id=None,
                confidence=0.0,
                distance=999.0,
                decision='NON_SIGNATURE',
                message='🚫 This does not appear to be a signature. Please upload a clear signature image.'
            )
        
        # Step 2: Preprocess signature
        try:
            processed_signature = preprocess_signature(signature_image, is_base64, normalize=True)
        except Exception as e:
            return self._create_result(
                identified=False,
                student_id=None,
                confidence=0.0,
                distance=999.0,
                decision='UNKNOWN',
                message=f'❌ Error processing image: {str(e)}'
            )
        
        # Step 3: Get embedding for input signature
        query_embedding = self.trainer.model.get_embedding(processed_signature)
        
        # Step 4: Compare with all students in database
        best_match = self._find_best_match(query_embedding)
        
        if best_match is None:
            return self._create_result(
                identified=False,
                student_id=None,
                confidence=0.0,
                distance=999.0,
                decision='UNKNOWN',
                message='❌ No match found. Student may not be trained yet.'
            )
        
        student_id, distance, confidence = best_match
        
        # Step 5: Make decision based on distance
        decision, message = self._make_decision(student_id, distance, confidence)
        
        # Step 6: Return result
        return self._create_result(
            identified=(decision == 'ACCEPT'),
            student_id=student_id if decision == 'ACCEPT' else None,
            confidence=confidence,
            distance=distance,
            decision=decision,
            message=message
        )
    
    def _find_best_match(self, query_embedding: np.ndarray) -> Optional[Tuple[str, float, float]]:
        """
        Find the best matching student for the query embedding
        
        Args:
            query_embedding: Embedding of the query signature
        
        Returns:
            Tuple of (student_id, distance, confidence) or None
        """
        best_student = None
        best_distance = float('inf')
        
        # Compare with all students
        for student_id, stored_embeddings in self.trainer.embeddings_db.items():
            # Compute distance to all stored embeddings for this student
            distances = []
            
            for stored_emb in stored_embeddings:
                dist = np.linalg.norm(query_embedding - stored_emb)
                distances.append(dist)
            
            # Use minimum distance (closest match)
            min_distance = np.min(distances)
            
            # Update best match if this is closer
            if min_distance < best_distance:
                best_distance = min_distance
                best_student = student_id
        
        if best_student is None:
            return None
        
        # Convert distance to confidence score
        confidence = self._distance_to_confidence(best_distance)
        
        return (best_student, float(best_distance), float(confidence))
    
    def _distance_to_confidence(self, distance: float) -> float:
        """
        Convert Euclidean distance to confidence score (0-1)
        STRICTER mapping to reduce false positives
        
        Args:
            distance: Euclidean distance
        
        Returns:
            Confidence score (0-1)
        """
        # Map distance to confidence using exponential decay
        # Lower distance → higher confidence
        # STRICTER thresholds to avoid false matches
        
        if distance < 0.15:
            return 0.98
        elif distance < 0.2:
            return 0.95
        elif distance < 0.25:
            return 0.90
        elif distance < 0.3:
            return 0.85
        elif distance < 0.35:
            return 0.75  # ACCEPT threshold
        elif distance < 0.5:
            return 0.60
        elif distance < 0.7:
            return 0.40  # REJECT threshold
        elif distance < 0.9:
            return 0.20
        elif distance < 1.0:
            return 0.10  # NONSIG threshold
        else:
            return 0.02
    
    def _make_decision(self, student_id: str, distance: float, confidence: float) -> Tuple[str, str]:
        """
        Make final decision based on distance and confidence
        
        Args:
            student_id: Best matching student ID
            distance: Distance to best match
            confidence: Confidence score
        
        Returns:
            Tuple of (decision, message)
        """
        # Non-signature: distance too high
        if distance > NONSIG_THRESHOLD:
            return (
                'NON_SIGNATURE',
                f'🚫 Not a signature. Distance ({distance:.3f}) indicates this is likely a random photo or non-signature image.'
            )
        
        # Unknown: distance beyond acceptable threshold
        if distance > REJECT_THRESHOLD:
            return (
                'UNKNOWN',
                f'❌ Unknown student. No match found in database. Distance: {distance:.3f}. This signature does not belong to any trained student.'
            )
        
        # Uncertain: distance in gray area
        if distance > ACCEPT_THRESHOLD and distance <= REJECT_THRESHOLD:
            return (
                'UNCERTAIN',
                f'❓ Uncertain match to {student_id}. Confidence is low ({confidence*100:.1f}%). Distance: {distance:.3f}. Consider re-training with more samples.'
            )
        
        # Accept: strong match (ONLY if very close)
        if distance <= ACCEPT_THRESHOLD:
            if confidence >= 0.85:
                return (
                    'ACCEPT',
                    f'✅ High confidence match! This signature belongs to {student_id}. Confidence: {confidence*100:.1f}%, Distance: {distance:.3f}'
                )
            else:
                return (
                    'ACCEPT',
                    f'✅ Match found: {student_id}. Confidence: {confidence*100:.1f}%, Distance: {distance:.3f}'
                )
        
        # Fallback
        return (
            'UNKNOWN',
            f'⚠️  Unable to identify. Distance: {distance:.3f}'
        )
    
    def _create_result(self, identified: bool, student_id: Optional[str], 
                      confidence: float, distance: float, decision: str, message: str) -> Dict:
        """
        Create standardized result dictionary
        
        Args:
            identified: Whether student was identified
            student_id: Student ID (or None)
            confidence: Confidence score (0-1)
            distance: Euclidean distance
            decision: Decision type
            message: Human-readable message
        
        Returns:
            Result dictionary
        """
        return {
            'identified': identified,
            'student_id': student_id,
            'confidence': confidence,
            'distance': distance,
            'decision': decision,
            'message': message,
            'threshold_info': {
                'accept_threshold': ACCEPT_THRESHOLD,
                'reject_threshold': REJECT_THRESHOLD,
                'nonsig_threshold': NONSIG_THRESHOLD
            }
        }
    
    def verify_signature(self, signature_image: str, claimed_student_id: str, 
                        is_base64: bool = True) -> Dict:
        """
        Verify if a signature belongs to a specific student (1:1 verification)
        
        Args:
            signature_image: Base64 encoded signature
            claimed_student_id: Student ID to verify against
            is_base64: Whether input is base64 encoded
        
        Returns:
            Verification result dictionary
        """
        # Check if student exists in database
        if claimed_student_id not in self.trainer.embeddings_db:
            return {
                'verified': False,
                'student_id': claimed_student_id,
                'confidence': 0.0,
                'distance': 999.0,
                'message': f'❌ Student {claimed_student_id} not found in database. Please train first.'
            }
        
        # Preprocess signature
        try:
            processed_signature = preprocess_signature(signature_image, is_base64, normalize=True)
        except Exception as e:
            return {
                'verified': False,
                'student_id': claimed_student_id,
                'confidence': 0.0,
                'distance': 999.0,
                'message': f'❌ Error processing image: {str(e)}'
            }
        
        # Get embedding
        query_embedding = self.trainer.model.get_embedding(processed_signature)
        
        # Compare with claimed student's embeddings only
        stored_embeddings = self.trainer.embeddings_db[claimed_student_id]
        
        # Find minimum distance to any of their stored embeddings
        distances = []
        for stored_emb in stored_embeddings:
            dist = np.linalg.norm(query_embedding - stored_emb)
            distances.append(dist)
        
        min_distance = float(np.min(distances))
        confidence = self._distance_to_confidence(min_distance)
        
        # Verify
        verified = min_distance <= ACCEPT_THRESHOLD
        
        if verified:
            message = f'✅ Verified! This signature belongs to {claimed_student_id}. Confidence: {confidence*100:.1f}%'
        else:
            message = f'❌ Verification failed. This signature does NOT belong to {claimed_student_id}. Distance: {min_distance:.3f}'
        
        return {
            'verified': verified,
            'student_id': claimed_student_id,
            'confidence': confidence,
            'distance': min_distance,
            'message': message
        }
    
    def get_student_info(self, student_id: str) -> Optional[Dict]:
        """
        Get information about a trained student
        
        Args:
            student_id: Student ID
        
        Returns:
            Student info dictionary or None
        """
        if student_id not in self.trainer.embeddings_db:
            return None
        
        embeddings = self.trainer.embeddings_db[student_id]
        metadata = self.trainer.metadata.get('students', {}).get(student_id, {})
        
        return {
            'student_id': student_id,
            'embedding_count': len(embeddings),
            'metadata': metadata
        }
    
    def get_all_students(self) -> list:
        """Get list of all trained students"""
        return list(self.trainer.embeddings_db.keys())


# Global identifier instance
identifier = SignatureIdentifier()


def identify_signature(signature_image: str, is_base64: bool = True) -> Dict:
    """
    Main entry point for signature identification
    
    Args:
        signature_image: Base64 encoded signature or numpy array
        is_base64: Whether input is base64 encoded
    
    Returns:
        Identification result
    """
    return identifier.identify_signature(signature_image, is_base64)


def verify_signature(signature_image: str, student_id: str, is_base64: bool = True) -> Dict:
    """
    Main entry point for signature verification
    
    Args:
        signature_image: Base64 encoded signature
        student_id: Student ID to verify against
        is_base64: Whether input is base64 encoded
    
    Returns:
        Verification result
    """
    return identifier.verify_signature(signature_image, student_id, is_base64)


def get_identifier():
    """Get the global identifier instance"""
    return identifier
