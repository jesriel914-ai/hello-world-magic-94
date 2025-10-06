"""
Signature Verification Script
Usage: python verify_signature.py --student_id <student_id> --reference <path> --test <path>
"""

import argparse
import os
from siamese_model import SiameseNetwork

def main():
    parser = argparse.ArgumentParser(description='Verify Signature using Siamese Network')
    parser.add_argument('--student_id', required=True, help='Student ID')
    parser.add_argument('--reference', required=True, help='Reference signature image path')
    parser.add_argument('--test', required=True, help='Test signature image path')
    parser.add_argument('--model_dir', default='./models', help='Directory containing trained models')
    
    args = parser.parse_args()
    
    print("Signature Verification")
    print("=" * 30)
    print(f"Student ID: {args.student_id}")
    print(f"Reference: {args.reference}")
    print(f"Test: {args.test}")
    print()
    
    # Check if files exist
    if not os.path.exists(args.reference):
        print(f"ERROR: Reference image {args.reference} not found!")
        return
    
    if not os.path.exists(args.test):
        print(f"ERROR: Test image {args.test} not found!")
        return
    
    # Load model
    model_path = os.path.join(args.model_dir, f"siamese_{args.student_id}.h5")
    metadata_path = os.path.join(args.model_dir, f"siamese_{args.student_id}_metadata.json")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model {model_path} not found!")
        print("Please train the model first using train_siamese.py")
        return
    
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata {metadata_path} not found!")
        return
    
    print("Loading trained model...")
    siamese = SiameseNetwork()
    siamese.load_model(model_path, metadata_path)
    
    print("Verifying signature...")
    is_verified, confidence = siamese.verify_signature(args.reference, args.test)
    
    print("\nResults:")
    print("-" * 20)
    print(f"Verified: {'YES' if is_verified else 'NO'}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Threshold: {siamese.threshold:.3f}")
    
    if is_verified:
        print("✅ Signatures MATCH - Same person")
    else:
        print("❌ Signatures DO NOT MATCH - Different person")

if __name__ == "__main__":
    main()