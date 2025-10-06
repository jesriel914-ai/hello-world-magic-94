//filepath: src\pages\SiameseSignatureModelTraining.tsx
import React from 'react';
import SiameseModelTraining from '@/ai-model-siamese/components/ModelTraining';

const SiameseSignatureModelTraining: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Siamese Signature Model Training</h1>
          <p className="text-gray-600">
            Train and verify signatures using Siamese networks for enhanced accuracy and forgery detection.
          </p>
        </div>
        
        <SiameseModelTraining />
      </div>
    </div>
  );
};

export default SiameseSignatureModelTraining;