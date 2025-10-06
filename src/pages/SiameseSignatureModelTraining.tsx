//filepath: src\pages\SiameseSignatureModelTraining.tsx
import React from 'react';
import TrainingSetup from '@/ai-model-siamese/components/TrainingSetup';
import Verification from '@/ai-model-siamese/components/Verification';
import useMobileDetection from '@/hooks/use-mobile-detection';

const SiameseSignatureModelTraining: React.FC = () => {
  const isMobile = useMobileDetection();

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Signature Model Training</h1>
        </div>
        
        {isMobile ? (
          <div className="flex flex-col space-y-4">
            <Verification />
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <TrainingSetup />
            <Verification />
          </div>
        )}
      </div>
    </div>
  );
};

export default SiameseSignatureModelTraining;