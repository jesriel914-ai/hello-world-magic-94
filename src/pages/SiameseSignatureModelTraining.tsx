//filepath: src\pages\SiameseSignatureModelTraining.tsx
import React, { useState } from 'react';
import TrainingSetup from '@/ai-model-siamese/components/TrainingSetup';
import Verification from '@/ai-model-siamese/components/Verification';
import IncrementalLearning from '@/ai-model-siamese/components/IncrementalLearning';
import useMobileDetection from '@/hooks/use-mobile-detection';

// Interface for shared state
interface ClassData {
  student: any | null;
  color: string;
  samples: any[];
  genuineSamples: any[];
  forgedSamples: any[];
}

const SiameseSignatureModelTraining: React.FC = () => {
  const isMobile = useMobileDetection();
  
  // Shared state only needed for TrainingSetup now
  const [classes, setClasses] = useState<ClassData[]>([
    { student: null, color: '#FF6B6B', samples: [], genuineSamples: [], forgedSamples: [] }
  ]);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">MODEL TRAINING</h1>
        </div>
        
        {isMobile ? (
          <div className="flex flex-col space-y-4">
            <TrainingSetup classes={classes} setClasses={setClasses} />
            <IncrementalLearning />
            <Verification />  {/* No props */}
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <div className="lg:col-span-2">
              <TrainingSetup classes={classes} setClasses={setClasses} />
            </div>
            <div className="lg:col-span-1">
              <IncrementalLearning />
            </div>
            <div className="lg:col-span-1">
              <Verification />  {/* No props */}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SiameseSignatureModelTraining;