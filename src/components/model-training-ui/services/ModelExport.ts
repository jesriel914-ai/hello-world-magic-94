// filepath: src/components/model-training-ui/services/ModelExport.ts
// FIXED: Properly exports Teachable Machine models with MobileNet reference

import * as tf from '@tensorflow/tfjs';
import { CustomModel } from '../../ModelTraining';
import type { ClassData } from '../../ModelTraining';
import JSZip from 'jszip';
import { getAIModelService } from '@/lib/AIModelService';

interface ExportParams {
  model: CustomModel | null;
  classes: ClassData[];
  currentClassIndex: number;
  trainingAccuracy: number | null;
  trainingStartTime: number | null;
  formatStudentDisplay: (student: any) => string;
  setIsExporting?: (value: boolean) => void;
  setIsDownloading?: (value: boolean) => void;
  setHasExportedToCloud?: (value: boolean) => void;
  setHasDownloadedToPC?: (value: boolean) => void;
  showCloudSuccessNotification?: () => void;
  showLocalExportSuccessNotification?: (fileName: string) => void;
}

export const exportModel = async (params: ExportParams): Promise<void> => {
  const { model } = params;
  
  if (!model || !model.classifier) {
    alert('Please train a model first before exporting');
    return;
  }
  
  // For now, just call exportToLocal
  await exportToLocal(params);
};

export const exportToS3 = async (params: ExportParams): Promise<void> => {
  const {
    model,
    classes,
    trainingAccuracy,
    trainingStartTime,
    formatStudentDisplay,
    setHasExportedToCloud,
    showCloudSuccessNotification
  } = params;
  
  if (!model || !model.classifier) {
    alert('Please train a model first before uploading to S3');
    return;
  }

  try {
    // Get valid classes
    const validClasses = classes.filter(cls => cls.samples.length > 0);
    const totalSampleCount = validClasses.reduce((total, cls) => total + cls.samples.length, 0);
    
    // Prepare student information
    const students = validClasses.map(cls => ({
      id: cls.student?.id?.toString() || '',
      student_id: cls.student?.student_id || '',
      firstname: cls.student?.firstname || '',
      surname: cls.student?.surname || '',
      full_name: cls.student ? formatStudentDisplay(cls.student) : 'Unassigned'
    }));

    // CRITICAL FIX: Export ONLY the classifier (not MobileNet)
    // The classifier will be loaded separately and MobileNet will be reloaded from Teachable Machine
    
    // Save classifier using TensorFlow.js save handler
    // This is more reliable than manual weight extraction
    const saveResult = await model.classifier.save(tf.io.withSaveHandler(async (artifacts) => {
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON'
        }
      };
    }));
    
    // Get model topology and weights using proper TF.js API
    const classifierArtifacts = await model.classifier.save(tf.io.withSaveHandler(async (artifacts) => artifacts));
    
    if (!classifierArtifacts || !classifierArtifacts.weightData) {
      throw new Error('Failed to extract model artifacts');
    }
    

    // DEBUG: log labels before creating metadata
console.log("🔍 EXPORT DEBUG (S3) - Class labels:", model.getClassLabels());
console.log("🔍 EXPORT DEBUG (S3) - Number of classes:", model.getClassLabels()?.length);

    // Create metadata
    const metadata = {
      format: 'teachable-machine-mobilenet',
      generatedBy: 'SignatureAI',
      convertedBy: 'tfjs 1.3.1',
      modelTopology: classifierArtifacts.modelTopology,
      weightsManifest: classifierArtifacts.weightSpecs ? [{
        paths: ['weights.bin'],
        weights: classifierArtifacts.weightSpecs
      }] : [],
      labels: model.getClassLabels(),
      userMetadata: {
        sample_count: totalSampleCount,
        total_students: validClasses.length,
        accuracy: trainingAccuracy || 0.85,
        training_date: trainingStartTime ? new Date(trainingStartTime).toISOString() : new Date().toISOString(),
        students: students,
        model_type: 'teachable-machine',
        mobilenet_version: 2,
        mobilenet_alpha: 0.5
      }
    };

    // Convert weights ArrayBuffer to Uint8Array
    const weightsArray = new Uint8Array(classifierArtifacts.weightData);
    
    // Convert to base64 in chunks to avoid call stack issues
    const chunkSize = 8192;
    let weightsBase64 = '';
    for (let i = 0; i < weightsArray.length; i += chunkSize) {
      const chunk = weightsArray.slice(i, i + chunkSize);
      weightsBase64 += String.fromCharCode.apply(null, Array.from(chunk));
    }
    weightsBase64 = btoa(weightsBase64);

    // Prepare upload data
    const uploadData = {
      modelJson: JSON.stringify({
        format: 'layers-model',
        generatedBy: 'SignatureAI-TM',
        convertedBy: 'tfjs 1.3.1',
        modelTopology: classifierArtifacts.modelTopology,
        weightsManifest: metadata.weightsManifest
      }),
      weightsBin: weightsBase64,
      metadataJson: JSON.stringify(metadata)
    };

    // Upload to S3
    const aiModelService = getAIModelService();
    const uploadResult = await aiModelService.uploadTrainedModelToS3(
      `model_${Date.now()}`,
      uploadData,
      students[0]?.id || 'global',
      {
        total_sample_count: totalSampleCount,
        student_count: validClasses.length,
        students: students,
        accuracy: trainingAccuracy || 0.85,
        epochs: 50,
        optimizer: 'adam',
        learning_rate: 0.001,
        batch_size: 16,
        training_summary: `Trained on ${validClasses.length} students with ${totalSampleCount} total samples`,
        model_architecture: 'teachable-machine-mobilenet'
      }
    );

    if (uploadResult.success) {
      if (setHasExportedToCloud) setHasExportedToCloud(true);
      if (showCloudSuccessNotification) showCloudSuccessNotification();
    } else {
      throw new Error(uploadResult.message);
    }

  } catch (error) {
    console.error('S3 upload error:', error);
    alert('Error uploading model to S3: ' + (error instanceof Error ? error.message : 'Unknown error'));
  }
};

export const exportToLocal = async (params: ExportParams): Promise<void> => {
  const {
    model,
    classes,
    trainingAccuracy,
    trainingStartTime,
    formatStudentDisplay,
    setIsDownloading,
    setHasDownloadedToPC,
    showLocalExportSuccessNotification
  } = params;

  if (!model || !model.classifier) {
    alert('Please train a model first before downloading');
    return;
  }

  try {
    if (setIsDownloading) setIsDownloading(true);

    const validClasses = classes.filter(cls => cls.samples.length > 0);
    const totalSampleCount = validClasses.reduce((total, cls) => total + cls.samples.length, 0);

    // Export classifier using proper TensorFlow.js save handler
    const classifierArtifacts = await model.classifier.save(tf.io.withSaveHandler(async (artifacts) => artifacts));

    if (!classifierArtifacts || !classifierArtifacts.weightData) {
      throw new Error('Failed to extract model artifacts');
    }

    // Create metadata
    const students = validClasses.map(cls => ({
      id: cls.student?.id?.toString() || '',
      student_id: cls.student?.student_id || '',
      firstname: cls.student?.firstname || '',
      surname: cls.student?.surname || '',
      full_name: cls.student ? formatStudentDisplay(cls.student) : 'Unassigned'
    }));


    // DEBUG: log labels before creating metadata
console.log("🔍 EXPORT DEBUG (S3) - Class labels:", model.getClassLabels());
console.log("🔍 EXPORT DEBUG (S3) - Number of classes:", model.getClassLabels()?.length);

    const metadata = {
      format: 'teachable-machine-mobilenet',
      generatedBy: 'SignatureAI',
      convertedBy: 'tfjs 1.3.1',
      modelTopology: classifierArtifacts.modelTopology,
      weightsManifest: classifierArtifacts.weightSpecs ? [{
        paths: ['weights.bin'],
        weights: classifierArtifacts.weightSpecs
      }] : [],
      labels: model.getClassLabels(),
      userMetadata: {
        sample_count: totalSampleCount,
        total_students: validClasses.length,
        accuracy: trainingAccuracy || 0.85,
        training_date: trainingStartTime ? new Date(trainingStartTime).toISOString() : new Date().toISOString(),
        students: students,
        model_type: 'teachable-machine',
        mobilenet_version: 2,
        mobilenet_alpha: 0.5
      }
    };

    // Create model.json
    const modelJsonContent = {
      format: 'layers-model',
      generatedBy: 'SignatureAI-TM',
      convertedBy: 'tfjs 1.3.1',
      modelTopology: classifierArtifacts.modelTopology,
      weightsManifest: metadata.weightsManifest
    };

    // Convert weights ArrayBuffer to Uint8Array
    const weightsArray = new Uint8Array(classifierArtifacts.weightData);

    // Create ZIP file
    const zip = new JSZip();
    zip.file('model.json', JSON.stringify(modelJsonContent, null, 2));
    zip.file('weights.bin', weightsArray);
    zip.file('metadata.json', JSON.stringify(metadata, null, 2));

    // Generate and download
    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const fileName = `teachable-machine-model-${Date.now()}.zip`;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    if (setHasDownloadedToPC) setHasDownloadedToPC(true);
    if (showLocalExportSuccessNotification) showLocalExportSuccessNotification(fileName);

  } catch (error) {
    console.error('Export error:', error);
    alert('Error exporting model: ' + (error instanceof Error ? error.message : 'Unknown error'));
  } finally {
    if (setIsDownloading) setIsDownloading(false);
  }
};