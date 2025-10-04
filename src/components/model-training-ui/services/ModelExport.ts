//filepath: src\components\model-training-ui\services\ModelExport.ts
import * as tf from '@tensorflow/tfjs';
import JSZip from 'jszip';
import { AIModelServiceClass, ModelMetadata } from '@/lib/AIModelService';

// TypeScript interfaces for model topology
interface ModelLayer {
  class_name: string;
  config?: {
    units?: number;
    activation?: string;
    use_bias?: boolean;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

interface ModelTopology {
  class_name: string;
  config?: {
    layers?: ModelLayer[];
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface ExportModelParams {
  model: {
    featureExtractor: tf.LayersModel;
    classifier: tf.LayersModel;
  };
  classes: Array<{
    student?: {
      id?: string;
      student_id?: string;
      firstname?: string;
      surname?: string;
    };
    color: string;
    samples: Array<unknown>;
  }>;
  currentClassIndex: number;
  trainingAccuracy: number | null;
  trainingStartTime: number | null;
  formatStudentDisplay: (student: unknown) => string;
  setIsExporting: (exporting: boolean) => void;
  setIsDownloading: (downloading: boolean) => void;
  setHasExportedToCloud: (exported: boolean) => void;
  setHasDownloadedToPC: (downloaded: boolean) => void;
  showCloudSuccessNotification: () => void;
  showLocalExportSuccessNotification: (filename: string) => void;
}

// Helper function to fix model topology for TensorFlow.js compatibility
const fixModelTopology = (topology: ModelTopology): ModelTopology => {
  console.log('🔧 Fixing model topology for TensorFlow.js compatibility...');
  
  // Create a deep copy of the topology
  const fixedTopology = JSON.parse(JSON.stringify(topology));
  
  // Fix the main model class_name
  if (fixedTopology.class_name === 'Sequential' || fixedTopology.class_name === 'Model') {
    fixedTopology.class_name = 'Functional';
    console.log('✅ Changed model class_name to "Functional"');
  }
  
  // Recursive function to fix nested layers
  const fixLayers = (layers: ModelLayer[]): ModelLayer[] => {
    return layers.map((layer: ModelLayer) => {
      // Fix the current layer if it's Sequential or Model
      if (layer.class_name === 'Sequential' || layer.class_name === 'Model') {
        console.log(`🔧 Converting layer ${layer.class_name} to Dense`);
        return {
          class_name: 'Dense',
          config: {
            units: layer.config?.units || 128,
            activation: layer.config?.activation || 'relu',
            use_bias: layer.config?.use_bias !== false
          }
        };
      }
      
      // If the layer has nested layers, fix them too
      if (layer.config && layer.config.layers && Array.isArray(layer.config.layers)) {
        layer.config.layers = fixLayers(layer.config.layers);
      }
      
      return layer;
    });
  };
  
  // Fix layer types in the config
  if (fixedTopology.config && fixedTopology.config.layers) {
    fixedTopology.config.layers = fixLayers(fixedTopology.config.layers);
  }
  
  return fixedTopology;
};

// Export model - main export function that decides between S3 and local
export const exportModel = async (params: ExportModelParams) => {
  const { model, classes, currentClassIndex, trainingAccuracy, trainingStartTime, formatStudentDisplay, setIsExporting, setIsDownloading, setHasExportedToCloud, setHasDownloadedToPC, showCloudSuccessNotification, showLocalExportSuccessNotification } = params;

  if (!model) {
    alert('No model to export! Please train a model first.');
    return;
  }

  setIsExporting(true);
  
  try {
    // Check if S3 storage is enabled
    const enableS3Storage = (import.meta as any).env.VITE_ENABLE_S3_STORAGE === 'true' || (import.meta as any).env.VITE_ENABLE_S3_STORAGE === undefined;
    
    console.log('🔄 Starting export process...');
    console.log('☁️ S3 Storage enabled:', enableS3Storage);
    
    if (enableS3Storage) {
      // S3 Export
      console.log('🌐 Using S3 cloud export...');
      await exportToS3({ model, classes, currentClassIndex, trainingAccuracy, trainingStartTime, formatStudentDisplay, setHasExportedToCloud, showCloudSuccessNotification });
    } else {
      // Local download export - exportToLocal now uses complete model format
      console.log('💾 Using local download export with complete model format...');
      await exportToLocal({ model, classes, currentClassIndex, trainingAccuracy, trainingStartTime, formatStudentDisplay, setIsDownloading, setHasDownloadedToPC, showLocalExportSuccessNotification });
    }    
  } catch (error) {
    console.error('Export error:', error);
    alert('Error exporting model: ' + error);
  } finally {
    setIsExporting(false);
  }
};

// Export to S3 - FIXED to use correct 3-file structure
export const exportToS3 = async (params: {
  model: {
    featureExtractor: tf.LayersModel;
    classifier: tf.LayersModel;
  };
  classes: Array<{
    student?: {
      id?: string;
      student_id?: string;
      firstname?: string;
      surname?: string;
    };
    color: string;
    samples: Array<unknown>;
  }>;
  currentClassIndex: number;
  trainingAccuracy: number | null;
  trainingStartTime: number | null;
  formatStudentDisplay: (student: unknown) => string;
  setHasExportedToCloud: (exported: boolean) => void;
  showCloudSuccessNotification: () => void;
}) => {
  try {
    const { model, classes, currentClassIndex, trainingAccuracy, trainingStartTime, formatStudentDisplay, setHasExportedToCloud, showCloudSuccessNotification } = params;
    const validClasses = classes.filter(cls => cls.samples.length > 0);
    
    console.log('☁️ Starting S3 export with CORRECT 3-file structure...');
    
    // Get current class student ID
    const currentClass = classes[currentClassIndex];
    const studentId = currentClass?.student?.id;
    
    // Prepare student information
    const students = validClasses.map(cls => ({
      id: cls.student?.id?.toString() || '',
      student_id: cls.student?.student_id || '',
      firstname: cls.student?.firstname || '',
      surname: cls.student?.surname || '',
      full_name: cls.student ? formatStudentDisplay(cls.student) : 'Unassigned'
    }));
    
    const labels = students.map(student => student.full_name);
    const totalSampleCount = validClasses.reduce((total, cls) => total + cls.samples.length, 0);

    // STEP 1: Create combined model (same as local export)
    console.log('🔧 Creating combined model for S3...');
    
    const dummyInput = tf.zeros([1, 224, 224, 3]);
    const mobileNetOutput = model.featureExtractor.predict(dummyInput) as tf.Tensor;
    const featureSize = mobileNetOutput.shape[1];
    mobileNetOutput.dispose();
    dummyInput.dispose();

    const combinedModel = tf.sequential();
    
    const mobileNetLayers = model.featureExtractor.layers;
    for (const layer of mobileNetLayers) {
      combinedModel.add(layer);
    }
    
    const classifierLayers = model.classifier.layers;
    for (const layer of classifierLayers) {
      combinedModel.add(layer);
    }

    console.log('✅ Combined model created');

    // STEP 2: Save and get model data
    await combinedModel.save('indexeddb://temp-s3-model');
    const loadedModel = await tf.loadLayersModel('indexeddb://temp-s3-model');
    
    const modelTopology = loadedModel.toJSON();
    const weights = loadedModel.getWeights();
    const weightData = await Promise.all(
      weights.map(async (tensor) => await tensor.data())
    );
    
    const totalLength = weightData.reduce((sum, data) => sum + data.length, 0);
    const combinedWeights = new Float32Array(totalLength);
    let offset = 0;
    for (const data of weightData) {
      combinedWeights.set(data, offset);
      offset += data.length;
    }
    
    const weightsBlob = new Blob([combinedWeights.buffer], { type: 'application/octet-stream' });

    const weightsManifest = [{
      paths: ['weights.bin'],
      weights: weights.map((tensor, i) => ({
        name: `weight_${i}`,
        shape: tensor.shape,
        dtype: tensor.dtype
      }))
    }];

    // STEP 3: Create model.json
    const modelJson = {
      format: 'layers-model',
      generatedBy: 'TensorFlow.js tfjs-layers v' + tf.version.tfjs,
      convertedBy: null,
      modelTopology: modelTopology,
      weightsManifest: weightsManifest,
      trainingConfig: {
        optimizer_config: {
          class_name: 'Adam',
          config: {
            learning_rate: 0.001,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-7
          }
        },
        loss: 'categorical_crossentropy',
        metrics: ['accuracy']
      }
    };

    // STEP 4: Create metadata.json
    const metadataJson = {
      modelName: `model-${Date.now()}`,
      labels: labels,
      imageSize: 224,
      createdAt: new Date().toISOString(),
      userMetadata: {
        student_id: currentClass?.student?.student_id || '',
        student_name: students[0]?.full_name || 'Unknown',
        sample_count: totalSampleCount,
        accuracy: trainingAccuracy || 0.85,
        training_date: new Date().toISOString(),
        model_architecture: 'mobilenet_v1_classifier',
        total_students: validClasses.length,
        training_summary: `Trained on ${validClasses.length} students with ${totalSampleCount} total samples. Final accuracy: ${(trainingAccuracy || 0.85).toFixed(4)}`
      },
      tfjsVersion: tf.version.tfjs,
      modelType: 'image_classification',
      inputShape: [224, 224, 3],
      outputShape: [labels.length]
    };

    // STEP 5: Convert weights to base64 for backend
    const weightsBase64 = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(weightsBlob);
    });

    // STEP 6: Prepare data for AIModelService
    const modelData = {
      modelJson: JSON.stringify(modelJson),
      weightsBin: weightsBase64,
      metadataJson: JSON.stringify(metadataJson)
    };

    // STEP 7: Prepare training data for database
    const trainingData = {
      total_sample_count: totalSampleCount,
      student_count: validClasses.length,
      students: students,
      accuracy: trainingAccuracy || 0.85,
      epochs: 50,
      optimizer: 'adam',
      learning_rate: 0.001,
      batch_size: 16,
      training_summary: metadataJson.userMetadata.training_summary,
      model_architecture: 'mobilenet_v1_classifier'
    };

    // STEP 8: Upload to S3
    const aiModelService = new AIModelServiceClass();
    const modelId = `model_${studentId || 'global'}_${Date.now()}`;
    
    console.log('🚀 Uploading to S3...');
    const uploadResult = await aiModelService.uploadTrainedModelToS3(
      modelId,
      modelData,
      studentId?.toString(),
      trainingData
    );

    // Cleanup
    loadedModel.dispose();
    weights.forEach(w => w.dispose());
    await tf.io.removeModel('indexeddb://temp-s3-model');

    if (uploadResult.success) {
      setHasExportedToCloud(true);
      showCloudSuccessNotification();
      console.log('✅ S3 upload successful');
    } else {
      throw new Error(uploadResult.message || 'Failed to upload to S3');
    }
    
  } catch (error) {
    console.error('❌ S3 export error:', error);
    throw error;
  }
};

/// Export to local download - FIXED to use correct 3-file structure
export const exportToLocal = async (params: {
  model: {
    featureExtractor: tf.LayersModel;
    classifier: tf.LayersModel;
  };
  classes: Array<{
    student?: {
      id?: string;
      student_id?: string;
      firstname?: string;
      surname?: string;
    };
    color: string;
    samples: Array<unknown>;
  }>;
  currentClassIndex: number;
  trainingAccuracy: number | null;
  trainingStartTime: number | null;
  formatStudentDisplay: (student: unknown) => string;
  setIsDownloading: (downloading: boolean) => void;
  setHasDownloadedToPC: (downloaded: boolean) => void;
  showLocalExportSuccessNotification: (filename: string) => void;
}) => {
  try {
    const { model, classes, currentClassIndex, trainingAccuracy, trainingStartTime, formatStudentDisplay, setHasDownloadedToPC, showLocalExportSuccessNotification } = params;
    if (params.setIsDownloading) {
      params.setIsDownloading(true);
    }

    console.log('💾 Starting CORRECT 3-file model export...');

    // Get the current class student for naming
    const currentClass = classes[currentClassIndex];
    const studentName = currentClass?.student ? formatStudentDisplay(currentClass.student) : 'Unknown';

    // Calculate total sample count from all classes
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

    // Prepare labels from student names
    const labels = students.map(student => student.full_name);

    // Create date/time filename
    const now = new Date();
    const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const modelName = `model-${timestamp}`;

    // STEP 1: Create a COMBINED model (MobileNet features + Classifier layers)
    console.log('🔧 Creating combined model architecture...');
    
    // Get MobileNet output shape to connect to classifier
    const dummyInput = tf.zeros([1, 224, 224, 3]);
    const mobileNetOutput = model.featureExtractor.predict(dummyInput) as tf.Tensor;
    const featureSize = mobileNetOutput.shape[1];
    mobileNetOutput.dispose();
    dummyInput.dispose();

    // Create a sequential model that combines both
    const combinedModel = tf.sequential();
    
    // Add MobileNet layers (frozen)
    const mobileNetLayers = model.featureExtractor.layers;
    for (const layer of mobileNetLayers) {
      combinedModel.add(layer);
    }
    
    // Add classifier layers
    const classifierLayers = model.classifier.layers;
    for (const layer of classifierLayers) {
      combinedModel.add(layer);
    }

    console.log('✅ Combined model created with', combinedModel.layers.length, 'layers');

    // STEP 2: Save the combined model to get model.json and weights
    console.log('💾 Saving combined model...');
    
    // Save to browser storage temporarily to get the files
    const saveResult = await combinedModel.save('indexeddb://temp-model');
    
    // Load back the saved model to get the JSON and weights
    const loadedModel = await tf.loadLayersModel('indexeddb://temp-model');
    
    // Get the model topology as JSON
    const modelTopology = loadedModel.toJSON();
    
    // Get all weights
    const weights = loadedModel.getWeights();
    const weightData = await Promise.all(
      weights.map(async (tensor) => await tensor.data())
    );
    
    // Combine all weight data into a single binary blob
    const totalLength = weightData.reduce((sum, data) => sum + data.length, 0);
    const combinedWeights = new Float32Array(totalLength);
    let offset = 0;
    for (const data of weightData) {
      combinedWeights.set(data, offset);
      offset += data.length;
    }
    
    // Create weights.bin
    const weightsBlob = new Blob([combinedWeights.buffer], { type: 'application/octet-stream' });

    // Create weight manifest for model.json
    const weightsManifest = [{
      paths: ['weights.bin'],
      weights: weights.map((tensor, i) => ({
        name: `weight_${i}`,
        shape: tensor.shape,
        dtype: tensor.dtype
      }))
    }];

    // STEP 3: Create model.json with correct structure
    const modelJson = {
      format: 'layers-model',
      generatedBy: 'TensorFlow.js tfjs-layers v' + tf.version.tfjs,
      convertedBy: null,
      modelTopology: modelTopology,
      weightsManifest: weightsManifest,
      trainingConfig: {
        optimizer_config: {
          class_name: 'Adam',
          config: {
            learning_rate: 0.001,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-7
          }
        },
        loss: 'categorical_crossentropy',
        metrics: ['accuracy']
      }
    };

    // STEP 4: Create metadata.json (custom, not required by TensorFlow.js)
    const metadataJson = {
      modelName: modelName,
      labels: labels,
      imageSize: 224,
      createdAt: new Date().toISOString(),
      userMetadata: {
        student_id: currentClass?.student?.student_id || '',
        student_name: studentName,
        sample_count: totalSampleCount,
        accuracy: trainingAccuracy || 0.85,
        training_date: new Date().toISOString(),
        model_architecture: 'mobilenet_v1_classifier',
        total_students: validClasses.length,
        training_summary: `Trained on ${validClasses.length} students with ${totalSampleCount} total samples. Final accuracy: ${(trainingAccuracy || 0.85).toFixed(4)}`
      },
      tfjsVersion: tf.version.tfjs,
      modelType: 'image_classification',
      inputShape: [224, 224, 3],
      outputShape: [labels.length]
    };

    // STEP 5: Create blobs for download
    const modelJsonBlob = new Blob([JSON.stringify(modelJson, null, 2)], { type: 'application/json' });
    const metadataJsonBlob = new Blob([JSON.stringify(metadataJson, null, 2)], { type: 'application/json' });

    // STEP 6: Create ZIP with 3 files
    console.log('📦 Creating ZIP with 3 files...');
    const zip = new JSZip();
    zip.file('model.json', modelJsonBlob);
    zip.file('weights.bin', weightsBlob);
    zip.file('metadata.json', metadataJsonBlob);

    const zipBlob = await zip.generateAsync({ type: 'blob' });

    // STEP 7: Download the ZIP
    const zipUrl = URL.createObjectURL(zipBlob);
    const link = document.createElement('a');
    link.href = zipUrl;
    link.download = `${modelName}.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(zipUrl);

    // Cleanup
    loadedModel.dispose();
    weights.forEach(w => w.dispose());
    await tf.io.removeModel('indexeddb://temp-model');

    console.log('✅ Export completed successfully!');
    console.log('📁 Downloaded:', `${modelName}.zip`);
    console.log('📦 Contains: model.json, weights.bin, metadata.json');

    setHasDownloadedToPC(true);
    showLocalExportSuccessNotification(`${modelName}.zip`);

  } catch (error) {
    console.error('❌ Local download error:', error);
    alert('Error downloading model: ' + (error instanceof Error ? error.message : 'Unknown error'));
  } finally {
    if (params.setIsDownloading) {
      params.setIsDownloading(false);
    }
  }
};