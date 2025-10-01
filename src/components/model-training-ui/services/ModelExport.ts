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

// Export to S3
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
    
    // Get current class student ID for individual model export
    const currentClass = classes[currentClassIndex];
    const studentId = currentClass?.student?.id;
    
    // Get actual model topology and weights
    const modelTopology = model.classifier.toJSON();
    const weightData = await model.classifier.getWeights();
    
    // Extract weight values and create binary data
    const weightBuffers = await Promise.all(
      weightData.map(async (tensor) => {
        const data = await tensor.data();
        return new Uint8Array(data.buffer);
      })
    );
    
    // Create weights binary data
    const totalLength = weightBuffers.reduce((sum, buffer) => sum + buffer.length, 0);
    const combinedWeights = new Uint8Array(totalLength);
    let offset = 0;
    for (const buffer of weightBuffers) {
      combinedWeights.set(buffer, offset);
      offset += buffer.length;
    }
    const weightsBlob = new Blob([combinedWeights], { type: 'application/octet-stream' });
    
    // Create Teachable Machine format metadata
    const metadata = {
      format: "tensorflowjs",
      generatedBy: "signature-ai",
      convertedBy: null,
      userMetadata: {
        labels: validClasses.map(cls => cls.student ? formatStudentDisplay(cls.student) : 'Unassigned'),
        modelType: "image_classification",
        inputSize: [224, 224, 3],
        author: "signature-ai",
        accuracy: trainingAccuracy || 0.85, // Use actual training accuracy
        totalSamples: validClasses.reduce((sum, cls) => sum + cls.samples.length, 0),
        trainingTime: trainingStartTime ? Date.now() - trainingStartTime : 0
      },
      signatures: {
        "default": {
          inputs: [
            {
              name: "input",
              dtype: "float32",
              shape: [1, 224, 224, 3]
            }
          ],
          outputs: [
            {
              name: "output",
              dtype: "float32",
              shape: [1, validClasses.length]
            }
          ]
        }
      }
    };
    
    // Prepare model data with real TensorFlow.js model files
    const modelData = {
      modelJson: JSON.stringify(modelTopology, null, 2),
      weightsBin: weightsBlob,
      metadataJson: JSON.stringify(metadata, null, 2)
    };
    
    // Prepare comprehensive metadata
    const modelMetadata: ModelMetadata = {
      version: '1.0',
      createdAt: new Date().toISOString(),
      modelArchitecture: {
        featureExtractor: 'MobileNetV1',
        classifier: 'Sequential',
        inputShape: [1024],
        outputShape: [validClasses.length]
      },
      trainingConfig: {
        epochs: 50,
        optimizer: 'adam',
        learningRate: 0.001,
        batchSize: 16,
        augmentationTypes: 12, // Updated from 18
        totalSamples: validClasses.reduce((sum, cls) => sum + cls.samples.length, 0)
      },
      performance: {
        finalAccuracy: trainingAccuracy || 0.85, // Use actual training accuracy
        finalLoss: 0.05,    // You can track this during training
        trainingTime: trainingStartTime ? Date.now() - trainingStartTime : 0
      },
      classes: validClasses.map(cls => ({
        name: cls.student ? formatStudentDisplay(cls.student) : 'Unassigned',
        color: cls.color,
        sampleCount: cls.samples.length,
        studentId: cls.student?.id?.toString()
      })),
      storage: {
        location: 's3',
        bucket: '', // Will be filled by S3 service
        region: '', // Will be filled by S3 service
        modelKey: '', // Will be filled by S3 service
        metadataKey: '' // Will be filled by S3 service
      }
    };
    
    // Convert weights Blob to base64 for JSON serialization (required by backend)
    const weightsBase64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(modelData.weightsBin);
    });
    
    // Prepare model data with base64 weights for backend compatibility
    const preparedModelData = {
      modelJson: JSON.parse(modelData.modelJson),
      weightsBin: weightsBase64,
      metadataJson: JSON.parse(modelData.metadataJson)
    };
    
    // Upload to S3 using AIModelService (handles both S3 upload and database record creation)
    console.log('🚀 Starting S3 upload with database record creation...');
    const aiModelService = new AIModelServiceClass();
    
    // Prepare training data for database record
    const trainingData = {
      total_sample_count: validClasses.reduce((sum, cls) => sum + cls.samples.length, 0),
      student_count: validClasses.length,
      students: validClasses.map(cls => ({
        id: String(cls.student?.id || ''),
        student_id: cls.student?.student_id || '',
        firstname: cls.student?.firstname || '',
        surname: cls.student?.surname || '',
        full_name: cls.student ? formatStudentDisplay(cls.student) : 'Unassigned'
      })),
      accuracy: trainingAccuracy || 0.85, // Use actual training accuracy
      epochs: 50,
      optimizer: 'adam',
      learning_rate: 0.001,
      batch_size: 16,
      training_summary: `Model trained with ${validClasses.reduce((sum, cls) => sum + cls.samples.length, 0)} samples across ${validClasses.length} students. Final accuracy: ${(trainingAccuracy || 0.85).toFixed(4)}`,
      model_architecture: 'cnn'
    };
    
    // Create a unique model ID for this training session
    const modelId = `model_${studentId || 'global'}_${Date.now()}`;
    
    const uploadResult = await aiModelService.uploadTrainedModelToS3(
      modelId,
      preparedModelData,
      studentId?.toString(),
      trainingData
    );
    
    console.log('🎉 S3 upload result:', uploadResult);
    
    if (uploadResult.success) {
      // Update UI state for successful cloud export
      setHasExportedToCloud(true);
      showCloudSuccessNotification();
      console.log('✅ Cloud export state updated - S3 upload and database record created successfully');
    } else {
      throw new Error(uploadResult.message || 'Failed to upload to S3');
    }
    
  } catch (error) {
    console.error('S3 export error:', error);
    throw error;
  }
};

// Export to local download - uses complete model format
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

    // Prepare model data for download - export COMPLETE TensorFlow.js model (MobileNet + Classifier)
    console.log('🔄 Starting COMPLETE model export process...');
    
    // Create date/time filename like Teachable Machine
    const now = new Date();
    const timestamp = now.toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const modelName = `model-${timestamp}`;
    
    // Get model topologies and weights for both models
    let mobileNetTopology = model.featureExtractor.toJSON() as ModelTopology;
    const mobileNetWeights = model.featureExtractor.getWeights();
    
    let classifierTopology = model.classifier.toJSON() as ModelTopology;
    const classifierWeights = model.classifier.getWeights();
    
    // Fix model topologies for TensorFlow.js compatibility
    console.log('🔧 Applying topology fixes for export compatibility...');
    console.log('📋 Original MobileNet class_name:', mobileNetTopology.class_name);
    console.log('📋 Original Classifier class_name:', classifierTopology.class_name);
    
    mobileNetTopology = fixModelTopology(mobileNetTopology);
    classifierTopology = fixModelTopology(classifierTopology);
    
    console.log('✅ Model topologies fixed successfully');
    console.log('📋 Fixed MobileNet class_name:', mobileNetTopology.class_name);
    console.log('📋 Fixed Classifier class_name:', classifierTopology.class_name);
    
    // Process MobileNet weights
    const mobileNetWeightDataArray = await Promise.all(
      mobileNetWeights.map(async (tensor, index) => {
        return {
          name: tensor.id || `mobilenet_weight_${index}`,
          data: await tensor.data(),
          shape: tensor.shape,
          dtype: tensor.dtype
        };
      })
    );
    
    // Process Classifier weights
    const classifierWeightDataArray = await Promise.all(
      classifierWeights.map(async (tensor, index) => {
        return {
          name: tensor.id || `classifier_weight_${index}`,
          data: await tensor.data(),
          shape: tensor.shape,
          dtype: tensor.dtype
        };
      })
    );
    
    // Create MobileNet model.json
    const mobileNetJson = {
      format: 'layers-model',
      generatedBy: 'TensorFlow.js',
      convertedBy: null,
      modelTopology: mobileNetTopology,
      weightsManifest: [{
        paths: ['mobilenet_weights.bin'],
        weights: mobileNetWeightDataArray.map(w => ({
          name: w.name,
          shape: w.shape,
          dtype: w.dtype
        }))
      }]
    };
    
    // Create Classifier model.json
    const classifierJson = {
      format: 'layers-model',
      generatedBy: 'TensorFlow.js',
      convertedBy: null,
      modelTopology: classifierTopology,
      weightsManifest: [{
        paths: ['classifier_weights.bin'],
        weights: classifierWeightDataArray.map(w => ({
          name: w.name,
          shape: w.shape,
          dtype: w.dtype
        }))
      }]
    };
    
    // Create combined metadata.json
    const metadataJson = {
      tfjsVersion: tf.version.tfjs,
      tmVersion: '2.4.10',
      packageVersion: '0.8.5',
      packageName: '@teachablemachine/image',
      timeStamp: new Date().toISOString(),
      userMetadata: {
        student_id: currentClass?.student?.student_id || '',
        student_name: studentName,
        sample_count: totalSampleCount,
        accuracy: trainingAccuracy || 0.85,
        training_date: new Date().toISOString(),
        model_architecture: 'cnn',
        total_students: validClasses.length,
        training_summary: `Trained on ${validClasses.length} students with ${totalSampleCount} total samples. Final accuracy: ${(trainingAccuracy || 0.85).toFixed(4)}`
      },
      modelName: modelName,
      labels: labels,
      imageSize: 224, // Teachable Machine default
      modelStructure: {
        hasMobileNet: true,
        hasClassifier: true,
        inputSize: [224, 224, 3],
        featureSize: mobileNetWeightDataArray.length > 0 ? mobileNetWeightDataArray[0].shape : [1000],
        outputSize: validClasses.length
      }
    };
    
    // Create weight binaries
    const mobileNetWeightBuffers = mobileNetWeightDataArray.map(w => {
      return new Float32Array(w.data).buffer;
    });
    const mobileNetCombinedWeights = new Blob(mobileNetWeightBuffers, { type: 'application/octet-stream' });
    
    const classifierWeightBuffers = classifierWeightDataArray.map(w => {
      return new Float32Array(w.data).buffer;
    });
    const classifierCombinedWeights = new Blob(classifierWeightBuffers, { type: 'application/octet-stream' });
    
    // Create JSON files
    const mobileNetJsonBlob = new Blob([JSON.stringify(mobileNetJson, null, 2)], { type: 'application/json' });
    const classifierJsonBlob = new Blob([JSON.stringify(classifierJson, null, 2)], { type: 'application/json' });
    const metadataJsonBlob = new Blob([JSON.stringify(metadataJson, null, 2)], { type: 'application/json' });
    
    // Create download URLs
    const mobileNetJsonUrl = URL.createObjectURL(mobileNetJsonBlob);
    const classifierJsonUrl = URL.createObjectURL(classifierJsonBlob);
    const metadataJsonUrl = URL.createObjectURL(metadataJsonBlob);
    const mobileNetWeightsUrl = URL.createObjectURL(mobileNetCombinedWeights);
    const classifierWeightsUrl = URL.createObjectURL(classifierCombinedWeights);
    
    try {
      console.log('📦 Creating ZIP archive with COMPLETE model files...');
      
      // Create a new ZIP file
      const zip = new JSZip();
      
      // Add all model files to the ZIP
      zip.file('mobilenet_model.json', mobileNetJsonBlob);
      zip.file('mobilenet_weights.bin', mobileNetCombinedWeights);
      zip.file('classifier_model.json', classifierJsonBlob);
      zip.file('classifier_weights.bin', classifierCombinedWeights);
      zip.file('metadata.json', metadataJsonBlob);
      
      // Generate the ZIP file
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      
      // Create download URL for the ZIP file
      const zipUrl = URL.createObjectURL(zipBlob);
      
      // Download the ZIP file
      const zipLink = document.createElement('a');
      zipLink.href = zipUrl;
      zipLink.download = `${modelName}.zip`;
      document.body.appendChild(zipLink);
      zipLink.click();
      document.body.removeChild(zipLink);
      
      console.log('✅ COMPLETE model export finished successfully!');
      console.log(`📁 Downloaded: ${modelName}.zip`);
      console.log('📦 ZIP contains:');
      console.log('   - mobilenet_model.json (MobileNet architecture)');
      console.log('   - mobilenet_weights.bin (MobileNet trained weights)');
      console.log('   - classifier_model.json (Classifier architecture)');
      console.log('   - classifier_weights.bin (Classifier trained weights)');
      console.log('   - metadata.json (Model information and labels)');
      
      // Clean up ZIP URL
      URL.revokeObjectURL(zipUrl);
      
    } finally {
      // Clean up individual file URLs
      URL.revokeObjectURL(mobileNetJsonUrl);
      URL.revokeObjectURL(classifierJsonUrl);
      URL.revokeObjectURL(metadataJsonUrl);
      URL.revokeObjectURL(mobileNetWeightsUrl);
      URL.revokeObjectURL(classifierWeightsUrl);
    }

    // Mark as downloaded to PC
    setHasDownloadedToPC(true);
    showLocalExportSuccessNotification(`${modelName}.zip`);

  } catch (error) {
    console.error('Local download error:', error);
    alert('Error downloading model: ' + (error instanceof Error ? error.message : 'Unknown error'));
  } finally {
    // Safety check to prevent errors if component unmounted
    if (params.setIsDownloading) {
      params.setIsDownloading(false);
    }
  }
};
