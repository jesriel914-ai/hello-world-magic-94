// filepath: src/components/model-training-ui/services/ModelImport.ts
// Loads MobileNet-based classifier models

import * as tf from '@tensorflow/tfjs';
import * as tmImage from '@teachablemachine/image';
import JSZip from 'jszip';

export interface ImportedModel {
  featureExtractor: any; // MobileNet feature extractor
  classifier: tf.LayersModel;
  metadata: {
    labels: string[];
    userMetadata: {
      accuracy: number;
      sample_count: number;
      total_students: number;
      model_type: string;
      mobilenet_version?: number;
      mobilenet_alpha?: number;
    };
  };
}

/**
 * Load local model from file selection
 */
export async function loadLocalModel(files: FileList): Promise<ImportedModel> {
  console.log('Loading local model files...');
  
  const fileArray = Array.from(files);
  
  const modelJsonFile = fileArray.find(f => f.name === 'model.json');
  const weightsBinFile = fileArray.find(f => f.name === 'weights.bin');
  const metadataJsonFile = fileArray.find(f => f.name === 'metadata.json');
  
  if (!modelJsonFile || !weightsBinFile || !metadataJsonFile) {
    throw new Error('Missing required files. Expected: model.json, weights.bin, metadata.json');
  }
  
  console.log('Found all required files');
  
  const metadataText = await metadataJsonFile.text();
  const metadata = JSON.parse(metadataText);
  
  console.log('Metadata:', metadata);
  
  // Extract MobileNet config (with fallbacks for different export formats)
  let mobilenetVersion = 2;
  let mobilenetAlpha = 0.5;
  
  if (metadata.userMetadata?.mobilenet_version) {
    mobilenetVersion = metadata.userMetadata.mobilenet_version;
  } else if (metadata.modelArchitecture?.featureExtractor === 'MobileNetV2') {
    mobilenetVersion = 2;
  }
  
  if (metadata.userMetadata?.mobilenet_alpha) {
    mobilenetAlpha = metadata.userMetadata.mobilenet_alpha;
  }
  
  console.log(`Loading MobileNet v${mobilenetVersion} (alpha: ${mobilenetAlpha})...`);
  
  const mobilenet = await tmImage.loadTruncatedMobileNet({
    version: mobilenetVersion,
    alpha: mobilenetAlpha,
    inputResolution: 224,
    classifierInputSize: 1024
  });
  
  console.log('MobileNet loaded successfully');
  console.log('Loading classifier from files...');
  
  const classifier = await tf.loadLayersModel(
    tf.io.browserFiles([modelJsonFile, weightsBinFile])
  );
  
  console.log('Classifier loaded successfully');
  
  // Extract labels (support different metadata formats)
  let labels: string[] = [];
  if (metadata.labels) {
    labels = metadata.labels;
  } else if (metadata.classes) {
    labels = metadata.classes.map((c: any) => c.name);
  }
  
  return {
    featureExtractor: mobilenet,
    classifier: classifier,
    metadata: {
      labels: labels,
      userMetadata: metadata.userMetadata || {
        accuracy: metadata.performance?.finalAccuracy || 0.85,
        sample_count: metadata.trainingConfig?.totalSamples || 0,
        total_students: labels.length,
        model_type: 'mobilenet-classifier'
      }
    }
  };
}

/**
 * Load model from S3 download data
 */
export async function loadModelFromDownload(downloadData: string): Promise<ImportedModel> {
  console.log('Loading model from download data...');
  
  const combinedData = JSON.parse(downloadData);
  
  if (!combinedData.modelJson || !combinedData.weightsBin || !combinedData.metadataJson) {
    throw new Error('Invalid download data format');
  }
  
  const metadata = JSON.parse(combinedData.metadataJson);
  
  console.log('Metadata:', metadata);
  
  // Extract MobileNet config
  let mobilenetVersion = 2;
  let mobilenetAlpha = 0.5;
  
  if (metadata.userMetadata?.mobilenet_version) {
    mobilenetVersion = metadata.userMetadata.mobilenet_version;
  } else if (metadata.modelArchitecture?.featureExtractor === 'MobileNetV2') {
    mobilenetVersion = 2;
  }
  
  if (metadata.userMetadata?.mobilenet_alpha) {
    mobilenetAlpha = metadata.userMetadata.mobilenet_alpha;
  }
  
  console.log(`Loading MobileNet v${mobilenetVersion} (alpha: ${mobilenetAlpha})...`);
  
  const mobilenet = await tmImage.loadTruncatedMobileNet({
    version: mobilenetVersion,
    alpha: mobilenetAlpha,
    inputResolution: 224,
    classifierInputSize: 1024
  });
  
  console.log('MobileNet loaded');
  console.log('Loading classifier...');

  console.log('Raw modelJson type:', typeof combinedData.modelJson);
  console.log('Raw modelJson length:', combinedData.modelJson?.length);
  console.log('First 200 chars of modelJson:', combinedData.modelJson?.substring(0, 200));

  // Parse the model JSON - might be double-encoded
  let parsedModelJson;
  try {
    parsedModelJson = JSON.parse(combinedData.modelJson);
    
    // Check if it's a string (double-encoded) and parse again
    if (typeof parsedModelJson === 'string') {
      console.log('Model JSON is double-encoded, parsing again...');
      parsedModelJson = JSON.parse(parsedModelJson);
    }
    
    console.log('Parsed modelJson type:', typeof parsedModelJson);
    console.log('Parsed modelJson keys:', Object.keys(parsedModelJson));
    
    // Now check for modelTopology
    if (!parsedModelJson || typeof parsedModelJson !== 'object') {
      throw new Error('Parsed model JSON is not an object');
    }
    
    console.log('Has modelTopology:', 'modelTopology' in parsedModelJson);
    
    // If there's no modelTopology field, the JSON IS the topology
if (!('modelTopology' in parsedModelJson)) {
  console.log('modelJson IS the topology, wrapping it...');
  
  // Extract layer information to build weight specs
  const layers = parsedModelJson.config?.layers || [];
  const weightSpecs: any[] = [];
  
  layers.forEach((layer: any, index: number) => {
    if (layer.class_name === 'Dense') {
      const units = layer.config.units;
      
      // For the first layer, use the feature size from metadata or default to 1280
      let inputShape: number;
      if (index === 0) {
        inputShape = 1280; // MobileNet v2 feature size
      } else {
        // Get input shape from previous Dense layer
        const prevDenseLayer = layers.slice(0, index).reverse().find((l: any) => l.class_name === 'Dense');
        inputShape = prevDenseLayer?.config?.units || 1280;
      }
      
      // Kernel weights
      weightSpecs.push({
        name: `${layer.config.name}/kernel`,
        shape: [inputShape, units],
        dtype: 'float32'
      });
      
      // Bias weights (if enabled)
      if (layer.config.use_bias !== false) {
        weightSpecs.push({
          name: `${layer.config.name}/bias`,
          shape: [units],
          dtype: 'float32'
        });
      }
    }
  });
  
  console.log('Generated weight specs:', weightSpecs);
  console.log('Output layer units:', layers[layers.length - 1]?.config?.units);
  
  parsedModelJson = {
    modelTopology: parsedModelJson,
    weightsManifest: [{
      paths: ['weights.bin'],
      weights: weightSpecs
    }]
  };
}
    
  } catch (e) {
    console.error('Failed to parse modelJson:', e);
    throw new Error('Invalid model JSON format');
  }

  // Create a proper model.json structure for tf.loadLayersModel
  const tfModelJson = {
    modelTopology: parsedModelJson.modelTopology,
    weightsManifest: parsedModelJson.weightsManifest || [{
      paths: ['weights.bin'],
      weights: []
    }]
  };

  const modelJsonBlob = new Blob([JSON.stringify(tfModelJson)], { type: 'application/json' });
  const modelJsonFile = new File([modelJsonBlob], 'model.json');

  // Process weights
  const base64Data = combinedData.weightsBin.includes(',') 
    ? combinedData.weightsBin.split(',')[1] 
    : combinedData.weightsBin;

  const binaryString = atob(base64Data);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  const weightsBlob = new Blob([bytes], { type: 'application/octet-stream' });
  const weightsFile = new File([weightsBlob], 'weights.bin');

  const classifier = await tf.loadLayersModel(
    tf.io.browserFiles([modelJsonFile, weightsFile])
  );

  console.log('Classifier loaded successfully');
    
// Extract labels (support different metadata formats)
let labels: string[] = [];

// Check multiple possible locations for labels
if (metadata.labels && Array.isArray(metadata.labels)) {
  labels = metadata.labels;
} else if (metadata.classes && Array.isArray(metadata.classes)) {
  labels = metadata.classes.map((c: any) => c.name || c);
} else if (metadata.modelStructure?.labels && Array.isArray(metadata.modelStructure.labels)) {
  labels = metadata.modelStructure.labels;
} else if (metadata.userMetadata?.students && Array.isArray(metadata.userMetadata.students)) {
  // Build labels from student list
  labels = metadata.userMetadata.students.map((s: any) => 
    s.full_name || `${s.firstname || ''} ${s.surname || ''}`.trim() || 'Unknown'
  );
} else if (metadata.trainingConfig?.classes && Array.isArray(metadata.trainingConfig.classes)) {
  labels = metadata.trainingConfig.classes.map((c: any) => c.name || c);
}

console.log('Extracted labels:', labels);

if (labels.length === 0) {
  console.error('Full metadata structure:', JSON.stringify(metadata, null, 2));
  throw new Error('Could not find labels in metadata. Check console for full metadata structure.');
}


    // ADD THE VALIDATION HERE - right before the return statement:
  console.log('Loaded model output shape:', classifier.outputs[0].shape);
  console.log('Expected number of classes:', labels.length);

  // Verify the output layer matches the number of labels
  const outputShape = classifier.outputs[0].shape[1];
  if (outputShape !== labels.length) {
    throw new Error(
      `Model output mismatch: Model has ${outputShape} outputs but metadata has ${labels.length} labels. ` +
      `This indicates the model structure doesn't match the training data.`
    );
  }

  console.log('Model structure validated - output matches labels');

  return {
    featureExtractor: mobilenet,
    classifier: classifier,
    metadata: {
      labels: labels,
      userMetadata: metadata.userMetadata || {
        accuracy: metadata.performance?.finalAccuracy || 0.85,
        sample_count: metadata.trainingConfig?.totalSamples || 0,
        total_students: labels.length,
        model_type: 'mobilenet-classifier'
      }
    }
  };
}