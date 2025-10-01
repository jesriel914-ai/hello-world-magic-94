import * as tf from '@tensorflow/tfjs';

// TypeScript interfaces for model topology
// TensorFlow.js expects inbound_nodes to be: string[][][][]
type InboundNodeConnection = string[][];
type InboundNode = InboundNodeConnection[];
type InboundNodes = InboundNode[];

interface ModelLayerConfig {
  units?: number;
  activation?: string;
  use_bias?: boolean;
  batch_input_shape?: number[] | null[];
  layers?: ModelLayer[];
  [key: string]: unknown;
}

interface ModelLayer {
  class_name: string;
  name?: string;
  config?: ModelLayerConfig;
  inbound_nodes?: InboundNodes;
  [key: string]: unknown;
}

interface ModelTopologyConfig {
  layers?: ModelLayer[];
  [key: string]: unknown;
}

interface ModelTopology {
  class_name: string;
  config?: ModelTopologyConfig;
  inbound_nodes?: InboundNode[];
  [key: string]: unknown;
}

// Helper function to fix model topology for TensorFlow.js compatibility
const fixModelTopology = (topology: ModelTopology): ModelTopology => {
  console.log('🔧 Input topology keys:', Object.keys(topology));
  
  // Create a deep copy of the topology
  const fixedTopology = JSON.parse(JSON.stringify(topology));
  
  // For TensorFlow.js Model compatibility, we need to ensure proper structure
  // Model layers need inbound_nodes and proper layer connections
  if (topology.class_name === 'Sequential' || topology.class_name === 'Model') {
    // Keep as Sequential for simpler models, convert to Model only for complex ones
    const hasComplexLayers = topology.config && topology.config.layers && 
      topology.config.layers.some((layer: ModelLayer) => 
        layer.class_name === 'Model' || layer.class_name === 'Functional'
      );
    
    if (hasComplexLayers) {
      fixedTopology.class_name = 'Model';
      // Ensure Model has proper inbound_nodes
      if (!fixedTopology.inbound_nodes) {
        fixedTopology.inbound_nodes = [];
      }
      console.log('✅ Changed model class_name to "Model" for complex layer compatibility');
    } else {
      fixedTopology.class_name = 'Sequential';
      console.log('✅ Kept model class_name as "Sequential" for simple model');
    }
  }
  
  // Recursive function to fix nested layers - preserve structure but ensure compatibility
  const fixLayers = (layers: ModelLayer[], depth: number = 0, parentLayer?: ModelLayer): ModelLayer[] => {
    const processingLog = `🔧 Processing ${layers.length} layers at depth ${depth}`;
    console.log(processingLog);
    return layers.map((layer: ModelLayer, index: number) => {
      const layerLog = `🔧 Layer ${index} at depth ${depth}: ${layer.class_name}`;
      console.log(layerLog);
      
      // Ensure layer has proper inbound_nodes structure
      if (!layer.inbound_nodes) {
        layer.inbound_nodes = [];
      }
      
      // For Model/Functional layers, ensure proper input/output connections
      if (layer.class_name === 'Model' || layer.class_name === 'Functional') {
        const foundLog = `🔧 Found ${layer.class_name} layer at depth ${depth}`;
        console.log(foundLog);
        
        // Ensure inbound_nodes exist and are properly structured
        if (!layer.inbound_nodes || layer.inbound_nodes.length === 0) {
          // Create proper inbound_nodes structure for Model layers
          // TensorFlow.js expects array format: [[[['input_1']]]]
          layer.inbound_nodes = [];
          layer.inbound_nodes.push([]);
          layer.inbound_nodes[0].push([]);
          layer.inbound_nodes[0][0].push(['input_1']);
          console.log('🔧 Added proper inbound_nodes structure for Model layer');
        }
        
        // If this layer has nested layers, process them recursively
        if (layer.config && layer.config.layers && Array.isArray(layer.config.layers)) {
          const nestedLayers = layer.config.layers;
          const nestedLog = `🔧 Processing ${nestedLayers.length} nested layers`;
          console.log(nestedLog);
          layer.config.layers = fixLayers(nestedLayers, depth + 1, layer);
          
          // Ensure the Model layer has proper input and output layer references
          if (nestedLayers.length > 0) {
            // Find input and output layers
            const inputLayer = nestedLayers.find((l: ModelLayer) => l.class_name === 'InputLayer');
            const outputLayer = nestedLayers[nestedLayers.length - 1];
            
            if (inputLayer && outputLayer) {
              // Ensure proper connections between input and output layers
              if (!layer.inbound_nodes[0] || !layer.inbound_nodes[0][0]) {
                layer.inbound_nodes[0] = [];
                layer.inbound_nodes[0].push([]);
                layer.inbound_nodes[0][0].push([inputLayer.name || 'input_1']);
              }
              console.log('🔧 Ensured proper input/output layer connections for Model');
            }
          }
        }
        
        return layer;
      }
      
      // For InputLayer, ensure proper structure
      if (layer.class_name === 'InputLayer') {
        if (!layer.config) {
          layer.config = {};
        }
        if (!layer.config.batch_input_shape) {
          layer.config.batch_input_shape = [null, 224, 224, 3]; // Default shape
        }
        if (!layer.name) {
          const inputName = `input_${depth}_${index}`;
          layer.name = inputName;
        }
        const fixedLog = `🔧 Fixed InputLayer structure: ${layer.name}`;
        console.log(fixedLog);
      }
      
      // For other layers, ensure basic structure
      if (!layer.name) {
        const defaultName = `${layer.class_name.toLowerCase()}_${depth}_${index}`;
        layer.name = defaultName;
      }
      
      // Ensure inbound_nodes is properly structured for non-Model layers
      if (!layer.inbound_nodes || layer.inbound_nodes.length === 0) {
        // For non-input layers, create connection to previous layer
        if (index > 0 && parentLayer && parentLayer.config && parentLayer.config.layers) {
          const parentLayers = parentLayer.config.layers;
          const prevLayer = parentLayers[index - 1];
          if (prevLayer) {
            const fallbackName = `${prevLayer.class_name.toLowerCase()}_${depth}_${index - 1}`;
            const layerName = prevLayer.name || fallbackName;
            layer.inbound_nodes = [];
            layer.inbound_nodes.push([]);
            layer.inbound_nodes[0].push([]);
            layer.inbound_nodes[0][0].push([layerName]);
          } else {
            const inputName = `input_${depth}_${index - 1}`;
            layer.inbound_nodes = [];
            layer.inbound_nodes.push([]);
            layer.inbound_nodes[0].push([]);
            layer.inbound_nodes[0][0].push([inputName]);
          }
        } else if (index === 0) {
          // First layer connects to input
          layer.inbound_nodes = [];
          layer.inbound_nodes.push([]);
          layer.inbound_nodes[0].push([]);
          layer.inbound_nodes[0][0].push(['input_1']);
        }
      }
      
      // If the layer has nested layers, fix them too
      if (layer.config && layer.config.layers && Array.isArray(layer.config.layers)) {
        const nestedLayerLog = `🔧 Found nested layers in ${layer.class_name}, processing recursively...`;
        console.log(nestedLayerLog);
        layer.config.layers = fixLayers(layer.config.layers, depth + 1, layer);
      }
      
      return layer;
    });
  };
  
  // Fix layer types in the config
  if (fixedTopology.config && fixedTopology.config.layers) {
    const startLog = `🔧 Starting layer fixing process with ${fixedTopology.config.layers.length} top-level layers`;
    console.log(startLog);
    fixedTopology.config.layers = fixLayers(fixedTopology.config.layers);
    
    // Ensure the top-level model has proper inbound_nodes
    if (!fixedTopology.inbound_nodes) {
      fixedTopology.inbound_nodes = [];
    }
    
    // Ensure proper input/output connections for the entire model
    const layers = fixedTopology.config.layers;
    if (layers.length > 0) {
      const firstLayer = layers[0];
      const lastLayer = layers[layers.length - 1];
      
      // Set up proper model-level connections
      // TensorFlow.js expects inbound_nodes to be an array of arrays of arrays
      if (fixedTopology.inbound_nodes.length === 0) {
        fixedTopology.inbound_nodes.push([]);
      }
      if (fixedTopology.inbound_nodes[0].length === 0) {
        fixedTopology.inbound_nodes[0].push([]);
      }
      if (fixedTopology.inbound_nodes[0][0].length === 0) {
        fixedTopology.inbound_nodes[0][0].push([firstLayer.name || 'input_1']);
      }
      
      console.log('🔧 Ensured proper model-level input/output connections');
    }
  } else {
    console.log('🔧 No config.layers found in topology');
  }
  
  console.log('🔧 Topology fixing completed');
  return fixedTopology;
};

// TypeScript interfaces for model import
interface ModelMetadata {
  tfjsVersion: string;
  tmVersion: string;
  packageVersion: string;
  packageName: string;
  timeStamp: string;
  userMetadata: {
    student_id: string;
    student_name: string;
    sample_count: number;
    accuracy: number;
    training_date: string;
    model_architecture: string;
    total_students: number;
    training_summary: string;
  };
  modelName: string;
  labels: string[];
  imageSize: number;
  modelStructure: {
    hasMobileNet: boolean;
    hasClassifier: boolean;
    inputSize: number[];
    featureSize: number[];
    outputSize: number;
  };
}

interface ModelJson {
  format: string;
  generatedBy: string;
  convertedBy: string | null;
  modelTopology: unknown;
  weightsManifest: Array<{
    paths: string[];
    weights: Array<{
      name: string;
      shape: number[];
      dtype: string;
    }>;
  }>;
}

export interface ImportedModel {
  featureExtractor: tf.LayersModel;
  classifier: tf.LayersModel;
  metadata: ModelMetadata;
  getClassLabels: () => string[];
}

/**
 * Load a local model from uploaded files
 * This function expects the exact files created by the export function:
 * - mobilenet_model.json
 * - mobilenet_weights.bin
 * - classifier_model.json
 * - classifier_weights.bin
 * - metadata.json
 */
export const loadLocalModel = async (files: FileList): Promise<ImportedModel> => {
  console.log('🔄 Starting local model loading process...');
  
  try {
    // Convert FileList to array and organize files by name
    const fileArray = Array.from(files);
    const fileMap = new Map<string, File>();
    
    fileArray.forEach(file => {
      fileMap.set(file.name, file);
    });
    
    console.log('📁 Found files:', Array.from(fileMap.keys()));
    
    // Check for required files
    const requiredFiles = [
      'mobilenet_model.json',
      'mobilenet_weights.bin',
      'classifier_model.json',
      'classifier_weights.bin',
      'metadata.json'
    ];
    
    const missingFiles = requiredFiles.filter(fileName => !fileMap.has(fileName));
    if (missingFiles.length > 0) {
      throw new Error(`Missing required model files. Please ensure all files are present: ${missingFiles.join(', ')}`);
    }
    
    console.log('✅ All required files found');
    
    // Read and parse model files
    const mobileNetJsonText = await fileMap.get('mobilenet_model.json')!.text();
    const mobileNetJson: ModelJson = JSON.parse(mobileNetJsonText);
    
    const classifierJsonText = await fileMap.get('classifier_model.json')!.text();
    const classifierJson: ModelJson = JSON.parse(classifierJsonText);
    
    const metadataText = await fileMap.get('metadata.json')!.text();
    const metadata: ModelMetadata = JSON.parse(metadataText);
    
    console.log('📖 Model files parsed successfully');
    
    // Create File objects for TensorFlow.js loader
    const mobileNetJsonFile = fileMap.get('mobilenet_model.json')!;
    const mobileNetWeightsFile = fileMap.get('mobilenet_weights.bin')!;
    
    // Read and parse the model files
    console.log('🔧 Reading MobileNet model topology...');
    console.log('🔧 MobileNet raw structure:', JSON.stringify(mobileNetJson, null, 2).substring(0, 500) + '...');
    
    console.log('🔧 Reading classifier model topology...');
    console.log('🔧 Classifier raw structure:', JSON.stringify(classifierJson, null, 2).substring(0, 500) + '...');
    
    // Handle different possible JSON structures
    let mobileNetTopology: ModelTopology;
    let classifierTopology: ModelTopology;
    
    // Check if the JSON has a modelTopology property (ModelJson format)
    if (mobileNetJson.modelTopology) {
      console.log('🔧 MobileNet is in ModelJson format, extracting modelTopology');
      // The modelTopology might be a string, so parse it if needed
      if (typeof mobileNetJson.modelTopology === 'string') {
        console.log('🔧 MobileNet modelTopology is a string, parsing it...');
        mobileNetTopology = JSON.parse(mobileNetJson.modelTopology) as unknown as ModelTopology;
      } else {
        mobileNetTopology = mobileNetJson.modelTopology as unknown as ModelTopology;
      }
    } else {
      console.log('🔧 MobileNet is in raw topology format');
      mobileNetTopology = mobileNetJson as unknown as ModelTopology;
    }
    
    if (classifierJson.modelTopology) {
      console.log('🔧 Classifier is in ModelJson format, extracting modelTopology');
      // The modelTopology might be a string, so parse it if needed
      if (typeof classifierJson.modelTopology === 'string') {
        console.log('🔧 Classifier modelTopology is a string, parsing it...');
        classifierTopology = JSON.parse(classifierJson.modelTopology) as unknown as ModelTopology;
      } else {
        classifierTopology = classifierJson.modelTopology as unknown as ModelTopology;
      }
    } else {
      console.log('🔧 Classifier is in raw topology format');
      classifierTopology = classifierJson as unknown as ModelTopology;
    }
    
    // Apply topology fixing with detailed logging
    console.log('🔧 Original MobileNet class_name:', mobileNetTopology.class_name);
    console.log('🔧 MobileNet config structure:', mobileNetTopology.config ? 'Has config' : 'No config');
    if (mobileNetTopology.config) {
      console.log('🔧 MobileNet config layers:', mobileNetTopology.config.layers ? `Has ${mobileNetTopology.config.layers.length} layers` : 'No layers');
    }
    
    console.log('🔧 Original Classifier class_name:', classifierTopology.class_name);
    console.log('🔧 Classifier config structure:', classifierTopology.config ? 'Has config' : 'No config');
    if (classifierTopology.config) {
      console.log('🔧 Classifier config layers:', classifierTopology.config.layers ? `Has ${classifierTopology.config.layers.length} layers` : 'No layers');
    }
    
    mobileNetTopology = fixModelTopology(mobileNetTopology);
    console.log('🔧 Fixed MobileNet class_name:', mobileNetTopology.class_name);
    
    classifierTopology = fixModelTopology(classifierTopology);
    console.log('🔧 Fixed Classifier class_name:', classifierTopology.class_name);
    
    // Log the final topology structure
    console.log('🔧 Final MobileNet topology:', JSON.stringify(mobileNetTopology, null, 2).substring(0, 500) + '...');
    console.log('🔧 Final Classifier topology:', JSON.stringify(classifierTopology, null, 2).substring(0, 500) + '...');
    
    // Apply topology fixing and create fixed JSON files
    console.log('🔧 Applying topology fixing to model files...');
    
    // Use the original weightManifest from the exported files
    const mobileNetWeightManifest = mobileNetJson.weightsManifest || [{
      paths: ['mobilenet_weights.bin'],
      weights: []
    }];
    
    const classifierWeightManifest = classifierJson.weightsManifest || [{
      paths: ['classifier_weights.bin'],
      weights: []
    }];
    
    console.log('🔧 Using original weightManifest structure');
    console.log('🔧 MobileNet weightManifest:', JSON.stringify(mobileNetWeightManifest, null, 2));
    console.log('🔧 Classifier weightManifest:', JSON.stringify(classifierWeightManifest, null, 2));
    
    // Check if weightManifest exists in original files
    console.log('🔧 Original MobileNet has weightsManifest:', !!mobileNetJson.weightsManifest);
    console.log('🔧 Original Classifier has weightsManifest:', !!classifierJson.weightsManifest);
    
    console.log('🔧 Creating fixed JSON files with original weightManifest');
    
    // Alternative approach: Try using the original JSON structure but with fixed topology
    console.log('🔧 Trying alternative approach with original structure...');
    const altMobileNetJson = {
      ...mobileNetJson,
      modelTopology: mobileNetTopology
    };
    
    const altClassifierJson = {
      ...classifierJson,
      modelTopology: classifierTopology
    };
    
    console.log('🔧 Alternative MobileNet JSON keys:', Object.keys(altMobileNetJson));
    console.log('🔧 Alternative Classifier JSON keys:', Object.keys(altClassifierJson));
    console.log('🔧 Alternative MobileNet has weightManifest:', 'weightsManifest' in altMobileNetJson);
    console.log('🔧 Alternative Classifier has weightManifest:', 'weightsManifest' in altClassifierJson);
    
    // Try the alternative approach first
    let fixedMobileNetJson: ModelJson, fixedClassifierJson: ModelJson;
    let useAlternative = false;
    
    try {
      fixedMobileNetJson = altMobileNetJson;
      fixedClassifierJson = altClassifierJson;
      useAlternative = true;
      
      console.log('🔧 Using alternative approach with original structure');
    } catch (altError) {
      console.log('🔧 Alternative approach failed, using standard approach:', altError);
      
      // Fall back to standard approach
      fixedMobileNetJson = {
        modelTopology: mobileNetTopology,
        format: 'layers-model',
        generatedBy: 'AMSUIP-fixed',
        convertedBy: 'TopologyFixer',
        weightsManifest: mobileNetWeightManifest
      };
      
      fixedClassifierJson = {
        modelTopology: classifierTopology,
        format: 'layers-model',
        generatedBy: 'AMSUIP-fixed',
        convertedBy: 'TopologyFixer',
        weightsManifest: classifierWeightManifest
      };
    }
    
    // Log the final JSON structure for debugging
    console.log('🔧 Final MobileNet JSON structure:', JSON.stringify(fixedMobileNetJson, null, 2).substring(0, 1000) + '...');
    console.log('🔧 Final Classifier JSON structure:', JSON.stringify(fixedClassifierJson, null, 2).substring(0, 1000) + '...');
    
    // Create Blob objects for the fixed JSON files
    const fixedMobileNetBlob = new Blob([JSON.stringify(fixedMobileNetJson, null, 2)], { type: 'application/json' });
    const fixedClassifierBlob = new Blob([JSON.stringify(fixedClassifierJson, null, 2)], { type: 'application/json' });
    
    // Create File objects for the fixed models
    const fixedMobileNetFile = new File([fixedMobileNetBlob], 'mobilenet_model.json', { type: 'application/json' });
    const fixedClassifierFile = new File([fixedClassifierBlob], 'classifier_model.json', { type: 'application/json' });
    
    // Log the exact content being passed to TensorFlow.js
    console.log('🔧 MobileNet file content preview:', await fixedMobileNetBlob.text().then(text => text.substring(0, 500) + '...'));
    console.log('🔧 Classifier file content preview:', await fixedClassifierBlob.text().then(text => text.substring(0, 500) + '...'));
    
    // Load models using tf.loadLayersModel with fixed files
    console.log('🔧 Loading MobileNet model from fixed files...');
    const mobileNetJsonHandler = tf.io.browserFiles([fixedMobileNetFile, mobileNetWeightsFile]);
    const mobileNet = await tf.loadLayersModel(mobileNetJsonHandler);
    
    console.log('🔧 Loading Classifier model from fixed files...');
    const classifierWeightsFile = fileMap.get('classifier_weights.bin')!;
    const classifierJsonHandler = tf.io.browserFiles([fixedClassifierFile, classifierWeightsFile]);
    const classifier = await tf.loadLayersModel(classifierJsonHandler);
    
    // Create the imported model object
    const importedModel: ImportedModel = {
      featureExtractor: mobileNet,
      classifier: classifier,
      metadata: metadata,
      getClassLabels: () => metadata.labels
    };
    
    console.log('🎉 Model loading completed successfully!');
    console.log(`📊 Model info: ${metadata.labels.length} classes, ${(metadata.userMetadata.accuracy * 100).toFixed(2)}% accuracy`);
    
    return importedModel;
    
  } catch (error) {
    console.error('❌ Error loading local model:', error);
    throw new Error(`Failed to load local model: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
};

/**
 * Validate that the loaded model structure matches expected format
 */
export const validateImportedModel = (model: ImportedModel): boolean => {
  try {
    // Check if both models exist
    if (!model.featureExtractor || !model.classifier) {
      console.error('❌ Missing feature extractor or classifier');
      return false;
    }
    
    // Check if models have expected input/output shapes
    const featureExtractorOutput = model.featureExtractor.predict(tf.zeros([1, 224, 224, 3])) as tf.Tensor;
    const classifierOutput = model.classifier.predict(featureExtractorOutput) as tf.Tensor;
    
    const expectedOutputSize = model.metadata.labels.length;
    
    if (classifierOutput.shape[1] !== expectedOutputSize) {
      console.error(`❌ Classifier output shape mismatch. Expected ${expectedOutputSize}, got ${classifierOutput.shape[1]}`);
      featureExtractorOutput.dispose();
      classifierOutput.dispose();
      return false;
    }
    
    // Clean up tensors
    featureExtractorOutput.dispose();
    classifierOutput.dispose();
    
    console.log('✅ Model validation passed');
    return true;
    
  } catch (error) {
    console.error('❌ Model validation failed:', error);
    return false;
  }
};

/**
 * Dispose of imported model tensors to free memory
 */
export const disposeImportedModel = (model: ImportedModel): void => {
  try {
    if (model.featureExtractor) {
      model.featureExtractor.dispose();
    }
    if (model.classifier) {
      model.classifier.dispose();
    }
    console.log('🧹 Imported model disposed successfully');
  } catch (error) {
    console.error('❌ Error disposing imported model:', error);
  }
};
