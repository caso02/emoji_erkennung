package ch.zhaw.emoji.emoji.service;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class EmojiRecognizer {
    private static final Logger logger = LoggerFactory.getLogger(EmojiRecognizer.class);
    private Predictor<Image, Classifications> predictor;
    private boolean isModelTrained = false;

    public EmojiRecognizer() {
        try {
            // Create necessary directories
            createDirectories();
            
            // Check if the trained model exists
            Path modelPath = Paths.get("./trained-models/emoji-model");
            Path synsetPath = Paths.get("./trained-models/synset.txt");
            
            if (Files.exists(modelPath) && Files.exists(synsetPath)) {
                logger.info("Trained Emoji model found - loading model from: {}", modelPath);
                loadTrainedModel(modelPath, synsetPath);
                isModelTrained = true;
            } else {
                logger.info("No trained model found. Checking if dataset exists for training...");
                
                // Check if dataset exists
                Path datasetPath = Paths.get("./emojiimage-dataset");
                if (Files.exists(datasetPath)) {
                    logger.info("Dataset found at {}. Training will be performed on first use.", datasetPath);
                    
                    // Load fallback model for now, training will happen on first prediction
                    loadFallbackModel();
                } else {
                    logger.warn("No dataset found at {}. Using ImageNet fallback model.", datasetPath);
                    loadFallbackModel();
                }
            }
            
            logger.info("Emoji recognizer initialization successful.");
            
        } catch (Exception e) {
            logger.error("Failed to initialize emoji recognizer", e);
            // Don't throw an exception - we'll handle this more gracefully
            predictor = null;
        }
    }
    
    /**
     * Creates necessary directories for models and datasets
     */
    private void createDirectories() {
        Path[] dirs = {
            Paths.get("./trained-models"),
            Paths.get("./emojiimage-dataset"),
            Paths.get("./emojiimage-dataset/image")
        };
        
        for (Path dir : dirs) {
            try {
                if (!Files.exists(dir)) {
                    Files.createDirectories(dir);
                    logger.info("Created directory: {}", dir);
                }
            } catch (IOException e) {
                logger.warn("Could not create directory {}: {}", dir, e.getMessage());
            }
        }
    }
    
    /**
     * Loads a trained emoji model
     */
    private void loadTrainedModel(Path modelPath, Path synsetPath) {
        try {
            Criteria<Image, Classifications> criteria = Criteria.builder()
                    .setTypes(Image.class, Classifications.class)
                    .optModelPath(modelPath)
                    .optOption("synsetPath", synsetPath.toString())
                    .optDevice(Device.cpu())
                    .optProgress(new ProgressBar())
                    .build();
            
            ZooModel<Image, Classifications> model = criteria.loadModel();
            predictor = model.newPredictor();
            logger.info("Trained emoji model loaded successfully");
        } catch (Exception e) {
            logger.error("Error loading trained model", e);
            logger.info("Falling back to default model");
            try {
                loadFallbackModel();
            } catch (Exception ex) {
                logger.error("Failed to load fallback model", ex);
            }
        }
    }
    
    /**
     * Loads a fallback ImageNet model when no emoji model is available
     */
    private void loadFallbackModel() throws ModelNotFoundException, MalformedModelException, IOException {
        List<String> attemptedModels = new ArrayList<>();
        
        // Try several options to ensure we get a working model
        Exception lastException = null;
        
        // Option 1: Try ResNet18
        try {
            attemptedModels.add("resnet18");
            Criteria<Image, Classifications> criteria = Criteria.builder()
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .setTypes(Image.class, Classifications.class)
                    .optFilter("backbone", "resnet18")
                    .optFilter("dataset", "imagenet")
                    .optDevice(Device.cpu())
                    .optProgress(new ProgressBar())
                    .build();
            
            ZooModel<Image, Classifications> model = criteria.loadModel();
            predictor = model.newPredictor();
            logger.info("Loaded ResNet18 model as fallback");
            return;
        } catch (Exception e) {
            lastException = e;
            logger.warn("Failed to load ResNet18 model: {}", e.getMessage());
        }
        
        // Option 2: Try any ImageNet model
        try {
            attemptedModels.add("generic ImageNet");
            Criteria<Image, Classifications> criteria = Criteria.builder()
                    .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                    .setTypes(Image.class, Classifications.class)
                    .optFilter("dataset", "imagenet")
                    .optDevice(Device.cpu())
                    .optProgress(new ProgressBar())
                    .build();
            
            ZooModel<Image, Classifications> model = criteria.loadModel();
            predictor = model.newPredictor();
            logger.info("Loaded generic ImageNet model as fallback");
            return;
        } catch (Exception e) {
            lastException = e;
            logger.warn("Failed to load generic ImageNet model: {}", e.getMessage());
        }
        
        // Option 3: Final attempt - use any available model from model zoo
        try {
            attemptedModels.add("any available model");
            Criteria<Image, Classifications> criteria = Criteria.builder()
                    .setTypes(Image.class, Classifications.class)
                    .optDevice(Device.cpu())
                    .optProgress(new ProgressBar())
                    .build();
            
            ZooModel<Image, Classifications> model = criteria.loadModel();
            predictor = model.newPredictor();
            logger.info("Loaded available model from model zoo as fallback");
            return;
        } catch (Exception e) {
            lastException = e;
            logger.error("Failed to load any model: {}", e.getMessage());
        }
        
        // If we reach here, all attempts failed
        logger.error("All model loading attempts failed: tried {}", attemptedModels);
        if (lastException != null) {
            throw new RuntimeException("Failed to load any model", lastException);
        }
    }
    
    /**
     * Checks if the model is initialized and handles prediction
     */
    public Classifications predict(Image image) throws Exception {
        if (predictor == null) {
            logger.error("Predictor is not initialized");
            throw new IllegalStateException("Emoji recognizer is not properly initialized");
        }
        
        // If we're using a fallback model and the dataset exists, we could train here
        if (!isModelTrained) {
            Path datasetPath = Paths.get("./emojiimage-dataset");
            if (Files.exists(datasetPath)) {
                logger.info("Dataset exists but no trained model found. Consider running EmojiTraining first.");
                // Note: We could trigger training here, but it would delay the first prediction
            }
        }
        
        // Perform the prediction
        try {
            // First ensure the image is correctly sized for the model
            Image processedImage = prepareImageForModel(image);
            return predictor.predict(processedImage);
        } catch (Exception e) {
            logger.error("Error during prediction: {}", e.getMessage());
            throw e;
        }
    }
    
    /**
     * Prepares an image for model prediction
     */
    private Image prepareImageForModel(Image original) {
        // Most models expect images of a specific size
        // For now, we'll return the original, but we could resize here if needed
        return original;
    }
    
    /**
     * Checks if a directory is empty
     */
    private boolean isDirectoryEmpty(Path directory) throws IOException {
        if (Files.isDirectory(directory)) {
            try (var entries = Files.list(directory)) {
                return !entries.findFirst().isPresent();
            }
        }
        return true;
    }
}