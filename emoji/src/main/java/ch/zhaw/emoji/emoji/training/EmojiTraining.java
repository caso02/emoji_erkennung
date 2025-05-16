package ch.zhaw.emoji.emoji.training;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.listener.TrainingListener;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.repository.Repository;
import ai.djl.training.Trainer;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ai.djl.nn.pooling.Pool;
import ai.djl.nn.Blocks;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.io.IOException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class EmojiTraining {
    private static final Logger logger = LoggerFactory.getLogger(EmojiTraining.class);
    
    /**
     * Verifies that dataset exists and has expected structure
     */
    public boolean verifyDataset(String datasetPath) {
        Path path = Paths.get(datasetPath);
        if (!Files.exists(path)) {
            logger.error("Dataset directory does not exist: {}", datasetPath);
            return false;
        }
        
        // Check for image directory
        Path imagePath = path.resolve("image");
        if (!Files.exists(imagePath) || !Files.isDirectory(imagePath)) {
            logger.error("Image directory not found at: {}", imagePath);
            return false;
        }
        
        // Check for subdirectories (each representing a class)
        try (var subdirs = Files.list(imagePath)) {
            long count = subdirs
                    .filter(Files::isDirectory)
                    .count();
            
            if (count == 0) {
                logger.error("No class subdirectories found in {}", imagePath);
                return false;
            }
            
            logger.info("Found {} emoji class directories", count);
            return true;
        } catch (IOException e) {
            logger.error("Error checking dataset structure: {}", e.getMessage());
            return false;
        }
    }
    
    /**
     * Main training method for emoji recognition model
     */
    public void trainModel(String datasetPath, String modelSavePath) throws Exception {
        logger.info("Starting emoji model training with dataset: {}", datasetPath);
        
        // Verify dataset and create output directory
        if (!verifyDataset(datasetPath)) {
            throw new IllegalArgumentException("Dataset verification failed: " + datasetPath);
        }
        
        // Create model save directory if it doesn't exist
        Path modelDir = Paths.get(modelSavePath);
        if (!Files.exists(modelDir)) {
            Files.createDirectories(modelDir);
            logger.info("Created model save directory: {}", modelDir);
        }
        
        // Load dataset
        Repository repository = Repository.newInstance("image_folder", Paths.get(datasetPath, "image"));
        ImageFolder dataset = ImageFolder.builder()
                .setRepository(repository)
                .addTransform(new Resize(64, 64))  // Resize all images to 64x64
                .addTransform(new ToTensor())      // Convert to tensor format
                .setSampling(32, true)             // Batch size 32
                .build();
        
        logger.info("Preparing dataset...");
        try {
            dataset.prepare();
            logger.info("Dataset prepared successfully with {} classes: {}", 
                    dataset.getSynset().size(), dataset.getSynset());
        } catch (Exception e) {
            logger.error("Failed to prepare dataset: {}", e.getMessage());
            throw e;
        }
        
        // Split into training and validation sets
        RandomAccessDataset[] splits = dataset.randomSplit(8, 2); // 80% training, 20% validation
        RandomAccessDataset trainingSet = splits[0];
        RandomAccessDataset validationSet = splits[1];
        
        logger.info("Training set size: {}, Validation set size: {}", 
                trainingSet.size(), validationSet.size());
        
        // Create model
        Model model = Model.newInstance("emoji-recognition-model");
        
        // Define CNN architecture
        logger.info("Creating CNN model for {} classes", dataset.getSynset().size());
        Block block = createCnnBlock(dataset.getSynset().size());
        model.setBlock(block);
        
        // Configure training parameters
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.001f)).build())
                .addTrainingListeners(TrainingListener.Defaults.logging());
        
        // Train the model
        try (Trainer trainer = model.newTrainer(config)) {
            // Set input shape
            Shape inputShape = new Shape(1, 3, 64, 64); // batch, channels, height, width
            trainer.initialize(inputShape);
            
            logger.info("Starting training for 1 epochs");
            int epochs = 1;
            EasyTrain.fit(trainer, epochs, trainingSet, validationSet);
            
            // Save the trained model
            model.save(modelDir, "emoji-model");
            
            // Save class names (synset)
            try (java.io.PrintWriter writer = new java.io.PrintWriter(modelDir.resolve("synset.txt").toFile())) {
                for (String className : dataset.getSynset()) {
                    writer.println(className);
                }
            }
            
            logger.info("Model training completed and saved to: {}", modelSavePath);
        }
    }
    
    /**
     * Creates the CNN network architecture
     */
    private static Block createCnnBlock(int numClasses) {
        SequentialBlock block = new SequentialBlock();
        
        // Layer 1: Conv -> BatchNorm -> ReLU -> MaxPool
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optStride(new Shape(1, 1))
                .optPadding(new Shape(1, 1))
                .setFilters(32)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        
        // Layer 2: Conv -> BatchNorm -> ReLU -> MaxPool
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optStride(new Shape(1, 1))
                .optPadding(new Shape(1, 1))
                .setFilters(64)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        
        // Layer 3: Conv -> BatchNorm -> ReLU -> MaxPool
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optStride(new Shape(1, 1))
                .optPadding(new Shape(1, 1))
                .setFilters(128)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        
        // Flatten and Fully Connected Layers
        block.add(Blocks.batchFlattenBlock());
        block.add(Linear.builder().setUnits(256).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(numClasses).build());
        
        return block;
    }
    
    /**
     * Command-line method to run training directly
     */
    public static void main(String[] args) {
        try {
            EmojiTraining trainer = new EmojiTraining();
            
            // Default paths
            String datasetPath = "./emojiimage-dataset";
            String modelSavePath = "./trained-models";
            
            // Override with args if provided
            if (args.length >= 1) datasetPath = args[0];
            if (args.length >= 2) modelSavePath = args[1];
            
            trainer.trainModel(datasetPath, modelSavePath);
        } catch (Exception e) {
            logger.error("Training failed", e);
            e.printStackTrace();
        }
    }
}