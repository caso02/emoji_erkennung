package ch.zhaw.emoji.emoji.util;

import ch.zhaw.emoji.emoji.training.EmojiTraining;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Utility class to help with initial setup and model training
 */
@Component
public class EmojiSetupUtil implements ApplicationRunner {
    private static final Logger logger = LoggerFactory.getLogger(EmojiSetupUtil.class);
    
    private final EmojiTraining emojiTraining;
    
    public EmojiSetupUtil(EmojiTraining emojiTraining) {
        this.emojiTraining = emojiTraining;
    }
    
    @Override
    public void run(ApplicationArguments args) throws Exception {
        logger.info("Checking emoji application setup...");
        
        // Check for required directories
        createRequiredDirectories();
        
        // Check if we need to train the model
        if (shouldTrainModel(args)) {
            trainModel();
        }
    }
    
    /**
     * Creates all required directories for the application
     */
    private void createRequiredDirectories() {
        String[] directories = {
            "./trained-models",
            "./emojiimage-dataset",
            "./emojiimage-dataset/image"
        };
        
        for (String dir : directories) {
            Path path = Paths.get(dir);
            if (!Files.exists(path)) {
                try {
                    Files.createDirectories(path);
                    logger.info("Created directory: {}", path);
                } catch (Exception e) {
                    logger.error("Failed to create directory {}: {}", path, e.getMessage());
                }
            }
        }
    }
    
    /**
     * Checks if model training should be performed
     */
    private boolean shouldTrainModel(ApplicationArguments args) {
        // Check if force training is requested
        if (args.containsOption("train")) {
            return true;
        }else{
            return false;
        }
        
    }
    
    /**
     * Executes model training process
     */
    private void trainModel() {
        try {
            logger.info("Starting emoji model training...");
            emojiTraining.trainModel("./emojiimage-dataset", "./trained-models");
            logger.info("Emoji model training completed successfully.");
        } catch (Exception e) {
            logger.error("Failed to train emoji model: {}", e.getMessage());
        }
    }
    
    /**
     * Utility method to check dataset structure
     */
    public void checkDatasetStructure() {
        try {
            Path datasetPath = Paths.get("./emojiimage-dataset");
            if (!Files.exists(datasetPath)) {
                logger.error("Dataset directory not found: {}", datasetPath);
                return;
            }
            
            // Check CSV file
            Path csvFile = datasetPath.resolve("full_emoji.csv");
            if (Files.exists(csvFile)) {
                logger.info("Found emoji metadata CSV: {}", csvFile);
            } else {
                logger.warn("Emoji metadata CSV not found: {}", csvFile);
            }
            
            // Check image directory
            Path imageDir = datasetPath.resolve("image");
            if (!Files.exists(imageDir)) {
                logger.error("Image directory not found: {}", imageDir);
                return;
            }
            
            // List subdirectories
            File[] subdirs = imageDir.toFile().listFiles(File::isDirectory);
            if (subdirs == null || subdirs.length == 0) {
                logger.warn("No emoji class directories found in {}", imageDir);
                return;
            }
            
            logger.info("Found {} emoji class directories:", subdirs.length);
            for (File dir : subdirs) {
                File[] files = dir.listFiles(f -> f.isFile() && f.getName().endsWith(".png"));
                int fileCount = files != null ? files.length : 0;
                logger.info(" - {} ({} images)", dir.getName(), fileCount);
            }
            
        } catch (Exception e) {
            logger.error("Error checking dataset structure: {}", e.getMessage());
        }
    }
}