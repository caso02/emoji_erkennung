package ch.zhaw.emoji.emoji.service;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class EmojiRecognizer {
    private static final Logger logger = LoggerFactory.getLogger(EmojiRecognizer.class);
    private Predictor<Image, Classifications> predictor;

    public EmojiRecognizer() {
        try {
            // Prüfe, ob das trainierte Modell existiert
            Path modelPath = Paths.get("./trained-models/emoji-model");
            Path synsetPath = Paths.get("./trained-models/synset.txt");
            
            if (Files.exists(modelPath) && Files.exists(synsetPath)) {
                logger.info("Lade trainiertes Emoji-Erkennungsmodell...");
                // Verwende dein trainiertes Modell
                Criteria<Image, Classifications> criteria = Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelPath(modelPath)
                        .optOption("synsetPath", synsetPath.toString())
                        .optDevice(Device.cpu())
                        .optProgress(new ProgressBar())
                        .build();
                
                ZooModel<Image, Classifications> model = criteria.loadModel();
                predictor = model.newPredictor();
            } else {
                logger.warn("Kein trainiertes Emoji-Modell gefunden. Verwende vortrainiertes ResNet für Demo-Zwecke.");
                // Verwende ein vortrainiertes Modell als Fallback
                Criteria<Image, Classifications> criteria = Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optFilter("backbone", "resnet18") // Verwende resnet18 - leichter als resnet50
                        .optFilter("dataset", "imagenet")  // ImageNet dataset
                        .optDevice(Device.cpu())
                        .optProgress(new ProgressBar())
                        .build();
                
                ZooModel<Image, Classifications> model = criteria.loadModel();
                predictor = model.newPredictor();
            }
            
            logger.info("Emoji-Erkennungsmodell erfolgreich geladen");
            
        } catch (ModelNotFoundException | MalformedModelException | IOException e) {
            logger.error("Fehler beim Laden des Modells", e);
            throw new RuntimeException("Emoji-Erkenner konnte nicht initialisiert werden", e);
        }
    }
    
    public Classifications predict(Image image) throws Exception {
        if (predictor == null) {
            throw new IllegalStateException("Predictor wurde nicht initialisiert");
        }
        return predictor.predict(image);
    }
}