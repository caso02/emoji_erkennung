package ch.zhaw.emoji.emoji.controller;

import ch.zhaw.emoji.emoji.service.EmojiRecognizer;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.awt.image.BufferedImage;
import java.io.IOException;
import javax.imageio.ImageIO;

@RestController
@RequestMapping("/api/emoji")
public class EmojiController {
    
    private final EmojiRecognizer emojiRecognizer = new EmojiRecognizer();
    
    @GetMapping("/ping")
    public String ping() {
        return "Emoji-Erkennungs-App läuft!";
    }
    
    @PostMapping("/analyze")
    public String analyzeEmoji(@RequestParam("image") MultipartFile file) {
        try {
            // Konvertiere die hochgeladene Datei in ein DJL Image
            BufferedImage bufferedImage = ImageIO.read(file.getInputStream());
            Image djlImage = ImageFactory.getInstance().fromImage(bufferedImage);
            
            // Führe die Prediction durch
            Classifications result = emojiRecognizer.predict(djlImage);
            
            // Gib das Ergebnis als JSON zurück
            return result.toJson();
            
        } catch (IOException e) {
            return "{\"error\": \"Fehler bei der Bildverarbeitung: " + e.getMessage() + "\"}";
        } catch (Exception e) {
            return "{\"error\": \"Analyse fehlgeschlagen: " + e.getMessage() + "\"}";
        }
    }
}