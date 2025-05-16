package ch.zhaw.emoji.emoji.controller;

import ch.zhaw.emoji.emoji.service.EmojiRecognizer;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.awt.image.BufferedImage;
import java.io.IOException;
import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RestController
@RequestMapping("/api/emoji")
public class EmojiController {
    
    private static final Logger logger = LoggerFactory.getLogger(EmojiController.class);
    private final EmojiRecognizer emojiRecognizer;
    
    @Autowired
    public EmojiController(EmojiRecognizer emojiRecognizer) {
        this.emojiRecognizer = emojiRecognizer;
        logger.info("EmojiController initialized");
    }
    
    @GetMapping("/ping")
    public String ping() {
        return "Emoji recognition app is running!";
    }
    
    @GetMapping("/status")
    public ResponseEntity<String> getStatus() {
        return ResponseEntity.ok("Emoji recognition service is active and ready");
    }
    
    @PostMapping("/analyze")
    public ResponseEntity<?> analyzeEmoji(@RequestParam("image") MultipartFile file) {
        logger.info("Received image analysis request. File size: {} bytes", file.getSize());
        
        if (file.isEmpty()) {
            logger.warn("Empty file received");
            return ResponseEntity.badRequest().body("{\"error\": \"Uploaded file is empty\"}");
        }
        
        try {
            // Check file type
            String contentType = file.getContentType();
            if (contentType == null || !contentType.startsWith("image/")) {
                logger.warn("Invalid file type: {}", contentType);
                return ResponseEntity.badRequest().body("{\"error\": \"Only image files are supported\"}");
            }
            
            // Convert uploaded file to DJL Image
            BufferedImage bufferedImage = ImageIO.read(file.getInputStream());
            if (bufferedImage == null) {
                logger.error("Failed to read image from uploaded file");
                return ResponseEntity.badRequest().body("{\"error\": \"Could not read image from uploaded file\"}");
            }
            
            // Process with DJL
            Image djlImage = ImageFactory.getInstance().fromImage(bufferedImage);
            logger.info("Image converted successfully. Width: {}, Height: {}", 
                    djlImage.getWidth(), djlImage.getHeight());
            
            // Perform the prediction
            Classifications result = emojiRecognizer.predict(djlImage);
            logger.info("Prediction successful. Top class: {}", 
                    result.best().getClassName());
            
            // Return the result
            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(result.toJson());
            
        } catch (IOException e) {
            logger.error("Image processing error", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("{\"error\": \"Image processing error: " + e.getMessage() + "\"}");
        } catch (Exception e) {
            logger.error("Analysis failed", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("{\"error\": \"Analysis failed: " + e.getMessage() + "\"}");
        }
    }
}