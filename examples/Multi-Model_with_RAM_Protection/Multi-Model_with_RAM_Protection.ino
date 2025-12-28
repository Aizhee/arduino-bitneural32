#include "BitNeural32.h"
#include "feature_extractor.h"  // Lightweight feature model not included
#include "classifier_model.h"   // Classification model not included

void setup() {
    Serial.begin(115200);
    bn_init();
    
    // Configure for low-RAM mode
    bn_set_board_type(BOARD_ESP32);
    bn_set_ram_limit(128000);  // Only 128 KB available
}

void loop() {
    float sensor_input[20];
    float features[64];
    float classification[10];
    
    // Read raw sensor data
    for (int i = 0; i < 20; i++) {
        sensor_input[i] = analogRead(A0) / 1023.0;
    }
    
    // Stage 1: Feature extraction (uses ~50 KB RAM)
    Serial.println("Extracting features...");
    unsigned long t1 = micros();
    bn_run_inference_protected(feature_extractor, sensor_input, features, 128000);
    Serial.print("Feature extraction: ");
    Serial.println((micros() - t1) / 1000.0);
    
    // Stage 2: Classification (uses ~60 KB RAM)
    Serial.println("Classifying...");
    unsigned long t2 = micros();
    bn_run_inference_protected(classifier_model, features, classification, 128000);
    Serial.print("Classification: ");
    Serial.println((micros() - t2) / 1000.0);
    
    // Print result
    int predicted_class = 0;
    for (int i = 1; i < 10; i++) {
        if (classification[i] > classification[predicted_class]) {
            predicted_class = i;
        }
    }
    Serial.print("Predicted: Class ");
    Serial.println(predicted_class);
    
    delay(500);
}