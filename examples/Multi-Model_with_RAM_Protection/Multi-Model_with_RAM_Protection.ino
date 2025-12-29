/* ============================================
 * BitNeural32 Multi-Model with INT8 Example
 * ============================================
 * 
 * Demonstrates ultra-low memory inference using
 * int8 activation datatype (Q3.4 fixed-point).
 * 
 * INT8 Benefits:
 *   - Accumulators: 8-bit (vs 32-bit float)
 *   - RAM usage: ~4x lower than float
 *   - Speed: Minimal FPU usage, cache-friendly
 * 
 * Trade-off:
 *   - Limited numeric range: ±7.9375 (Q3.4)
 *   - Best for simple models, Tier 1 only
 *   - Good for ESP32-C2 (120 MHz, 276 KB SRAM)
 */

/* OPTIONAL: Set activation datatype before including BitNeural32.h */
#define BN_ACTIVATION_MODE BN_ACTIVATION_INT8

#include "BitNeural32.h"
#include "feature_extractor.h"  // Lightweight feature model not included
#include "classifier_model.h"   // Classification model not included

void setup() {
    Serial.begin(115200);
    bn_init();
    
    // Configure for ultra-low-RAM mode with int8
    // With int8, we can run on devices with <256 KB SRAM
    bn_set_board_type(BOARD_ESP32);
    bn_set_ram_limit(64000);  // Only 64 KB available (int8 makes this possible)
    
    Serial.println("BitNeural32 INT8 Multi-Model Example");
    Serial.println("Using Q3.4 fixed-point accumulators");
    Serial.println("RAM Budget: 64 KB");
}

void loop() {
    float sensor_input[20];
    float features[64];
    float classification[10];
    
    // Read raw sensor data
    for (int i = 0; i < 20; i++) {
        sensor_input[i] = analogRead(A0) / 1023.0;
    }
    
    // Stage 1: Feature extraction (uses ~25 KB RAM with int8)
    Serial.println("Extracting features (int8 mode)...");
    unsigned long t1 = micros();
    int status1 = bn_run_inference_protected(feature_extractor, sensor_input, features, 64000);
    unsigned long elapsed1 = (micros() - t1) / 1000;
    Serial.print("Feature extraction: ");
    Serial.print(elapsed1);
    Serial.println(" ms");
    
    if (status1 != BN_SUCCESS) {
        Serial.print("ERROR: Feature extraction failed with code ");
        Serial.println(status1);
        return;
    }
    
    // Stage 2: Classification (uses ~30 KB RAM with int8)
    Serial.println("Classifying (int8 mode)...");
    unsigned long t2 = micros();
    int status2 = bn_run_inference_protected(classifier_model, features, classification, 64000);
    unsigned long elapsed2 = (micros() - t2) / 1000;
    Serial.print("Classification: ");
    Serial.print(elapsed2);
    Serial.println(" ms");
    
    if (status2 != BN_SUCCESS) {
        Serial.print("ERROR: Classification failed with code ");
        Serial.println(status2);
        return;
    }
    
    // Print result
    int predicted_class = 0;
    for (int i = 1; i < 10; i++) {
        if (classification[i] > classification[predicted_class]) {
            predicted_class = i;
        }
    }
    Serial.print("Predicted: Class ");
    Serial.print(predicted_class);
    Serial.print(" (confidence: ");
    Serial.print(classification[predicted_class]);
    Serial.println(")");
    
    Serial.print("Total inference time: ");
    Serial.print(elapsed1 + elapsed2);
    Serial.println(" ms");
    Serial.println("---");
    
    delay(1000);
}