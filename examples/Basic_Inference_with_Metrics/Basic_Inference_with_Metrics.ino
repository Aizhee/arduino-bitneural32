#include "BitNeural32.h"
#include "model_data.h"  // Generated model headers not included

void setup() {
    Serial.begin(115200);
    bn_init();
    
    // Set board type for optimization
    bn_set_board_type(BOARD_ESP32_S3);
    
    // Limit RAM to 256 KB (protects against buffer overflow)
    bn_set_ram_limit(262144);
    
    Serial.println("BitNeural32 initialized!");
    Serial.println("Board: ESP32-S3");
    Serial.println("Metrics embedded in model_data.h (see comments)");
}

void loop() {
    // Example 10-element input
    float input[10] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    float output[10];
    
    unsigned long start = micros();
    
    // Run inference with RAM protection
    bn_run_inference_protected(model_data, input, output, 262144);
    
    unsigned long elapsed = micros() - start;
    
    // Print results
    Serial.print("Inference time: ");
    Serial.print(elapsed / 1000.0);
    Serial.println(" ms");
    
    Serial.println("Output:");
    for (int i = 0; i < 10; i++) {
        Serial.print("  [");
        Serial.print(i);
        Serial.print("]: ");
        Serial.println(output[i], 6);
    }
    
    delay(1000);
}
