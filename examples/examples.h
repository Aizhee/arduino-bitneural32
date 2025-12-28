// ============================================
// BitNeural32 v2.0 - Arduino IDE Examples
// ============================================

// Example 1: Basic Inference with Metrics
// ----------------------------------------
#include "bitneural.h"
#include "model_data.h"  // Generated with allow_metrics=True

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

// ============================================
// Example 2: LSTM for Sequence Processing
// ============================================

#include "bitneural.h"
#include "lstm_model.h"  // Model with LSTM layer

#define SEQUENCE_LENGTH 10
#define INPUT_SIZE 32
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10

float input_sequence[SEQUENCE_LENGTH][INPUT_SIZE];  // Time series data
float lstm_hidden[HIDDEN_SIZE] = {0};  // Initialize to zero
float lstm_cell[HIDDEN_SIZE] = {0};    // Cell state for LSTM
float output[OUTPUT_SIZE];

void setup() {
    Serial.begin(115200);
    bn_init();
    bn_set_board_type(BOARD_ESP32);
    
    Serial.println("LSTM Sequence Processing Model");
}

void loop() {
    // Process a sequence of 10 timesteps
    unsigned long start = millis();
    
    for (int t = 0; t < SEQUENCE_LENGTH; t++) {
        // Load timestep data (from sensor, file, etc)
        read_timestep_data(input_sequence[t], t);
        
        // Create context for LSTM layer
        bn_context_t lstm_ctx = {
            .input = input_sequence[t],
            .output = output,
            .input_len = INPUT_SIZE,
            .output_len = OUTPUT_SIZE,
            .hidden_state = lstm_hidden,
            .cell_state = lstm_cell,
            .state_size = HIDDEN_SIZE,
            .use_dual_core = 1,
            .ram_limit_bytes = 262144,
            .current_ram_usage = 0
        };
        
        // Run LSTM kernel
        kernel_lstm(&lstm_ctx);
        
        // Process output for this timestep
        Serial.print("Timestep ");
        Serial.print(t);
        Serial.print(": ");
        Serial.println(output[0], 4);  // Print first output
        
        // Hidden and cell states are automatically updated by kernel_lstm()
    }
    
    unsigned long elapsed = millis() - start;
    Serial.print("Full sequence (");
    Serial.print(SEQUENCE_LENGTH);
    Serial.print(" steps): ");
    Serial.print(elapsed);
    Serial.println(" ms");
    
    delay(2000);
}

void read_timestep_data(float* buffer, int timestep) {
    // Example: Read from ADC, sensor, file, etc
    for (int i = 0; i < INPUT_SIZE; i++) {
        buffer[i] = analogRead(A0) / 1023.0;  // Simulated sensor data
    }
}

// ============================================
// Example 3: GRU for Audio Classification
// ============================================

#include "bitneural.h"
#include "audio_gru_model.h"

#define AUDIO_FRAMES 20
#define AUDIO_FEATURES 16
#define GRU_HIDDEN 32
#define NUM_CLASSES 4

// Circular buffer for real-time audio
float audio_buffer[AUDIO_FRAMES][AUDIO_FEATURES];
int buffer_pos = 0;
float gru_hidden[GRU_HIDDEN] = {0};
float output[NUM_CLASSES];

void setup() {
    Serial.begin(115200);
    bn_init();
    bn_set_board_type(BOARD_ESP32_S3);  // Dual-core
    
    Serial.println("Audio Classification with GRU");
}

void loop() {
    // Collect audio frame (simulated)
    float frame[AUDIO_FEATURES];
    collect_audio_frame(frame);
    
    // Shift into circular buffer
    memcpy(audio_buffer[buffer_pos], frame, AUDIO_FEATURES * sizeof(float));
    buffer_pos = (buffer_pos + 1) % AUDIO_FRAMES;
    
    // When buffer full, run inference
    static int frame_count = 0;
    frame_count++;
    
    if (frame_count >= AUDIO_FRAMES) {
        // Process each audio frame through GRU
        for (int t = 0; t < AUDIO_FRAMES; t++) {
            int idx = (buffer_pos + t) % AUDIO_FRAMES;
            
            bn_context_t gru_ctx = {
                .input = audio_buffer[idx],
                .output = output,
                .input_len = AUDIO_FEATURES,
                .output_len = NUM_CLASSES,
                .hidden_state = gru_hidden,
                .cell_state = NULL,  // GRU doesn't use cell state
                .state_size = GRU_HIDDEN,
                .use_dual_core = 1,
                .ram_limit_bytes = 262144,
                .current_ram_usage = 0
            };
            
            kernel_gru(&gru_ctx);
        }
        
        // Final output classification
        int best_class = 0;
        float best_score = output[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (output[i] > best_score) {
                best_score = output[i];
                best_class = i;
            }
        }
        
        Serial.print("Detected class: ");
        Serial.print(best_class);
        Serial.print(" (confidence: ");
        Serial.print(best_score, 3);
        Serial.println(")");
        
        frame_count = 0;
    }
}

void collect_audio_frame(float* frame) {
    // Example: Extract MFCC features from audio
    for (int i = 0; i < AUDIO_FEATURES; i++) {
        frame[i] = analogRead(A0) / 1023.0;  // Simulated
    }
}

// ============================================
// Example 4: Multi-Model Inference with RAM Protection
// ============================================

#include "bitneural.h"
#include "feature_extractor.h"  // Lightweight feature model
#include "classifier_model.h"   // Classification model

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

// ============================================
// Example 5: Real-time Dual-Core Processing
// ============================================

#include "bitneural.h"
#include "model1.h"
#include "model2.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

// Core 0: Data collection
// Core 1: Inference
TaskHandle_t core0_task, core1_task;
float input_buffer[64];
float output_buffer[10];
volatile bool new_data = false;

void core0_collect(void* param) {
    while (1) {
        // Collect sensor data
        for (int i = 0; i < 64; i++) {
            input_buffer[i] = analogRead(A0) / 1023.0;
        }
        new_data = true;
        vTaskDelay(10);  // 10ms
    }
}

void core1_infer(void* param) {
    bn_init();
    bn_set_board_type(BOARD_ESP32_S3);  // Dual-core enabled
    
    while (1) {
        if (new_data) {
            // Run inference on core 1
            bn_run_inference(model1, input_buffer, output_buffer);
            
            // Process results
            Serial.print("Output: ");
            Serial.println(output_buffer[0]);
            
            new_data = false;
        }
        vTaskDelay(5);
    }
}

void setup() {
    Serial.begin(115200);
    
    // Create FreeRTOS tasks
    xTaskCreatePinnedToCore(core0_collect, "Collection", 2048, NULL, 1, &core0_task, 0);
    xTaskCreatePinnedToCore(core1_infer, "Inference", 4096, NULL, 1, &core1_task, 1);
    
    Serial.println("Dual-core processing started!");
}

void loop() {
    // Main task can do other work
    vTaskDelay(1000);
}

// ============================================
// Helper: Check Metrics from Generated Header
// ============================================

/*
To access metrics from your generated model_data.h:

Look for the metrics comment block:

    / *
     *  ============================================
     *  INFERENCE METRICS & PERFORMANCE DATA
     *  ============================================
     *  Target Board:    ESP32-S3
     *  Inference Time:  ~2.5 ms
     *  RAM Usage:       ~2048 bytes
     * /

Use these values to:
1. Set appropriate RAM limits
2. Estimate total latency
3. Verify model fits in available memory
4. Plan multi-model architectures

Example calculation:
- Model 1 inference: 2.5 ms
- Model 2 inference: 1.8 ms
- Total: 4.3 ms (fits in 10ms budget ✓)
*/

// ============================================
// Notes for Arduino IDE Integration
// ============================================

/*
1. File Organization:
   - Copy bitneural.h to sketchbook/libraries/BitNeural32/
   - Copy bitneural.c, kernels.c to same directory
   - Generate model_data.h with Python compiler
   - Include in your sketch

2. Board Configuration in Arduino IDE:
   - Board: "ESP32 Dev Module" or "ESP32-S3-DevKitC"
   - CPU Frequency: 240 MHz (recommended)
   - Flash Size: 4 MB
   - PSRAM: Optional (not required)

3. Memory Tips:
   - Typical inference: 1-4 KB RAM
   - LSTM/GRU: +256-512 bytes per hidden unit
   - Model data: Stored in Flash (const)
   - Use PSRAM if available (bn_set_ram_limit() can account for it)

4. Performance Tips:
   - Use double buffering for continuous inference
   - Run inference on core 1 (let core 0 handle WiFi/I2C)
   - Profile with micros() to find bottlenecks
   - Use inline assembly if needed for critical kernels

5. Debugging:
   - Print timestamps with millis()/micros()
   - Log layer execution times
   - Monitor free heap with esp_get_free_heap_size()
   - Watch for stack overflow (segfault patterns)
*/
