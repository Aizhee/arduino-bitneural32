#include "BitNeural32.h"
#include "model1.h" // Generated model headers not included
#include "model2.h" // Generated model headers not included
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
