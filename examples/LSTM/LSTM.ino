#include "BitNeural32.h"
#include "lstm_model.h"  // Model with LSTM layer not included

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