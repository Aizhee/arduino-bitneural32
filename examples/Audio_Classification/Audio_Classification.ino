#define BN_ACTIVATION_MODE BN_ACTIVATION_FLOAT

#include "BitNeural32.h"
#include "audio_gru_model.h" // Generated model headers not included

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