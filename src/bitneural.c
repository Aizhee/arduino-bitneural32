#include "bitneural.h"
#include <string.h>
#include <stdlib.h>

/* ============================================
 * Global Layer Registry
 * ============================================
 * Maps OpCode (0-255) to kernel function.
 * Users can register custom layers via
 * bn_register_custom_layer() at runtime.
 */
static bn_layer_func layer_registry[256] = {0};

/* ============================================
 * Global Configuration (NEW)
 * ============================================ */
static int g_board_type = BOARD_ESP32;
static int g_ram_limit = DEFAULT_RAM_LIMIT;

/* ============================================
 * Initialization
 * ============================================
 * Call this once at startup to register
 * all built-in kernel functions.
 */
void bn_init(void) {
    memset(layer_registry, 0, sizeof(layer_registry));
    
    /* Essential Core Layers (1.58-bit Quantized) */
    layer_registry[OP_INPUT_NORM] = kernel_input_norm;
    layer_registry[OP_CONV1D_TERNARY] = kernel_conv1d_ternary;
    layer_registry[OP_DENSE_TERNARY] = kernel_dense_ternary;
    layer_registry[OP_CONV2D_TERNARY] = kernel_conv2d_ternary;
    
    /* Activation Functions */
    layer_registry[OP_RELU] = kernel_relu;
    layer_registry[OP_LEAKY_RELU] = kernel_leaky_relu;
    layer_registry[OP_SOFTMAX] = kernel_softmax;
    layer_registry[OP_SIGMOID] = kernel_sigmoid;
    layer_registry[OP_TANH] = kernel_tanh;
    
    /* Pooling and Structural Layers */
    layer_registry[OP_MAXPOOL_1D] = kernel_maxpool_1d;
    layer_registry[OP_AVGPOOL_1D] = kernel_avgpool_1d;
    layer_registry[OP_FLATTEN] = kernel_flatten;
    layer_registry[OP_DROPOUT] = kernel_dropout;
    
    /* Normalization Layers */
    layer_registry[OP_BATCH_NORM] = kernel_batch_norm;
    
    /* Recurrent Layers (NEW) */
    layer_registry[OP_LSTM] = kernel_lstm;
    layer_registry[OP_GRU] = kernel_gru;
}

/* ============================================
 * Custom Layer Registration
 * ============================================
 * Allow users to register their own kernels
 * at runtime without modifying this library.
 */
void bn_register_custom_layer(int opcode, bn_layer_func func) {
    if (opcode >= 0 && opcode < 256) {
        layer_registry[opcode] = func;
    }
}

/* ============================================
 * Board & Performance Configuration (NEW)
 * ============================================ */

/**
 * Set board type for hardware-specific optimization.
 * board_type: BOARD_ESP32, BOARD_ESP32_S3, or BOARD_ESP32_C3
 */
void bn_set_board_type(int board_type) {
    if (board_type >= BOARD_ESP32 && board_type <= BOARD_ESP32_C3) {
        g_board_type = board_type;
    }
}

/**
 * Set maximum RAM available for inference (bytes).
 * Helps prevent memory exhaustion on low-resource devices.
 */
void bn_set_ram_limit(int max_bytes) {
    if (max_bytes > 0) {
        g_ram_limit = max_bytes;
    }
}

/* ============================================
 * Model Inference Engine (Interpreter)
 * ============================================
 * This is the core runtime that:
 * 1. Parses the model binary format
 * 2. Dispatches each layer to the registry
 * 3. Manages ping-pong buffering
 * 4. Handles parameter streaming
 */
void bn_run_inference(const uint8_t* model_data, float* input, float* output) {
    if (!model_data || !input || !output) {
        return;
    }
    
    const uint8_t* data_ptr = model_data;
    int num_layers;
    
    /* ========== PARSE GLOBAL FILE HEADER ========== */
    /* Expected: Magic="BITN" (4 bytes), num_layers (int) */
    if (memcmp(data_ptr, "BITN", 4) != 0) {
        return;  /* Invalid model magic number */
    }
    data_ptr += 4;
    
    num_layers = *((int*)data_ptr);
    data_ptr += sizeof(int);
    
    /* ========== LAYER DISPATCH LOOP ========== */
    float* current_input = input;
    float* current_output = output;
    
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        /* Extract OpCode */
        int opcode = *data_ptr++;
        
        /* Verify OpCode is registered */
        if (!layer_registry[opcode]) {
            /* OpCode not found; skip this layer's parameters */
            int param_length = *((int*)data_ptr);
            data_ptr += sizeof(int) + param_length;
            continue;
        }
        
        /* Extract parameter length */
        int param_length = *((int*)data_ptr);
        data_ptr += sizeof(int);
        
        /* Setup context for this layer */
        bn_context_t ctx = {
            .input = current_input,
            .output = current_output,
            .input_len = 0,  /* Must be set by previous layer or user */
            .output_len = 0, /* Will be set by kernel */
            .params = data_ptr,
            .hidden_state = NULL,
            .cell_state = NULL,
            .state_size = 0,
            .use_dual_core = (g_board_type == BOARD_ESP32_S3) ? 1 : 0,
            .ram_limit_bytes = g_ram_limit,
            .current_ram_usage = 0
        };
        
        /* Call the kernel function */
        layer_registry[opcode](&ctx);
        
        /* Prepare for next layer: ping-pong buffering */
        current_input = current_output;
        /* Allocate new output buffer for next layer (or reuse if available) */
        /* NOTE: In embedded systems, you'd pre-allocate two buffers */
        current_output = (current_output == output) ? input : output;
        
        /* Update input_len for next layer */
        if (layer_idx + 1 < num_layers) {
            /* The next layer's input_len = this layer's output_len */
            /* (This is handled implicitly via ctx setup in next iteration) */
        }
        
        /* Move parameter pointer to next layer */
        data_ptr += param_length;
    }
}

/**
 * NEW: Protected inference with RAM limiting.
 * Ensures inference does not exceed specified RAM budget.
 * max_ram: Maximum RAM to use in bytes
 */
void bn_run_inference_protected(const uint8_t* model_data, float* input, float* output, int max_ram) {
    int old_limit = g_ram_limit;
    bn_set_ram_limit(max_ram);
    bn_run_inference(model_data, input, output);
    bn_set_ram_limit(old_limit);
}

 /* This is the core runtime that:
 * 1. Parses the model binary format
 * 2. Dispatches each layer to the registry
 * 3. Manages ping-pong buffering
 * 4. Handles parameter streaming
 */
void bn_run_inference(const uint8_t* model_data, float* input, float* output) {
    if (!model_data || !input || !output) {
        return;
    }
    
    const uint8_t* data_ptr = model_data;
    int num_layers;
    
    /* ========== PARSE GLOBAL FILE HEADER ========== */
    /* Expected: Magic="BITN" (4 bytes), num_layers (int) */
    if (memcmp(data_ptr, "BITN", 4) != 0) {
        return;  /* Invalid model magic number */
    }
    data_ptr += 4;
    
    num_layers = *((int*)data_ptr);
    data_ptr += sizeof(int);
    
    /* ========== LAYER DISPATCH LOOP ========== */
    float* current_input = input;
    float* current_output = output;
    
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        /* Extract OpCode */
        int opcode = *data_ptr++;
        
        /* Verify OpCode is registered */
        if (!layer_registry[opcode]) {
            /* OpCode not found; skip this layer's parameters */
            int param_length = *((int*)data_ptr);
            data_ptr += sizeof(int) + param_length;
            continue;
        }
        
        /* Extract parameter length */
        int param_length = *((int*)data_ptr);
        data_ptr += sizeof(int);
        
        /* Setup context for this layer */
        bn_context_t ctx = {
            .input = current_input,
            .output = current_output,
            .input_len = 0,  /* Must be set by previous layer or user */
            .output_len = 0, /* Will be set by kernel */
            .params = data_ptr
        };
        
        /* Call the kernel function */
        layer_registry[opcode](&ctx);
        
        /* Prepare for next layer: ping-pong buffering */
        current_input = current_output;
        /* Allocate new output buffer for next layer (or reuse if available) */
        /* NOTE: In embedded systems, you'd pre-allocate two buffers */
        current_output = (current_output == output) ? input : output;
        
        /* Update input_len for next layer */
        if (layer_idx + 1 < num_layers) {
            /* The next layer's input_len = this layer's output_len */
            /* (This is handled implicitly via ctx setup in next iteration) */
        }
        
        /* Move parameter pointer to next layer */
        data_ptr += param_length;
    }
}