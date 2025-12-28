#ifndef BITNEURAL_H
#define BITNEURAL_H

#include <stdint.h>

/* ============================================
 * OpCode Table: Blueprint for Layers
 * ============================================ */

/* Essential Core Layers (1.58-bit Quantized) */
#define OP_INPUT_NORM       0   /* Input normalization (mean/std) */
#define OP_CONV1D_TERNARY   1   /* 1D Convolution with 2-bit weights */
#define OP_DENSE_TERNARY    2   /* Dense layer with 2-bit weights */
#define OP_CONV2D_TERNARY   3   /* 2D Convolution with 2-bit weights */

/* Activation Functions */
#define OP_RELU             10  /* Rectified Linear Unit */
#define OP_LEAKY_RELU       11  /* Leaky ReLU */
#define OP_SOFTMAX          12  /* Softmax for probabilities */
#define OP_SIGMOID          13  /* Sigmoid activation */
#define OP_TANH             14  /* Hyperbolic tangent */

/* Pooling and Structural Layers */
#define OP_MAXPOOL_1D       20  /* 1D Max pooling */
#define OP_AVGPOOL_1D       21  /* 1D Average pooling */
#define OP_FLATTEN          22  /* Flatten (no-op in memory) */
#define OP_DROPOUT          23  /* Dropout (ignored at inference) */

/* Normalization Layers */
#define OP_BATCH_NORM       30  /* Batch normalization */

/* Recurrent Layers (NEW) */
#define OP_LSTM             40  /* LSTM cell with 2-bit quantized weights */
#define OP_GRU              41  /* GRU cell with 2-bit quantized weights */

/* Custom/User-defined Layer */
#define OP_CUSTOM           255 /* Custom user-defined layer */

/* ============================================
 * Board & Performance Configuration
 * ============================================ */

/* ESP32 Board Types for optimization */
#define BOARD_ESP32         0   /* Original ESP32 (single core) */
#define BOARD_ESP32_S3      1   /* ESP32-S3 (dual core, SIMD) */
#define BOARD_ESP32_C3      2   /* ESP32-C3 (single core) */

/* Default RAM limits (bytes) */
#define DEFAULT_RAM_LIMIT   262144  /* 256 KB - typical available ESP32 RAM */

/* ============================================
 * Context Structure
 * ============================================ */

typedef struct {
    float* input;           /* Input buffer pointer */
    float* output;          /* Output buffer pointer */
    int input_len;          /* Length of input */
    int output_len;         /* Length of output (must be set by kernel) */
    const uint8_t* params;  /* Binary blob for layer parameters */
    
    /* NEW: Recurrent state management */
    float* hidden_state;    /* Hidden state buffer (for LSTM/GRU) */
    float* cell_state;      /* Cell state buffer (for LSTM) */
    int state_size;         /* Size of hidden/cell state */
    
    /* NEW: Dual-core and RAM protection */
    int use_dual_core;      /* 1 = use dual core, 0 = single core */
    int ram_limit_bytes;    /* RAM budget for this layer (0 = unlimited) */
    int current_ram_usage;  /* Track RAM usage during inference */
} bn_context_t;

/* ============================================
 * Function Pointer Type for Plugin System
 * ============================================ */

typedef void (*bn_layer_func)(bn_context_t* ctx);

/* ============================================
 * Core API Functions
 * ============================================ */

void bn_init(void);
void bn_register_custom_layer(int opcode, bn_layer_func func);
void bn_run_inference(const uint8_t* model_data, float* input, float* output);

/* NEW: Dual-core and RAM limiting API */
void bn_set_board_type(int board_type);
void bn_set_ram_limit(int max_bytes);
void bn_run_inference_protected(const uint8_t* model_data, float* input, float* output, int max_ram);

/* ============================================
 * Kernel Function Declarations
 * ============================================ */

/* Core quantized kernels */
void kernel_input_norm(bn_context_t* ctx);
void kernel_conv1d_ternary(bn_context_t* ctx);
void kernel_dense_ternary(bn_context_t* ctx);
void kernel_conv2d_ternary(bn_context_t* ctx);

/* Activation kernels */
void kernel_relu(bn_context_t* ctx);
void kernel_leaky_relu(bn_context_t* ctx);
void kernel_softmax(bn_context_t* ctx);
void kernel_sigmoid(bn_context_t* ctx);
void kernel_tanh(bn_context_t* ctx);

/* Pooling kernels */
void kernel_maxpool_1d(bn_context_t* ctx);
void kernel_avgpool_1d(bn_context_t* ctx);
void kernel_flatten(bn_context_t* ctx);
void kernel_dropout(bn_context_t* ctx);

/* Normalization kernels */
void kernel_batch_norm(bn_context_t* ctx);

/* NEW: Recurrent kernels */
void kernel_lstm(bn_context_t* ctx);
void kernel_gru(bn_context_t* ctx);

#endif // BITNEURAL_H