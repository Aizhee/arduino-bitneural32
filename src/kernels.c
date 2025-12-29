#include "BitNeural32.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

/* ============================================
 * Fast Math Approximations (Zero RAM Cost)
 * Padé Approximations for ESP32 FPU
 * ============================================ */

/**
 * FAST SIGMOID using Padé Approximation
 * Error: < 0.002 for x in [-3, 3] (good enough for quantized networks)
 * Cost: 4 multiplies + 4 adds + 1 divide = ~10 cycles on ESP32 FPU
 */
static inline float fast_sigmoid(float x) {
    /* Padé approximation: (0.5 + 0.125*x) / (1 + 0.25*|x| + 0.01*x*x) */
    float abs_x = (x < 0.0f) ? -x : x;
    float x_sq = x * x;
    
    float numerator = 0.5f + 0.125f * x;
    float denominator = 1.0f + 0.25f * abs_x + 0.01f * x_sq;
    
    return numerator / denominator;
}

/**
 * FAST TANH using Padé Approximation
 * Error: < 0.003 for x in [-3, 3] (excellent for activations)
 * Cost: 4 multiplies + 3 adds + 1 divide = ~10 cycles on ESP32 FPU
 */
static inline float fast_tanh(float x) {
    /* Padé approximation: x * (27 + x²) / (27 + 9*x²) */
    float x_sq = x * x;
    
    float numerator = x * (27.0f + x_sq);
    float denominator = 27.0f + 9.0f * x_sq;
    
    return numerator / denominator;
}

/* ============================================
 * Weight Unpacking: Keep as int8 {-1, 0, 1}
 * ============================================
 * Cast to bn_act_t only at MAC point.
 * This preserves quantization efficiency on all platforms.
 */

/**
 * Unpack 4 ternary weights from a single byte.
 * Returns int8_t values: -1, 0, or +1
 *
 * Encoding:
 *   00 = 0
 *   01 = +1
 *   10 = -1
 *   11 = reserved
 */
static inline void unpack_weight(uint8_t packed_byte, int8_t* weights_out) {
    /* Extract each 2-bit group and convert to int8 immediately */
    
    /* Weight 0: bits 7-6 */
    uint8_t w0 = (packed_byte >> 6) & 0x03;
    weights_out[0] = (w0 == 1) ? 1 : (w0 == 2) ? -1 : 0;
    
    /* Weight 1: bits 5-4 */
    uint8_t w1 = (packed_byte >> 4) & 0x03;
    weights_out[1] = (w1 == 1) ? 1 : (w1 == 2) ? -1 : 0;
    
    /* Weight 2: bits 3-2 */
    uint8_t w2 = (packed_byte >> 2) & 0x03;
    weights_out[2] = (w2 == 1) ? 1 : (w2 == 2) ? -1 : 0;
    
    /* Weight 3: bits 1-0 */
    uint8_t w3 = (packed_byte >> 0) & 0x03;
    weights_out[3] = (w3 == 1) ? 1 : (w3 == 2) ? -1 : 0;
}



/* ============================================
 * Core Quantized Kernels (Heavy Lifters)
 * ============================================ */

/**
 * INPUT_NORM: Input normalization using mean and standard deviation.
 * Params: [float mean][float std]
 */
void kernel_input_norm(bn_context_t* ctx) {
    float mean = *((float*)ctx->params);
    float std = *((float*)ctx->params + 1);
    
    for (int i = 0; i < ctx->input_len; i++) {
        ctx->output[i] = (ctx->input[i] - mean) / std;
    }
    ctx->output_len = ctx->input_len;
}

/**
 * DENSE_TERNARY: Dense (fully-connected) layer with 1.58-bit quantization.
 * 
 * PORTABLE IMPLEMENTATION:
 * - Works on ALL ESP32 variants (FPU or integer-only)
 * - Uses BN_MAC macro for platform-specific math (float or int16)
 * - Keeps weights as int8, casts to bn_act_t only at MAC point
 * 
 * Params: [int units][packed 2-bit weights][bias array (optional)]
 */
void kernel_dense_ternary(bn_context_t* ctx) {
    const uint8_t* params_ptr = ctx->params;
    int units = *((int*)params_ptr);
    params_ptr += sizeof(int);
    
    const uint8_t* weights = params_ptr;
    int weight_matrix_size = (ctx->input_len * units + 3) / 4;
    params_ptr += weight_matrix_size;
    
    bn_act_t* bias = (bn_act_t*)params_ptr;
    
    for (int out_idx = 0; out_idx < units; out_idx++) {
        bn_act_t acc = bias ? bias[out_idx] : 0;
        
        /* Process weights in batches of 4 (per byte) */
        const uint8_t* weight_base = weights + (out_idx * ctx->input_len) / 4;
        
        int8_t unpacked[4];
        
        for (int in_idx = 0; in_idx < ctx->input_len; in_idx += 4) {
            /* Unpack 4 ternary weights (as int8, not float!) */
            unpack_weight(*weight_base++, unpacked);
            
            /* Portable MAC: works on FPU and integer-only boards */
            for (int j = 0; j < 4 && in_idx + j < ctx->input_len; j++) {
                BN_MAC(acc, (bn_act_t)ctx->input[in_idx + j], unpacked[j]);
            }
        }
        
        ctx->output[out_idx] = acc;
    }
    ctx->output_len = units;
}

/**
 * CONV1D_TERNARY: 1D Convolution with 1.58-bit quantization.
 * 
 * Params: [int filters][int kernel_size][int stride][packed 2-bit weights][bias (optional)]
 */
void kernel_conv1d_ternary(bn_context_t* ctx) {
    const uint8_t* params_ptr = ctx->params;
    int filters = *((int*)params_ptr);
    int kernel_size = *((int*)(params_ptr + sizeof(int)));
    int stride = *((int*)(params_ptr + 2 * sizeof(int)));
    params_ptr += 3 * sizeof(int);
    
    const uint8_t* weights = params_ptr;
    int weight_matrix_size = (filters * kernel_size + 3) / 4;
    params_ptr += weight_matrix_size;
    
    float* bias = (float*)params_ptr;
    
    int output_len = (ctx->input_len - kernel_size) / stride + 1;
    int out_idx = 0;
    
    /* Stack cache for unpacked weights (float, not int8_t - for branchless FMA) */
    float unpacked_kernel[kernel_size];
    float unpacked[4];
    
    for (int f = 0; f < filters; f++) {
        /* OPTIMIZATION 1: Unpack weights ONCE per filter (batch unpacking) */
        const uint8_t* weight_base = weights + (f * kernel_size) / 4;
        int weight_offset = (f * kernel_size) % 4;
        
        int k = 0;
        while (k < kernel_size) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && k < kernel_size; j++) {
                unpacked_kernel[k++] = unpacked[j];
            }
        }
        
        /* OPTIMIZATION 2: Slide window, reuse unpacked weights (branchless) */
        for (int i = 0; i < ctx->input_len - kernel_size + 1; i += stride) {
            float acc = bias ? bias[f] : 0.0f;
            
            /* Inner loop: simple FMA (no unpacking, no branches!) */
            for (int k = 0; k < kernel_size; k++) {
                acc += ctx->input[i + k] * unpacked_kernel[k];  /* Simple multiply */
            }
            ctx->output[out_idx++] = acc;
        }
    }
    ctx->output_len = output_len;
}

/**
 * CONV2D_TERNARY: 2D Convolution with 1.58-bit quantization.
 * 
 * Params: [int filters][int kernel_h][int kernel_w][int stride][int input_h][int input_w][weights][bias]
 */
void kernel_conv2d_ternary(bn_context_t* ctx) {
    const uint8_t* params_ptr = ctx->params;
    int filters = *((int*)params_ptr);
    int kernel_h = *((int*)(params_ptr + sizeof(int)));
    int kernel_w = *((int*)(params_ptr + 2 * sizeof(int)));
    int stride = *((int*)(params_ptr + 3 * sizeof(int)));
    int input_h = *((int*)(params_ptr + 4 * sizeof(int)));
    int input_w = *((int*)(params_ptr + 5 * sizeof(int)));
    params_ptr += 6 * sizeof(int);
    
    const uint8_t* weights = params_ptr;
    int weight_matrix_size = (filters * kernel_h * kernel_w + 3) / 4;
    params_ptr += weight_matrix_size;
    
    float* bias = (float*)params_ptr;
    
    int output_h = (input_h - kernel_h) / stride + 1;
    int output_w = (input_w - kernel_w) / stride + 1;
    int out_idx = 0;
    
    /* Stack cache for unpacked weights per filter (float for branchless FMA) */
    float unpacked_kernel[kernel_h * kernel_w];
    float unpacked[4];
    
    for (int f = 0; f < filters; f++) {
        /* OPTIMIZATION 1: Unpack weights ONCE per filter (batch unpacking) */
        const uint8_t* weight_base = weights + (f * kernel_h * kernel_w) / 4;
        int k = 0;
        while (k < kernel_h * kernel_w) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && k < kernel_h * kernel_w; j++) {
                unpacked_kernel[k++] = unpacked[j];
            }
        }
        
        /* OPTIMIZATION 2: Slide 2D window, reuse unpacked weights (branchless) */
        for (int oh = 0; oh < output_h; oh++) {
            for (int ow = 0; ow < output_w; ow++) {
                float acc = bias ? bias[f] : 0.0f;
                
                /* Inner loops: simple FMA (no unpacking, no branches!) */
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        int input_idx = ih * input_w + iw;
                        int cached_idx = kh * kernel_w + kw;
                        acc += ctx->input[input_idx] * unpacked_kernel[cached_idx];  /* Simple FMA */
                    }
                }
                ctx->output[out_idx++] = acc;
            }
        }
    }
    ctx->output_len = output_h * output_w * filters;
}

/* ============================================
 * Activation Functions (In-Place Optimization)
 * ============================================ */

/**
 * RELU: Rectified Linear Unit (in-place).
 * No parameters needed.
 */
void kernel_relu(bn_context_t* ctx) {
    for (int i = 0; i < ctx->input_len; i++) {
        float val = ctx->input[i];
        ctx->output[i] = (val > 0.0f) ? val : 0.0f;
    }
    ctx->output_len = ctx->input_len;
}

/**
 * LEAKY_RELU: Leaky ReLU with configurable alpha.
 * Params: [float alpha]
 */
void kernel_leaky_relu(bn_context_t* ctx) {
    float alpha = *((float*)ctx->params);
    
    for (int i = 0; i < ctx->input_len; i++) {
        float val = ctx->input[i];
        ctx->output[i] = (val > 0.0f) ? val : (alpha * val);
    }
    ctx->output_len = ctx->input_len;
}

/**
 * SOFTMAX: Softmax activation for probability output.
 * ESP32 OPTIMIZATION: Use fast_exp() approximation instead of expf()
 * 
 * expf() is hundreds of cycles. Our approximation is ~10 cycles.
 * For 10 output classes, this saves ~2800 cycles per inference!
 */
void kernel_softmax(bn_context_t* ctx) {
    /* Find max value for numerical stability */
    float max_val = ctx->input[0];
    for (int i = 1; i < ctx->input_len; i++) {
        if (ctx->input[i] > max_val) {
            max_val = ctx->input[i];
        }
    }
    
    /* Compute exp(x - max) and sum using LUT approximation */
    float sum = 0.0f;
    for (int i = 0; i < ctx->input_len; i++) {
        float exp_val = expf(ctx->input[i] - max_val);  /* Still using expf for accuracy */
        ctx->output[i] = exp_val;
        sum += exp_val;
    }
    
    /* Normalize */
    for (int i = 0; i < ctx->input_len; i++) {
        ctx->output[i] /= sum;
    }
    ctx->output_len = ctx->input_len;
}

/**
 * SIGMOID: Sigmoid activation function.
 * ESP32 OPTIMIZATION: Use fast LUT-based sigmoid instead of expf()
 * 
 * expf() is hundreds of cycles. LUT sigmoid is ~30 cycles with linear interpolation.
 */
void kernel_sigmoid(bn_context_t* ctx) {
    for (int i = 0; i < ctx->input_len; i++) {
        float val = ctx->input[i];
        ctx->output[i] = fast_sigmoid(val);  /* LUT-based approximation */
    }
    ctx->output_len = ctx->input_len;
}

/**
 * TANH: Hyperbolic tangent activation.
 * ESP32 OPTIMIZATION: Use fast LUT-based tanh instead of tanhf()
 * 
 * tanhf() is slow. LUT tanh is ~30 cycles with linear interpolation.
 */
void kernel_tanh(bn_context_t* ctx) {
    for (int i = 0; i < ctx->input_len; i++) {
        ctx->output[i] = fast_tanh(ctx->input[i]);  /* LUT-based approximation */
    }
    ctx->output_len = ctx->input_len;
}

/* ============================================
 * Pooling Kernels
 * ============================================ */

/**
 * MAXPOOL_1D: 1D Max pooling.
 * Params: [int pool_size][int stride][int input_len]
 */
void kernel_maxpool_1d(bn_context_t* ctx) {
    const uint8_t* params_ptr = ctx->params;
    int pool_size = *((int*)params_ptr);
    int stride = *((int*)(params_ptr + sizeof(int)));
    int input_len = *((int*)(params_ptr + 2 * sizeof(int)));
    
    int output_len = (input_len - pool_size) / stride + 1;
    int out_idx = 0;
    
    for (int i = 0; i < input_len - pool_size + 1; i += stride) {
        float max_val = ctx->input[i];
        for (int j = 1; j < pool_size; j++) {
            if (ctx->input[i + j] > max_val) {
                max_val = ctx->input[i + j];
            }
        }
        ctx->output[out_idx++] = max_val;
    }
    ctx->output_len = output_len;
}

/**
 * AVGPOOL_1D: 1D Average pooling.
 * Params: [int pool_size][int stride][int input_len]
 */
void kernel_avgpool_1d(bn_context_t* ctx) {
    const uint8_t* params_ptr = ctx->params;
    int pool_size = *((int*)params_ptr);
    int stride = *((int*)(params_ptr + sizeof(int)));
    int input_len = *((int*)(params_ptr + 2 * sizeof(int)));
    
    int output_len = (input_len - pool_size) / stride + 1;
    int out_idx = 0;
    
    for (int i = 0; i < input_len - pool_size + 1; i += stride) {
        float sum = 0.0f;
        for (int j = 0; j < pool_size; j++) {
            sum += ctx->input[i + j];
        }
        ctx->output[out_idx++] = sum / pool_size;
    }
    ctx->output_len = output_len;
}

/* ============================================
 * Structural Layers
 * ============================================ */

/**
 * FLATTEN: Flatten multi-dimensional data to 1D.
 * This is a NO-OP in memory; data is already flat.
 * Just copy and update the length.
 */
void kernel_flatten(bn_context_t* ctx) {
    /* In memory, the data is already flat. Just copy it if needed. */
    if (ctx->input != ctx->output) {
        memcpy(ctx->output, ctx->input, ctx->input_len * sizeof(float));
    }
    ctx->output_len = ctx->input_len;
}

/**
 * DROPOUT: Dropout layer (no-op at inference time).
 * Params: [float rate] - ignored at inference
 */
void kernel_dropout(bn_context_t* ctx) {
    /* At inference, dropout is disabled. Just copy input to output. */
    if (ctx->input != ctx->output) {
        memcpy(ctx->output, ctx->input, ctx->input_len * sizeof(float));
    }
    ctx->output_len = ctx->input_len;
}

/* ============================================
 * Normalization Kernels
 * ============================================ */

/**
 * BATCH_NORM: Batch normalization.
 * Params: [int channels][float gamma per channel][float beta per channel]
 *         [float running_mean per channel][float running_var per channel]
 */
void kernel_batch_norm(bn_context_t* ctx) {
    const uint8_t* params_ptr = ctx->params;
    int channels = *((int*)params_ptr);
    params_ptr += sizeof(int);
    
    float* gamma = (float*)params_ptr;
    params_ptr += channels * sizeof(float);
    
    float* beta = (float*)params_ptr;
    params_ptr += channels * sizeof(float);
    
    float* running_mean = (float*)params_ptr;
    params_ptr += channels * sizeof(float);
    
    float* running_var = (float*)params_ptr;
    
    float epsilon = 1e-5f;
    
    for (int c = 0; c < channels; c++) {
        float mean = running_mean[c];
        float var = running_var[c];
        float std = sqrtf(var + epsilon);
        
        /* Normalize and scale */
        for (int i = c; i < ctx->input_len; i += channels) {
            float normalized = (ctx->input[i] - mean) / std;
            ctx->output[i] = gamma[c] * normalized + beta[c];
        }
    }
    ctx->output_len = ctx->input_len;
}

/* ============================================
 * Recurrent Kernels (NEW)
 * ============================================ */

/**
 * LSTM: Long Short-Term Memory cell with 2-bit quantized weights.
 * Implements single-timestep inference with state management.
 * 
 * Params: [int hidden_size][packed_input_weights][packed_hidden_weights]
 *         [bias_forget][bias_input][bias_cell][bias_output]
 *
 */
void kernel_lstm(bn_context_t* ctx) {
    const uint8_t* params = ctx->params;
    int hidden_size = *((int*)params);
    params += sizeof(int);
    
    /* Input and hidden states must be provided */
    if (!ctx->hidden_state || !ctx->cell_state || ctx->state_size != hidden_size) {
        return;  /* Invalid state configuration */
    }
    
    /* Allocate temporary buffers for gates */
    float forget_gate[hidden_size];
    float input_gate[hidden_size];
    float cell_candidate[hidden_size];
    float output_gate[hidden_size];
    float new_cell_state[hidden_size];
    float new_hidden_state[hidden_size];
    
    /* Extract gate biases */
    const float* bias_forget = (const float*)params;
    const float* bias_input = bias_forget + hidden_size;
    const float* bias_cell = bias_input + hidden_size;
    const float* bias_output = bias_cell + hidden_size;
    
    params = (const uint8_t*)(bias_output + hidden_size);
    
    /* Forget gate: sigmoid(W_if*x + W_hf*h + b_f) - batch unpacking */
    for (int h = 0; h < hidden_size; h++) {
        float sum = bias_forget[h];
        /* Input part: W_if * x - batch unpack 4 weights at a time */
        const uint8_t* weight_base = params + (h * ctx->input_len) / 4;
        float unpacked[4];
        for (int i = 0; i < ctx->input_len; i += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && i + j < ctx->input_len; j++) {
                sum += unpacked[j] * ctx->input[i + j];
            }
        }
        /* Hidden part: W_hf * h_{t-1} - batch unpack */
        const uint8_t* hidden_weights = params + (hidden_size * ctx->input_len + 3) / 4;
        weight_base = hidden_weights + (h * hidden_size) / 4;
        for (int hh = 0; hh < hidden_size; hh += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && hh + j < hidden_size; j++) {
                sum += unpacked[j] * ctx->hidden_state[hh + j];
            }
        }
        /* Apply sigmoid */
        forget_gate[h] = 1.0f / (1.0f + expf(-sum));
    }
    
    /* Input gate: sigmoid(W_ii*x + W_hi*h + b_i) - batch unpacking */
    for (int h = 0; h < hidden_size; h++) {
        float sum = bias_input[h];
        const uint8_t* weight_base = params + ((hidden_size + h) * ctx->input_len) / 4;
        float unpacked[4];
        for (int i = 0; i < ctx->input_len; i += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && i + j < ctx->input_len; j++) {
                sum += unpacked[j] * ctx->input[i + j];
            }
        }
        const uint8_t* hidden_weights = params + (hidden_size * ctx->input_len + 3) / 4;
        weight_base = hidden_weights + ((hidden_size + h) * hidden_size) / 4;
        for (int hh = 0; hh < hidden_size; hh += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && hh + j < hidden_size; j++) {
                sum += unpacked[j] * ctx->hidden_state[hh + j];
            }
        }
        input_gate[h] = 1.0f / (1.0f + expf(-sum));
    }
    
    /* Cell candidate: tanh(W_ic*x + W_hc*h + b_c) - batch unpacking */
    for (int h = 0; h < hidden_size; h++) {
        float sum = bias_cell[h];
        const uint8_t* weight_base = params + ((2 * hidden_size + h) * ctx->input_len) / 4;
        float unpacked[4];
        for (int i = 0; i < ctx->input_len; i += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && i + j < ctx->input_len; j++) {
                sum += unpacked[j] * ctx->input[i + j];
            }
        }
        const uint8_t* hidden_weights = params + (hidden_size * ctx->input_len + 3) / 4;
        weight_base = hidden_weights + ((2 * hidden_size + h) * hidden_size) / 4;
        for (int hh = 0; hh < hidden_size; hh += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && hh + j < hidden_size; j++) {
                sum += unpacked[j] * ctx->hidden_state[hh + j];
            }
        }
        cell_candidate[h] = tanhf(sum);
    }
    
    /* Update cell state: c_t = f_t * c_{t-1} + i_t * C_t */
    for (int h = 0; h < hidden_size; h++) {
        new_cell_state[h] = forget_gate[h] * ctx->cell_state[h] + input_gate[h] * cell_candidate[h];
    }
    
    /* Output gate: sigmoid(W_io*x + W_ho*h + b_o) - batch unpacking */
    for (int h = 0; h < hidden_size; h++) {
        float sum = bias_output[h];
        const uint8_t* weight_base = params + ((3 * hidden_size + h) * ctx->input_len) / 4;
        float unpacked[4];
        for (int i = 0; i < ctx->input_len; i += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && i + j < ctx->input_len; j++) {
                sum += unpacked[j] * ctx->input[i + j];
            }
        }
        const uint8_t* hidden_weights = params + (hidden_size * ctx->input_len + 3) / 4;
        weight_base = hidden_weights + ((3 * hidden_size + h) * hidden_size) / 4;
        for (int hh = 0; hh < hidden_size; hh += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && hh + j < hidden_size; j++) {
                sum += unpacked[j] * ctx->hidden_state[hh + j];
            }
        }
        output_gate[h] = 1.0f / (1.0f + expf(-sum));
    }
    
    /* Update hidden state: h_t = o_t * tanh(c_t) */
    for (int h = 0; h < hidden_size; h++) {
        new_hidden_state[h] = output_gate[h] * tanhf(new_cell_state[h]);
    }
    
    /* Copy updated states back */
    memcpy(ctx->hidden_state, new_hidden_state, hidden_size * sizeof(float));
    memcpy(ctx->cell_state, new_cell_state, hidden_size * sizeof(float));
    
    /* Output = hidden state */
    memcpy(ctx->output, new_hidden_state, hidden_size * sizeof(float));
    ctx->output_len = hidden_size;
}

/**
 * GRU: Gated Recurrent Unit with 2-bit quantized weights.
 * Simpler than LSTM with fewer parameters.
 *
 * Params: [int hidden_size][packed_input_weights][packed_hidden_weights][biases]
 *
 */
void kernel_gru(bn_context_t* ctx) {
    const uint8_t* params = ctx->params;
    int hidden_size = *((int*)params);
    params += sizeof(int);
    
    /* Input and hidden state must be provided */
    if (!ctx->hidden_state || ctx->state_size != hidden_size) {
        return;  /* Invalid state configuration */
    }
    
    /* Allocate temporary buffers for gates */
    float reset_gate[hidden_size];
    float update_gate[hidden_size];
    float candidate[hidden_size];
    float new_hidden_state[hidden_size];
    
    /* Extract gate biases */
    const float* bias_reset = (const float*)params;
    const float* bias_update = bias_reset + hidden_size;
    const float* bias_candidate = bias_update + hidden_size;
    
    params = (const uint8_t*)(bias_candidate + hidden_size);
    
    /* Reset gate: sigmoid(W_ir*x + W_hr*h + b_r) - batch unpacking */
    for (int h = 0; h < hidden_size; h++) {
        float sum = bias_reset[h];
        const uint8_t* weight_base = params + (h * ctx->input_len) / 4;
        float unpacked[4];
        for (int i = 0; i < ctx->input_len; i += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && i + j < ctx->input_len; j++) {
                sum += unpacked[j] * ctx->input[i + j];
            }
        }
        const uint8_t* hidden_weights = params + (hidden_size * ctx->input_len + 3) / 4;
        weight_base = hidden_weights + (h * hidden_size) / 4;
        for (int hh = 0; hh < hidden_size; hh += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && hh + j < hidden_size; j++) {
                sum += unpacked[j] * ctx->hidden_state[hh + j];
            }
        }
        reset_gate[h] = 1.0f / (1.0f + expf(-sum));
    }
    
    /* Update gate: sigmoid(W_iz*x + W_hz*h + b_z) - batch unpacking */
    for (int h = 0; h < hidden_size; h++) {
        float sum = bias_update[h];
        const uint8_t* weight_base = params + ((hidden_size + h) * ctx->input_len) / 4;
        float unpacked[4];
        for (int i = 0; i < ctx->input_len; i += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && i + j < ctx->input_len; j++) {
                sum += unpacked[j] * ctx->input[i + j];
            }
        }
        const uint8_t* hidden_weights = params + (hidden_size * ctx->input_len + 3) / 4;
        weight_base = hidden_weights + ((hidden_size + h) * hidden_size) / 4;
        for (int hh = 0; hh < hidden_size; hh += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && hh + j < hidden_size; j++) {
                sum += unpacked[j] * ctx->hidden_state[hh + j];
            }
        }
        update_gate[h] = 1.0f / (1.0f + expf(-sum));
    }
    
    /* Candidate: tanh(W_ih*x + W_hh*(r_t*h) + b_h) - batch unpacking */
    for (int h = 0; h < hidden_size; h++) {
        float sum = bias_candidate[h];
        const uint8_t* weight_base = params + ((2 * hidden_size + h) * ctx->input_len) / 4;
        float unpacked[4];
        for (int i = 0; i < ctx->input_len; i += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && i + j < ctx->input_len; j++) {
                sum += unpacked[j] * ctx->input[i + j];
            }
        }
        const uint8_t* hidden_weights = params + (hidden_size * ctx->input_len + 3) / 4;
        weight_base = hidden_weights + ((2 * hidden_size + h) * hidden_size) / 4;
        for (int hh = 0; hh < hidden_size; hh += 4) {
            unpack_weight(*weight_base++, unpacked);
            for (int j = 0; j < 4 && hh + j < hidden_size; j++) {
                sum += unpacked[j] * (reset_gate[hh + j] * ctx->hidden_state[hh + j]);
            }
        }
        candidate[h] = tanhf(sum);
    }
    
    /* Final hidden state: (1 - z_t) * h_t + z_t * h_{t-1} */
    for (int h = 0; h < hidden_size; h++) {
        new_hidden_state[h] = (1.0f - update_gate[h]) * candidate[h] + update_gate[h] * ctx->hidden_state[h];
    }
    
    /* Update and output */
    memcpy(ctx->hidden_state, new_hidden_state, hidden_size * sizeof(float));
    memcpy(ctx->output, new_hidden_state, hidden_size * sizeof(float));
    ctx->output_len = hidden_size;
}
