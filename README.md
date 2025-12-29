# BitNeural32: 1.58-bit Neural Network Inference Engine for ESP32


**BitNeural32** is a lightweight, general-purpose neural network inference engine optimized for the ESP32 microcontroller. It implements 1.58-bit quantization (weights: {-1, 0, 1}) with float32 activations for efficient deep learning inference on embedded devices. Fully compatible with Arduino IDE.

> See also: [BitNeural32 Python Compiler Library](https://github.com/aizhee/python-bitneural32)

## Key Features

✨ **1.58-bit Quantization**: Weights packed as 2-bit values (4 weights per byte) using ternary {-1, 0, 1}  
✨ **Low Memory Footprint**: Model weights stored in Flash memory; activations use minimal RAM  
✨ **Flexible Architecture**: Interpreter-based design supports 15+ layer types  
✨ **Extensible**: Plugin system allows users to register custom layers at runtime  
✨ **Python Compiler**: Automated Keras→BitNeural conversion with binary model generation  
✨ **Optimized Kernels**: Addition/subtraction-only arithmetic (no floating-point multiplication for weights)

---

## Feature Tier Support Matrix

BitNeural32 supports three feature tiers based on **floating-point hardware support**. Architectural choices are made explicit to ensure portability.

### **Understanding FPU Hardware Differences**

| Chip | FPU | Speed | Math Path | Tier A | Tier B | Tier C |
|------|-----|-------|-----------|--------|--------|--------|
| **ESP32** | ✅ 32-bit | 240 MHz, dual-core | Native float | ✅ Fast | ✅ Fast | ✅ Rec |
| **ESP32-S2** | ✅ 32-bit | 240 MHz, single-core | Native float | ✅ Fast | ✅ Fast | ⚠️ Slow |
| **ESP32-S3** | ✅ 32-bit + Vector | 240 MHz, dual-core | SIMD accelerated | ✅ Very Fast | ✅ Very Fast | ✅ Rec |
| **ESP32-C3** | ❌ None | 160 MHz, single-core | Software float | ✅ Medium | ⚠️ Slow | ❌ Very Slow |
| **ESP32-C2** | ❌ None | 120 MHz, single-core | Software float | ✅ Medium | ⚠️ Slow | ❌ Very Slow |

**Key Insight**: On C3/C2, every float multiply is **software-emulated** (~50 cycles). Integer math is faster.

---

### **Tier 1: Universal (Integer-Only Baseline)**

✅ **Works on**: All ESP32 variants (including C3, C2)  
📦 **Compile with**: `-D BN_ACTIVATION_TYPE=int16_t` (optional, default is float)  
⚡ **Performance**: Best on C3/C2, good on all boards  

**Features**:
- Dense, Conv1D, Conv2D with 1.58-bit quantization
- ReLU, Leaky ReLU (branch-based, no float needed)
- Max Pool, Avg Pool (integer accumulation)
- Input normalization
- Flat activation type (int16_t with Q7.8 fixed-point)

**Example Use Case**: IoT edge inference, power-constrained devices, C3/C2 deployments

**Compilation Example**:
```bash
# Build for ESP32-C3 (no FPU)
idf.py -D BN_ACTIVATION_TYPE=int16_t build
```

---

### **Tier 2: FPU-Optimized (Float Activations)**

✅ **Works on**: ESP32, ESP32-S2, ESP32-S3  
⚠️ **Not recommended for**: ESP32-C3, ESP32-C2 (software float is slow)  
📦 **Compile with**: Default (or `-D BN_ACTIVATION_TYPE=float`)  

**Additional Features**:
- Softmax (requires float exponentiation)
- Sigmoid, Tanh (with fast approximations)
- Batch Normalization
- Dropout (inference no-op, uses float)

**Example Use Case**: Classification with probabilities, modern models

---

### **Tier 3: Advanced (Recurrent Networks)**

✅ **Fully supported on**: ESP32, ESP32-S3  
⚠️ **Experimental on**: ESP32-C3, ESP32-C2 (will be very slow, test throughly)  
❌ **Not recommended on**: C3/C2 (LSTM/GRU are FPU-heavy)  

**Additional Features**:
- LSTM (stateful, 4 float gates per timestep)
- GRU (stateful, 3 float gates per timestep)
- Time-series and sequential processing

**Example Use Case**: Speech recognition, activity detection, time-series forecasting

---

### **Board Decision Table**

| Goal | Best Choice | Why |
|------|-------------|-----|
| **Maximum compatibility** | Tier 1 (int16) | Runs on all ESP32, fast on C3/C2 |
| **Balanced performance** | Tier 2 + ESP32/S3 | Full feature set, proven performance |
| **Sequential models** | Tier 3 + ESP32-S3 | Dual-core, 512 KB SRAM, SIMD help |
| **Ultra-low power** | Tier 1 + C3/C2 | Smallest, lowest clock, integer math |
| **Production (unknown board)** | Tier 1 default | Guaranteed to work, can upgrade features later |

---

## Feature Tier Support Matrix

BitNeural32 supports three feature tiers across different ESP32 variants. Choose based on your board and model complexity.

### **Tier A: Core Inference** (All ESP32 variants)
✅ **Recommended for**: General-purpose inference, classification, regression  
✅ **Fully Supported On**: ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C2  
✅ **Features**:
- Dense (fully connected) layers with 1.58-bit quantization
- 1D & 2D Convolution with 1.58-bit quantization
- ReLU, Leaky ReLU activations
- Max Pool, Average Pool (1D)
- Input normalization

**Example Use Case**: Audio classification, sensor data analysis, simple image recognition

---

### **Tier B: Advanced Activations & Normalization** (ESP32, S2, S3 only)
⚠️ **Not Recommended for**: ESP32-C3, ESP32-C2 (performance not guaranteed)  
✅ **Fully Supported On**: ESP32, ESP32-S2, ESP32-S3  
✅ **Additional Features**:
- Softmax (multi-class classification)
- Sigmoid (binary classification/gate)
- Tanh (bounded non-linearity)
- Batch Normalization
- Dropout (inference no-op)

**Example Use Case**: Multi-label classification, probabilistic models

---

### **Tier C: Recurrent Neural Networks** (ESP32, ESP32-S3 recommended)
⚠️ **Experimental**: Limited testing on C3/C2  
✅ **Recommended For**: ESP32, ESP32-S3  
🔶 **Use With Caution On**: ESP32-C3, ESP32-C2 (may exceed RAM or require careful model sizing)  
✅ **Additional Features**:
- LSTM (Long Short-Term Memory) with stateful inference
- GRU (Gated Recurrent Unit) with stateful inference
- Sequential/time-series processing

**Example Use Case**: Speech recognition, time-series forecasting, activity detection

---

### **Board Capability Reference**

| Board | Tier A | Tier B | Tier C | Notes |
|-------|--------|--------|--------|-------|
| **ESP32** | ✅ Full | ✅ Full | ✅ Recommended | 240 MHz, dual core, 512 KB SRAM |
| **ESP32-S2** | ✅ Full | ✅ Full | ⚠️ Limited | 240 MHz, single core, 320 KB SRAM |
| **ESP32-S3** | ✅ Full | ✅ Full | ✅ Recommended | 240 MHz, dual core, 512 KB SRAM, SIMD |
| **ESP32-C3** | ✅ Full | ⚠️ Caution | ⚠️ Experimental | 160 MHz, single core, 400 KB SRAM |

---

## Project Structure

```
arduino-bitneural32/
├── src/
│   ├── bitneural.c          # Inference interpreter engine
│   ├── BitNeural32.h          # Core API and definitions
│   └── kernels.c            # 15+ layer implementations
├── examples/
│   └── examples.h           # Example usage and sketches
├── library.properties       # Arduino IDE library metadata
├── CMakeLists.txt           # Build configuration
└── README.md                # This file
```

## Installation

### Arduino IDE

1. Download or clone this repository
2. Copy the `BitNeural32` folder to your Arduino libraries directory:
   - Windows: `Documents\Arduino\libraries\`
   - macOS: `~/Documents/Arduino/libraries/`
   - Linux: `~/Arduino/libraries/`
3. Restart Arduino IDE
4. Library will appear under Sketch → Include Library → BitNeural32

or

**install it on the `arduino library manager`, just search for `"BitNeural32"`**

---

## OpCode Table: Supported Layers

| Tier | OpCode | Layer Name | Parameters | Notes |
|------|--------|-----------|------------|-------|
| **A** | 0 | INPUT_NORM | mean, std | Input preprocessing |
| **A** | 1 | CONV1D_TERNARY | filters, kernel_size, stride, weights | 1.58-bit quantized 1D conv |
| **A** | 2 | DENSE_TERNARY | units, weights, bias | 1.58-bit quantized dense |
| **A** | 3 | CONV2D_TERNARY | filters, kernel_h, kernel_w, stride, weights | 1.58-bit quantized 2D conv |
| **A** | 10 | RELU | — | Rectified Linear Unit |
| **A** | 11 | LEAKY_RELU | alpha | Leaky ReLU with slope |
| **B** | 12 | SOFTMAX | — | Softmax (multi-class classification) |
| **B** | 13 | SIGMOID | — | Sigmoid activation (binary/gate) |
| **B** | 14 | TANH | — | Hyperbolic tangent |
| **A** | 20 | MAXPOOL_1D | pool_size, stride | 1D max pooling |
| **A** | 21 | AVGPOOL_1D | pool_size, stride | 1D average pooling |
| **A** | 22 | FLATTEN | — | No-op (memory already flat) |
| **B** | 23 | DROPOUT | rate | No-op at inference |
| **B** | 30 | BATCH_NORM | channels, gamma, beta, mean, var | Batch normalization |
| **C** | 40 | LSTM | hidden_size, weights, bias | LSTM cell with 2-bit quantized weights (stateful) |
| **C** | 41 | GRU | hidden_size, weights, bias | GRU cell with 2-bit quantized weights (stateful) |
| **—** | 255 | CUSTOM | custom_id | User-defined custom layer |

---

## Deployment Guide

### Step 1: Genenrate your model

> Learn more at [Python Compiler library](https://github.com/Aizhee/python-bitneural32/blob/main/README.md#installation)

### Step 2: Run on ESP32

#### Using Arduino IDE

[Install the library](#installation)

When using Arduino IDE, place the generated `model_data.h` file in the same directory as your sketch (`.ino` file). Arduino IDE will automatically include it during compilation.

**Directory structure**:
```
My Sketches/
├── MyNeuralNetworkProject/
│   ├── MyNeuralNetworkProject.ino
│   └── model_data.h          ← Place generated header here
└── ...
```

Then include it in your sketch:

```c
//forces activation function to use float
#define BN_ACTIVATION_MODE BN_ACTIVATION_FLOAT 

#include "BitNeural32.h"
#include "model_data.h"

void app_main() {
    bn_init();
    bn_set_board_type(BOARD_ESP32_S3);  // Dual-core
    
    float input[10] = {...};
    float output[10];
    
    int status = bn_run_inference(model_data, input, output);
    
    if (status == BN_SUCCESS) {
        printf("Prediction: %f\n", output[0]);
    } else {
        // Handle error
        printf("Inference failed with code: %d\n", status);
        if (status == BN_ERR_INVALID_OPCODE) {
            printf("Model corruption or mismatched bitneural32 version\n");
        }
    }
}
```

#### Error Handling

All inference functions return **status codes** for debugging:

```c
#define BN_SUCCESS                  0   /* Success */
#define BN_ERR_INVALID_MODEL       -1   /* Bad magic number */
#define BN_ERR_NULL_POINTER        -2   /* NULL input/output/model */
#define BN_ERR_INVALID_OPCODE      -3   /* Unknown layer type */
#define BN_ERR_RAM_EXCEEDED        -4   /* RAM budget exceeded */
#define BN_ERR_TENSOR_SIZE_MISMATCH -5  /* Dimension mismatch */
```

---

## Configuration: Activation Datatype

BitNeural32 uses **`bitneural_config.h`** for Arduino-friendly configuration. This lets you choose the numeric representation (float vs. integer) used for accumulators.

### Three Configuration Patterns

**Pattern A: Edit the Config Header (Recommended for Arduino IDE)**

```c
/* In bitneural_config.h */
#define BN_ACTIVATION_MODE BN_ACTIVATION_INT16
```

Then compile normally. No flags needed.

**Pattern B: Define Before Include (Advanced Arduino Users)**

In your sketch (.ino file):
```cpp
#define BN_ACTIVATION_MODE BN_ACTIVATION_INT16
#include <bitneural.h>

void setup() {
    // Rest of code...
}
```

**Pattern C: Compiler Flag (PlatformIO / ESP-IDF)**

In `platformio.ini`:
```ini
build_flags = -D BN_ACTIVATION_MODE=BN_ACTIVATION_INT16
```

Or in ESP-IDF:
```bash
idf.py -D BN_ACTIVATION_MODE=BN_ACTIVATION_INT16 build
```

---

### Datatype Options

```c
#define BN_ACTIVATION_FLOAT   0  /* 32-bit float (default, FPU-optimized) */
#define BN_ACTIVATION_INT8    1  /* 8-bit int (ultra-low memory, Q3.4 fixed-point) */
#define BN_ACTIVATION_INT16   2  /* 16-bit int (integer-only, Q7.8 fixed-point) */
#define BN_ACTIVATION_INT32   3  /* 32-bit int (higher precision, Q15.16 fixed-point) */
```

| Datatype | Best For | Memory | Speed | Precision | Range |
|----------|----------|--------|-------|-----------|-------|
| **FLOAT** | ESP32/S2/S3 with FPU | Higher | Fast | High | ±∞ |
| **INT8** | Ultra-constrained (C2, Tier 1 only) | Lowest | Very Fast | Low (Q3.4) | ±7.9375 |
| **INT16** | C3/C2 or power-saving | Lower | Fast (no FPU) | Medium (Q7.8) | ±127.99 |
| **INT32** | High precision needs | Highest | Medium | Very High (Q15.16) | ±32767.99 |

**Why Compile-Time?**
- Affects kernel function signatures (parameter types)
- Affects struct memory layout
- Affects code size and ABI
- Runtime switching would require unsafe void pointers or unions

**Why Runtime for Functions?**
- Activation functions (ReLU, Sigmoid, etc.) are always selected per-layer via OpCode
- Professional pattern (TFLite, ONNX, etc.)

---

---

### How This Works: The BN_MAC Abstraction

All kernels use the **`BN_MAC()`** macro, which expands at compile-time to the correct math path:

```c
/* bitneural_config.h determines which path */
#if BN_ACTIVATION_MODE == BN_ACTIVATION_FLOAT
  #define BN_MAC(acc, in, w) \
    if ((w) == 1) { (acc) += (in); } \
    else if ((w) == -1) { (acc) -= (in); }

#elif BN_ACTIVATION_MODE == BN_ACTIVATION_INT8
  #define BN_MAC(acc, in, w) \
    int8_t in_i8 = (int8_t)((in) * 16); \
    if ((w) == 1) { (acc) += in_i8; } \
    else if ((w) == -1) { (acc) -= in_i8; }

#elif BN_ACTIVATION_MODE == BN_ACTIVATION_INT16
  #define BN_MAC(acc, in, w) \
    int16_t in_i16 = (int16_t)((in) * 128); \
    if ((w) == 1) { (acc) += in_i16; } \
    else if ((w) == -1) { (acc) -= in_i16; }

#elif BN_ACTIVATION_MODE == BN_ACTIVATION_INT32
  #define BN_MAC(acc, in, w) \
    int32_t in_i32 = (int32_t)((in) * 65536); \
    if ((w) == 1) { (acc) += in_i32; } \
    else if ((w) == -1) { (acc) -= in_i32; }
#endif
```

**Kernel code (single source, works with any datatype):**

```c
void kernel_dense_ternary(bn_context_t* ctx) {
    for (int out = 0; out < units; out++) {
        bn_act_t acc = bias[out];  /* float or int16_t, doesn't matter */
        
        for (int in = 0; in < ctx->input_len; in += 4) {
            unpack_weight(*weight_ptr++, unpacked);
            for (int j = 0; j < 4; j++) {
                BN_MAC(acc, input[in + j], unpacked[j]);  /* Expands correctly */
            }
        }
        output[out] = acc;
    }
}
```

## Architecture Overview

### C Runtime (`bitneural.c` + `kernels.c`)

**Interpreter-based design**:
- `bn_init()`: Register all built-in kernels
- `bn_run_inference(model_data, input, output)`: Main entry point
- Layer dispatch via kernel registry (OpCode → function pointer)
- Ping-pong buffer management for memory efficiency

**Kernel implementations** (15+ layers):
- **Quantized**: `kernel_dense_ternary()`, `kernel_conv1d_ternary()`, `kernel_conv2d_ternary()`
  - Unpack 2-bit weights on-the-fly using: `if (w==1) acc += x; else if (w==-1) acc -= x;`
- **Activations**: ReLU, Leaky ReLU, Softmax, Sigmoid, Tanh
- **Pooling**: Max Pool 1D, Avg Pool 1D
- **Normalization**: INPUT_NORM, BATCH_NORM
- **Structural**: Flatten, Dropout (no-op at inference)

### Memory & Buffer Management

**Ping-Pong Buffer Strategy**:
- Inference uses two pre-allocated buffers of equal size
- Layer outputs alternate between them (buffer A → B → A → ...)
- Reduces memory footprint: 2× max_tensor_size instead of N× for N layers

**RAM Budget Control**:
```c
// Set max RAM usage before inference
bn_set_ram_limit(256 * 1024);  // 256 KB limit
int status = bn_run_inference_protected(model, input, output, 256 * 1024);

if (status == BN_ERR_RAM_EXCEEDED) {
    // Model too large for available RAM
}
```

**Tensor Metadata**:
- Input/output dimensions automatically propagated between layers
- Each layer receives `input_len` from previous layer's `output_len`
- Kernel must set `ctx.output_len` before returning

**For Custom Layers**:
```c
typedef struct {
    float* data;            /* Tensor buffer */
    int length;             /* Number of elements */
} bn_tensor_desc_t;        /* For future buffer management APIs */
```

### Python Compiler (`compiler.py`)

**Workflow**:
1. Load Keras model
2. Quantize weights to {-1, 0, 1} using threshold-based approach
3. Pack 4 weights into 1 byte (2-bit encoding)
4. Generate binary blob with layer metadata
5. Output C header with `const uint8_t model_data[]`

**Key functions**:
- `quantize_weights_ternary()`: Float32 → {-1, 0, 1}
- `pack_weights_2bit()`: 4 weights → 1 byte
- `BitNeuralCompiler.compile_model()`: Main compiler
- `save_c_header()`: Export to C

---

## Binary Model Format

```
Offset  Size  Field
0       4     Magic: "BITN"
4       4     Number of layers (int32)
8       1     OpCode of layer 0
9       4     Parameter length (int32)
13      N     Parameters for layer 0
...
```

Each layer descriptor:
```
[OpCode (1 byte)] [Param length (4 bytes)] [Params (N bytes)]
```

---

## Weight Quantization

### Ternary Quantization

Converts floating-point weights to {-1, 0, 1}:

```python
threshold = median(|weights|)
quantized[weight >  threshold] = 1
quantized[weight < -threshold] = -1
quantized[otherwise]           = 0
```

**Advantage**: Eliminates floating-point multiplication (replaces with add/subtract)

### 2-Bit Packing

4 ternary weights → 1 byte:

```
Encoding: 0→00, 1→01, -1→10, reserved→11
Byte layout: [w1: 2bits][w2: 2bits][w3: 2bits][w4: 2bits]

Example:
weights = [1, -1, 0, 1]
byte = (0b01 << 6) | (0b10 << 4) | (0b00 << 2) | 0b01 = 0x6801
```

**Size reduction**: 10,000 weights = 2.5 KB (packed) vs 40 KB (float32)

---

## Plugin System: Custom Layers

Extend without modifying library code:

```c
// Define custom kernel
void my_custom_layer(bn_context_t* ctx) {
    float alpha = *((float*)ctx->params);
    
    for (int i = 0; i < ctx->input_len; i++) {
        ctx->output[i] = ctx->input[i] * alpha;
    }
    ctx->output_len = ctx->input_len;
}

// Register at runtime
void app_main() {
    bn_init();
    bn_register_custom_layer(100, my_custom_layer);  // OpCode 100
    
    bn_run_inference(model_data, input, output);
}
```

In Python compiler, use `OP_CUSTOM` or extend `BitNeuralCompiler.LAYER_COMPILER_MAP`.

---

## Memory Optimization

### RAM Usage Example (10→64→32→10 network)

```
Input buffer:        10 floats  = 40 bytes
Intermediate buffer: 64 floats = 256 bytes
Output buffer:       10 floats  = 40 bytes
---
Total: ~350 bytes (vs. ~40 KB for full float32 precision)
```

### Flash Usage Example

```
Weights: 10,000 parameters
  Ternary packed: 2.5 KB
  Float32:       40 KB
  Compression: ~94%
```

### Ping-Pong Buffering

Reuse buffers between layers to minimize RAM:
```c
// Two pre-allocated buffers
float buffer_a[MAX_SIZE];
float buffer_b[MAX_SIZE];

// Layer 1: input → buffer_a
// Layer 2: buffer_a → buffer_b
// Layer 3: buffer_b → buffer_a
// ...
```

---

## Performance Characteristics

### Inference Speed (ESP32 @ 240 MHz)

- **Dense layer** (1000→1000): ~10-50 ms
- **Conv1D** (100 inputs, 32 filters, kernel 5): ~5-20 ms
- **LSTM layer** (32 units, 64 timesteps): ~20-80 ms
- **Full network** (10→64→32→10): ~1-5 ms

(Exact timings depend on clock speed, optimization flags, cache behavior, and board type)

### Optimization Tips

1. **Enable compiler optimizations**: `-O3` in CMakeLists.txt
2. **Use smaller models** when possible (reduces computation)
3. **Pre-allocate buffers** to avoid malloc overhead
4. **Cache-friendly loops** in custom kernels
5. **Profile with ESP32 profiler** to identify bottlenecks
6. **Set board type**: Use `bn_set_board_type()` to enable hardware-specific optimizations (SIMD on ESP32-S3)
7. **RAM limiting**: Use `bn_run_inference_protected()` on memory-constrained devices

---

## Building the Project

### Dependencies

- **GCC/Clang** (ESP32 toolchain or native for testing)
- **CMake** ≥ 3.10
- **Python 3.7+** (for compiler.py)
  - Required: Keras, NumPy
- **FreeRTOS** (required for dual-core inference support)
  - Typically included with ESP-IDF for ESP32

### Build Instructions

```bash
# Clone repository
git clone <repo>
cd BitNeural32

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j4

# (Optional) Install
make install
```

### CMakeLists.txt Example

```cmake
cmake_minimum_required(VERSION 3.10)
project(BitNeural32)

# Core library
add_library(bitneural
    src/bitneural.c
    src/kernels.c
)

target_include_directories(bitneural PUBLIC include)
target_compile_options(bitneural PRIVATE -O3 -Wall -Wextra -std=c99)

# Link with math library
target_link_libraries(bitneural PRIVATE m)
```

---

## API Reference

### Core Functions

#### `void bn_init(void)`
Initialize layer registry and register all built-in kernels. Call once at startup.

#### `void bn_run_inference(const uint8_t* model_data, float* input, float* output)`
Execute inference pipeline.

**Parameters**:
- `model_data`: Binary model blob (starts with "BITN" magic)
- `input`: Input feature vector
- `output`: Output predictions (written by engine)

#### `void bn_register_custom_layer(int opcode, bn_layer_func func)`
Register custom kernel at runtime.

**Parameters**:
- `opcode`: 0-255 (use 100-254 for custom)
- `func`: Function matching `bn_layer_func` signature

#### `void bn_set_board_type(int board_type)` **(NEW)**
Set board type for hardware-specific optimization.

**Parameters**:
- `board_type`: `BOARD_ESP32` (single core), `BOARD_ESP32_S3` (dual core with SIMD), or `BOARD_ESP32_C3` (single core)

#### `void bn_set_ram_limit(int max_bytes)` **(NEW)**
Set maximum RAM available for inference to prevent memory exhaustion.

**Parameters**:
- `max_bytes`: Maximum RAM budget in bytes (default: 256 KB)

#### `void bn_run_inference_protected(const uint8_t* model_data, float* input, float* output, int max_ram)` **(NEW)**
Execute inference with RAM protection enabled.

**Parameters**:
- `model_data`: Binary model blob
- `input`: Input feature vector
- `output`: Output predictions
- `max_ram`: Maximum RAM allowed for this inference (bytes)

### Structures

```c
typedef struct {
    float* input;           /* Input buffer pointer */
    float* output;          /* Output buffer pointer */
    int input_len;          /* Length of input */
    int output_len;         /* Length of output (must be set by kernel) */
    const uint8_t* params;  /* Binary blob for layer parameters */
    
    /* Recurrent state management */
    float* hidden_state;    /* Hidden state buffer (for LSTM/GRU) */
    float* cell_state;      /* Cell state buffer (for LSTM) */
    int state_size;         /* Size of hidden/cell state */
    
    /* Dual-core and RAM protection */
    int use_dual_core;      /* 1 = use dual core, 0 = single core */
    int ram_limit_bytes;    /* RAM budget for this layer (0 = unlimited) */
    int current_ram_usage;  /* Track RAM usage during inference */
} bn_context_t;

typedef void (*bn_layer_func)(bn_context_t* ctx);
```

---

## Troubleshooting & Feature Support

### Choosing the Right Tier for Your Board

**I have ESP32 with LSTM/GRU layer and it runs slowly or crashes**
- Your board: **ESP32-C3, ESP32-C2**
- ✅ **Solution**: Downgrade to **Tier B** (remove LSTM/GRU), or upgrade to **ESP32-S3**
- 📝 **Why**: C3/C2 have single core @ 160/120 MHz; recurrent layers need more compute

**I'm getting BN_ERR_RAM_EXCEEDED on ESP32-C2**
- Your board: **ESP32-C2** (~272 KB SRAM)
- ✅ **Solution**:
  1. Reduce model size (fewer dense units)
  2. Use `bn_set_ram_limit(150 * 1024)` to enforce stricter budget
  3. Stick to **Tier A** (Dense, Conv, ReLU, Pooling only)

**Softmax or Sigmoid gives incorrect probabilities**
- ✅ **Check**: Did you compile with **Tier B** support? ([OpCode table](#opcode-table-supported-layers))
- 📝 **Verify**: Model file header contains OpCode 12 or 13
- ✅ **Fallback**: Use ReLU-based classification if Tier B unavailable

### Performance & FPU Issues

**Inference is very slow on ESP32-C3/C2**

Your board has **no hardware FPU**. All float operations are software-emulated.

- ✅ **Solution 1 (Recommended)**: Compile with integer activations
  ```bash
  idf.py -D BN_ACTIVATION_TYPE=int16_t build
  ```
  This forces **Tier 1** (int16 fixed-point), which runs 2-5× faster on C3/C2.

- ✅ **Solution 2**: Reduce model complexity
  - Fewer dense units (e.g., 16 → 8)
  - Shorter sequences (recurrent models)
  - Remove Softmax/Sigmoid if possible

- ✅ **Solution 3**: Profile and optimize bottleneck
  ```c
  // Add cycle counter before/after kernel
  uint32_t start = xthal_get_ccount();
  kernel_dense_ternary(&ctx);
  uint32_t end = xthal_get_ccount();
  printf("Dense took %lu cycles\n", end - start);
  ```

**Why is Softmax/Sigmoid slow on any board?**

These functions require **exponential computation** (even with approximations, ~50-100 cycles).

- ✅ **For classification**: Use ReLU + argmax instead if acceptable
- ✅ **For gating**: Use Leaky ReLU (single branch, no exp)
- 📝 **Best practice**: Offload expensive activations to the cloud if time-critical

**LSTM/GRU timeout on ESP32-C3**

Recurrent layers have **4× the compute** of dense (4 gates).

- ✅ **Solution**: Use **ESP32 or S3** for LSTM/GRU
- ✅ **Alternative**: Replace with dilated 1D Conv (cheaper approximation)
- ✅ **Fallback**: Reduce hidden size (e.g., 64 → 32)

---

## Troubleshooting

### "Unsupported layer type" during compilation

**Solution**: Extend `BitNeuralCompiler.LAYER_COMPILER_MAP` in `compiler.py`:

```python
class MyLayerCompiler(LayerCompiler):
    def compile(self, layer):
        # Your implementation
        return OP_CUSTOM, blob

BitNeuralCompiler.LAYER_COMPILER_MAP['MyLayer'] = MyLayerCompiler
```

### Output is all zeros or NaN

**Possible causes**:
- **Magic number mismatch**: Verify model data starts with "BITN"
- **Buffer overflow**: Check input/output buffer sizes
- **Uninitialized weights**: Ensure model_data is properly linked in memory

### Understanding Error Codes

**BN_ERR_INVALID_OPCODE (-3)**
- **Cause**: Model contains unsupported layer (OpCode not registered)
- **Solution**: 
  - Verify model matches compiled BitNeural32 version
  - Check if layer is in your [feature tier](#feature-tier-support-matrix)
  - Re-compile model with `bitneural32 compiler --tier-check`

**BN_ERR_INVALID_MODEL (-1)**
- **Cause**: Model data corrupted or not compiled with BitNeural32
- **Solution**: Re-generate model from Python compiler

**BN_ERR_NULL_POINTER (-2)**
- **Cause**: Input, output, or model_data pointer is NULL
- **Solution**: Check all pointers are properly initialized before `bn_run_inference()`

**BN_ERR_RAM_EXCEEDED (-4)**
- **Cause**: Inference would exceed RAM budget
- **Solution**: 
  - Reduce `bn_set_ram_limit()` value
  - Simplify model (fewer units/filters)
  - Use smaller data types if possible

### Inference is slow

**Solutions**:
1. Enable `-O3` optimization in CMakeLists.txt
2. Profile with ESP32 profiler to find bottleneck
3. Reduce model size (fewer layers/units)
4. Consider custom kernel optimization for critical layers


---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## References

- **BitNet Paper**: [arxiv.org/abs/2310.11453](https://arxiv.org/abs/2310.11453)
- **ESP32 Documentation**: [espressif.com](https://www.espressif.com/)
- **TensorFlow/Keras**: [tensorflow.org](https://www.tensorflow.org/)
- **Ternary Networks**: [arxiv.org/abs/1609.00222](https://arxiv.org/abs/1609.00222)

---

**Made with ❤️ by Aizhee for embedded machine learning**

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/O4O0XNVKI)
