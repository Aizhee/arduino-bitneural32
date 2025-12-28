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

---

## OpCode Table: Supported Layers

| OpCode | Layer Name | Parameters | Notes |
|--------|-----------|------------|-------|
| 0 | INPUT_NORM | mean, std | Input preprocessing |
| 1 | CONV1D_TERNARY | filters, kernel_size, stride, weights | 1.58-bit quantized 1D conv |
| 2 | DENSE_TERNARY | units, weights, bias | 1.58-bit quantized dense |
| 3 | CONV2D_TERNARY | filters, kernel_h, kernel_w, stride, weights | 1.58-bit quantized 2D conv |
| 10 | RELU | — | Rectified Linear Unit |
| 11 | LEAKY_RELU | alpha | Leaky ReLU with slope |
| 12 | SOFTMAX | — | Softmax (classification) |
| 13 | SIGMOID | — | Sigmoid activation |
| 14 | TANH | — | Hyperbolic tangent |
| 20 | MAXPOOL_1D | pool_size, stride | 1D max pooling |
| 21 | AVGPOOL_1D | pool_size, stride | 1D average pooling |
| 22 | FLATTEN | — | No-op (memory already flat) |
| 23 | DROPOUT | rate | No-op at inference |
| 30 | BATCH_NORM | channels, gamma, beta, mean, var | Batch normalization |
| 255 | CUSTOM | custom_id | User-defined custom layer |

---

## Usage Guide

### Step 1: Install BitNeural32
```bash
pip install bitneural32
```

### Step 2: Train Your Keras Model

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_train, Y_train, epochs=10)
model.save('classifier.h5')
```

### Step 3: Compile to C Header

```python
from bitneural32.compiler import BitNeuralCompiler

compiler = BitNeuralCompiler()
compiler.compile_model(model, input_data=X_train)
compiler.save_c_header('model_data.h')
```

### Optional: Quantization-Aware Training (QAT)

Use BitNeural32’s custom ternary layers to train models that match the runtime quantization behavior. This improves accuracy after export.

```python
import tensorflow as tf
import numpy as np
from bitneural32.qat import TernaryDense, TernaryConv1D
from bitneural32.compiler import BitNeuralCompiler

# 1. Build a QAT model (example: Dense)
qat_model = tf.keras.Sequential([
    TernaryDense(32, activation='relu', input_shape=(8,)),
    TernaryDense(16, activation='relu'),
    TernaryDense(4, activation='softmax')
])

qat_model.compile(optimizer='adam', loss='mse')
X_train = np.random.randn(200, 8)
Y_train = np.random.randn(200, 4)
qat_model.fit(X_train, Y_train, epochs=5, verbose=0)

# 2. Export with the same compiler (QAT layers are recognized)
compiler = BitNeuralCompiler()
compiler.compile_model(qat_model, input_data=X_train)
compiler.save_c_header('model_data_qat.h')
```

Notes:
- `TernaryConv1D` assumes mono-channel inputs and flattens weights as (filters, kernel_size) for optimized ESP32 kernels. For multi-channel convolution, prefer standard `Conv2D` with appropriate reshaping.
- QAT layers’ names (`TernaryDense`, `TernaryConv1D`) are mapped in the compiler and compiled using the optimized ternary kernels.

### Step 3: Run on ESP32

```c
#include "BitNeural32.h"
#include "model_data.h"

void app_main() {
    bn_init();
    
    float input[10] = {...};
    float output[10];
    
    bn_run_inference(model_data, input, output);
    
    printf("Prediction: %f\n", output[0]);
}
```

---

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
- **Full network** (10→64→32→10): ~1-5 ms

(Exact timings depend on clock speed, optimization flags, cache behavior)

### Optimization Tips

1. **Enable compiler optimizations**: `-O3` in CMakeLists.txt
2. **Use smaller models** when possible (reduces computation)
3. **Pre-allocate buffers** to avoid malloc overhead
4. **Cache-friendly loops** in custom kernels
5. **Profile with ESP32 profiler** to identify bottlenecks

---

## Building the Project

### Dependencies

- **GCC/Clang** (ESP32 toolchain or native for testing)
- **CMake** ≥ 3.10
- **Python 3.7+** (for compiler.py)
  - Optional: TensorFlow, Keras, NumPy
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

## Complete Example Workflow

### Python: Train and Export

```python
import tensorflow as tf
import numpy as np
from bitneural32.compiler import BitNeuralCompiler

# 1. Create and train model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='mse')
X_train = np.random.randn(100, 8)
Y_train = np.random.randn(100, 4)
model.fit(X_train, Y_train, epochs=5, verbose=0)

# 2. Compile to BitNeural format
compiler = BitNeuralCompiler()
compiler.compile_model(model, input_data=X_train)
compiler.save_c_header('model_data.h')

# 3. Show report
report = compiler.get_compilation_report()
print(f"✓ Model compiled: {report['total_size_bytes']} bytes")
print(f"  Layers: {report['num_layers']}")
```

### C: Run on ESP32

```c
#include "BitNeural32.h"
#include "model_data.h"
#include <stdio.h>

void app_main() {
    // Initialize
    bn_init();
    
    // Prepare input
    float input[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    float output[4];
    
    // Run inference
    printf("Running inference...\n");
    bn_run_inference(model_data, input, output);
    
    // Display results
    printf("Output probabilities:\n");
    for (int i = 0; i < 4; i++) {
        printf("  Class %d: %.6f\n", i, output[i]);
    }
}
```

---

## API Reference

### Core Functions

#### `void bn_init(void)`
Initialize layer registry. Call once at startup.

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

### Structures

```c
typedef struct {
    float* input;           // Input buffer
    float* output;          // Output buffer
    int input_len;          // Input size
    int output_len;         // Output size (set by kernel)
    const uint8_t* params;  // Layer parameters
} bn_context_t;

typedef void (*bn_layer_func)(bn_context_t* ctx);
```

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
