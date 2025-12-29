#ifndef BITNEURAL_CONFIG_H
#define BITNEURAL_CONFIG_H

/* ============================================
 * BitNeural32 Configuration Header
 * ============================================
 *
 * This file lets users choose the activation
 * datatype (compile-time, affects accumulators).
 *
 * ACTIVATION FUNCTIONS (OpCode dispatch) are
 * always runtime-selectable, independent of
 * this setting.
 *
 * ============================================ */

/* Activation Datatype Options */
#define BN_ACTIVATION_FLOAT   0  /* 32-bit float, FPU-optimized */
#define BN_ACTIVATION_INT8    1  /* 8-bit integer, ultra-low memory (Q3.4 fixed-point) */
#define BN_ACTIVATION_INT16   2  /* 16-bit integer, no FPU needed (Q7.8 fixed-point) */
#define BN_ACTIVATION_INT32   3  /* 32-bit integer, for higher precision (Q15.16 fixed-point) */

/* ============================================
 * USER SELECTION: Choose one of three ways
 * ============================================
 *
 * Option A: Edit this file (most Arduino users)
 * Simply uncomment/change the line below:
 *
 *   #define BN_ACTIVATION_MODE BN_ACTIVATION_INT16
 *
 * Option B: Define in your sketch before include (advanced users)
 * In your .ino file, add before #include <bitneural.h>:
 *
 *   #define BN_ACTIVATION_MODE BN_ACTIVATION_INT16
 *   #include <bitneural.h>
 *
 * Option C: Compiler flag (PlatformIO / ESP-IDF)
 * In platformio.ini or build command:
 *
 *   -D BN_ACTIVATION_MODE=BN_ACTIVATION_INT16
 *
 * ============================================ */

#ifndef BN_ACTIVATION_MODE
  /* Default: float (works on all ESP32 boards with FPU) */
  #define BN_ACTIVATION_MODE BN_ACTIVATION_FLOAT
#endif

#endif // BITNEURAL_CONFIG_H
