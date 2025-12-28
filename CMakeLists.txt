cmake_minimum_required(VERSION 3.10)

project(BitNeural32 C)

set(CMAKE_C_STANDARD 99)
set(BITNEURAL32_VERSION "0.0.1")

# Arduino IDE compatibility - set output directories
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

# Add src directory to include path (for VS Code IntelliSense and Arduino IDE)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Source files
file(GLOB SOURCES "src/*.c")

# Create library
add_library(BitNeural32 ${SOURCES})

# Include directories for the library target
target_include_directories(BitNeural32 PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

# Optional: example executable (commented for Arduino IDE compatibility)
# add_executable(BitNeural32_example examples/examples.h)
# target_link_libraries(BitNeural32_example BitNeural32)