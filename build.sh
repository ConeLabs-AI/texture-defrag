#!/bin/bash

set -e

BUILD_DIR="build"
PROJECT_FILE="texture-defrag/texture-defrag.pro"

# Clean and create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Run qmake to generate Makefile
echo "Running qmake..."
qmake ../"$PROJECT_FILE" -spec linux-g++

# Build the project
echo "Building..."
make -j$(nproc)

echo "Build complete. Executable: $BUILD_DIR/texture-defrag"
