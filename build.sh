#!/bin/bash

# Build script for Neural Network project
# Usage: 
#   ./build.sh           - Build release version
#   ./build.sh debug     - Build debug version
#   ./build.sh clean     - Clean build artifacts
#   ./build.sh test      - Build and run tests
#   ./build.sh help      - Show help

set -e  # Exit on any error

# Configuration
PROJECT_NAME="nn"
SRC_DIR="src"
SOURCE_FILES="$SRC_DIR/program.c $SRC_DIR/nn.c $SRC_DIR/io.c $SRC_DIR/test.c"
OUTPUT_DIR="bin"
EXECUTABLE="$OUTPUT_DIR/$PROJECT_NAME"

# Compiler settings
CC="clang"
CFLAGS_COMMON="-std=c23 -Wall -Wextra -I$SRC_DIR"
CFLAGS_RELEASE="-Ofast -DNDEBUG -mllvm -force-vector-width=8"
CFLAGS_DEBUG="-g -O0 -DDEBUG"
LDFLAGS="-lm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

show_help() {
    echo "Neural Network Build Script"
    echo ""
    echo "Usage:"
    echo "  ./build.sh           Build release version"
    echo "  ./build.sh debug     Build debug version"
    echo "  ./build.sh clean     Clean build artifacts"
    echo "  ./build.sh test      Build and run tests"
    echo "  ./build.sh help      Show this help"
    echo ""
    echo "Examples:"
    echo "  ./build.sh                    # Build optimized release"
    echo "  ./build.sh debug              # Build with debug info"
    echo "  ./build.sh test               # Build and run tests"
    echo "  ./build.sh clean              # Clean build files"
}

check_prerequisites() {
    # Check if clang is available
    if ! command -v $CC &> /dev/null; then
        echo -e "${RED}✗ Error: Clang compiler not found. Please install clang.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Clang compiler found${NC}"

    # Check if source directory exists
    if [ ! -d "$SRC_DIR" ]; then
        echo -e "${RED}✗ Error: Source directory '$SRC_DIR' not found.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Source directory found${NC}"

    # Check if source files exist
    for file in $SOURCE_FILES; do
        if [ ! -f "$file" ]; then
            echo -e "${RED}✗ Error: Source file '$file' not found.${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}✓ All source files found${NC}"

    # Create output directory if it doesn't exist
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
        echo -e "${GREEN}✓ Created output directory: $OUTPUT_DIR${NC}"
    fi
}

clean_build() {
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    
    if [ -d "$OUTPUT_DIR" ]; then
        rm -f "$OUTPUT_DIR"/*.o "$OUTPUT_DIR"/$PROJECT_NAME "$OUTPUT_DIR"/${PROJECT_NAME}_debug
        echo -e "${GREEN}✓ Cleaned $OUTPUT_DIR${NC}"
    fi
    
    # Remove any temporary files
    rm -f *.o *.out
    
    echo -e "${GREEN}✓ Clean completed${NC}"
}

build_project() {
    local build_type=$1
    
    echo -e "${YELLOW}Building $build_type version...${NC}"
    
    # Prepare compiler flags
    local flags="$CFLAGS_COMMON"
    local output_file
    
    if [ "$build_type" = "Debug" ]; then
        flags="$flags $CFLAGS_DEBUG"
        output_file="$OUTPUT_DIR/${PROJECT_NAME}_debug"
    else
        flags="$flags $CFLAGS_RELEASE"
        output_file="$EXECUTABLE"
    fi
    
    # Build command
    local build_cmd="$CC $flags $SOURCE_FILES -o $output_file $LDFLAGS"
    
    echo -e "${CYAN}Executing: $build_cmd${NC}"
    
    # Execute build
    if $build_cmd; then
        echo -e "${GREEN}✓ Build successful: $output_file${NC}"
        
        # Show file info
        if [ -f "$output_file" ]; then
            local size=$(du -h "$output_file" | cut -f1)
            local date=$(date -r "$output_file" "+%Y-%m-%d %H:%M:%S")
            echo -e "${GRAY}  Size: $size${NC}"
            echo -e "${GRAY}  Created: $date${NC}"
        fi
        return 0
    else
        echo -e "${RED}✗ Build failed${NC}"
        return 1
    fi
}

run_tests() {
    echo -e "${YELLOW}Building and running tests...${NC}"
    
    # Build debug version for testing
    if build_project "Debug"; then
        local test_executable="$OUTPUT_DIR/${PROJECT_NAME}_debug"
        
        if [ -f "$test_executable" ]; then
            echo -e "${CYAN}Running tests...${NC}"
            if "$test_executable" test; then
                echo -e "${GREEN}✓ All tests passed${NC}"
            else
                echo -e "${RED}✗ Some tests failed${NC}"
                return 1
            fi
        else
            echo -e "${RED}✗ Test executable not found${NC}"
            return 1
        fi
    else
        return 1
    fi
}

# Main execution
case "${1:-}" in
    help|--help|-h)
        show_help
        exit 0
        ;;
    clean)
        check_prerequisites
        clean_build
        exit 0
        ;;
    test)
        check_prerequisites
        run_tests
        exit $?
        ;;
    debug)
        check_prerequisites
        if build_project "Debug"; then
            echo ""
            echo -e "${GREEN}Debug build completed successfully!${NC}"
            echo -e "${CYAN}Run with: ./$OUTPUT_DIR/${PROJECT_NAME}_debug${NC}"
        else
            echo ""
            echo -e "${RED}Build failed!${NC}"
            exit 1
        fi
        ;;
    "")
        # Default: Release build
        check_prerequisites
        if build_project "Release"; then
            echo ""
            echo -e "${GREEN}Build completed successfully!${NC}"
            echo -e "${CYAN}Run with: ./$EXECUTABLE${NC}"
        else
            echo ""
            echo -e "${RED}Build failed!${NC}"
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo "Use './build.sh help' for usage information."
        exit 1
        ;;
esac
