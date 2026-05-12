#!/bin/bash

# ThinFormer Installation Script for Linux/macOS
# This script copies ThinFormer files to MMDetection directory

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the directory where this script is located (ThinFormer root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
THINFORMER_ROOT="$SCRIPT_DIR"

# Function to print help
print_help() {
    cat <<EOF
ThinFormer Installation Script

Usage:
    ./install.sh <mmdetection_root>

Parameters:
    mmdetection_root    Path to MMDetection root directory (required)

Examples:
    ./install.sh /home/user/mmdetection
    ./install.sh ~/mmdetection

EOF
}

# Function to print error and exit
error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print info
print_info() {
    echo -e "${CYAN}$1${NC}"
}

# Function to test if path exists
test_path() {
    local path=$1
    local description=$2
    if [ ! -d "$path" ]; then
        error_exit "$description not found at: $path"
    fi
}

# Function to copy file
copy_file() {
    local source=$1
    local dest=$2
    local description=$3
    
    if [ ! -f "$source" ]; then
        echo -e "${YELLOW}WARNING: $description source not found: $source${NC}"
        return 1
    fi
    
    local dest_dir=$(dirname "$dest")
    if [ ! -d "$dest_dir" ]; then
        mkdir -p "$dest_dir"
        print_info "Created directory: $dest_dir"
    fi
    
    cp "$source" "$dest"
    print_success "Copied $description"
    return 0
}

# Check for arguments
if [ $# -eq 0 ]; then
    print_help
    exit 1
fi

# Get MMDetection root
MMDETECTION_ROOT="$1"

# Resolve absolute paths
if [ ! -d "$MMDETECTION_ROOT" ]; then
    error_exit "MMDetection root directory not found: $MMDETECTION_ROOT"
fi

# Convert to absolute path
MMDETECTION_ROOT="$(cd "$MMDETECTION_ROOT" && pwd)"

echo ""
print_info "========================================"
print_info "ThinFormer Installation"
print_info "========================================"
echo ""
echo "ThinFormer root:  $THINFORMER_ROOT"
echo "MMDetection root: $MMDETECTION_ROOT"
echo ""

# Test required directories
test_path "$MMDETECTION_ROOT/mmdet/models/backbones" "MMDetection backbones directory"
test_path "$MMDETECTION_ROOT/mmdet/models/detectors" "MMDetection detectors directory"
test_path "$THINFORMER_ROOT/models/backbones" "ThinFormer backbones directory"
test_path "$THINFORMER_ROOT/models/detectors" "ThinFormer detectors directory"

echo ""
print_info "Copying files..."
echo ""

COPY_COUNT=0

# Copy backbone
if copy_file \
    "$THINFORMER_ROOT/models/backbones/Thinformer.py" \
    "$MMDETECTION_ROOT/mmdet/models/backbones/Thinformer.py" \
    "Thinformer backbone"; then
    ((COPY_COUNT++))
fi

# Copy detectors
if copy_file \
    "$THINFORMER_ROOT/models/detectors/base_detr_dynamic.py" \
    "$MMDETECTION_ROOT/mmdet/models/detectors/base_detr_dynamic.py" \
    "base_detr_dynamic detector"; then
    ((COPY_COUNT++))
fi

if copy_file \
    "$THINFORMER_ROOT/models/detectors/deformable_detr_dynamic.py" \
    "$MMDETECTION_ROOT/mmdet/models/detectors/deformable_detr_dynamic.py" \
    "deformable_detr_dynamic detector"; then
    ((COPY_COUNT++))
fi

if copy_file \
    "$THINFORMER_ROOT/models/detectors/dino_dynamic.py" \
    "$MMDETECTION_ROOT/mmdet/models/detectors/dino_dynamic.py" \
    "dino_dynamic detector"; then
    ((COPY_COUNT++))
fi

# Copy config files
# Copy main Thinformer config
if [ -f "$THINFORMER_ROOT/configs/Thinformer.py" ]; then
    mkdir -p "$MMDETECTION_ROOT/configs/thinformer"
    cp "$THINFORMER_ROOT/configs/Thinformer.py" "$MMDETECTION_ROOT/configs/thinformer/Thinformer.py"
    print_success "Copied config: Thinformer.py"
    ((COPY_COUNT++))
fi

# Copy dataset config
if [ -f "$THINFORMER_ROOT/configs/_base_/datasets/panda_detection.py" ]; then
    mkdir -p "$MMDETECTION_ROOT/configs/_base_/datasets"
    cp "$THINFORMER_ROOT/configs/_base_/datasets/panda_detection.py" "$MMDETECTION_ROOT/configs/_base_/datasets/panda_detection.py"
    print_success "Copied config: panda_detection.py"
    ((COPY_COUNT++))
fi

echo ""
print_info "========================================"
print_success "Installation Complete!"
print_info "========================================"
echo ""
echo "Total files copied: $COPY_COUNT"