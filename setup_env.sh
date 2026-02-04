#!/bin/bash

# Define the environment directory
VENV_DIR=".venv"

echo "=========================================="
echo "    Project Setup Script (Mac/Linux)      "
echo "=========================================="

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: 'python3' is not installed or not in PATH."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists in $VENV_DIR."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found!"
fi

echo "=========================================="
echo "          Setup Complete!                 "
echo "=========================================="
echo ""
echo "To activate the environment manually, run:"
echo "    source $VENV_DIR/bin/activate"
echo ""
