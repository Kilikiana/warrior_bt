#!/bin/bash

# Ross Cameron Trading Bot Environment Setup
echo "Setting up virtual environment for Warrior BT..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install TA-Lib dependencies (macOS specific)
echo "Installing TA-Lib dependencies..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Warning: Homebrew not found. Please install TA-Lib manually."
        echo "Visit: https://github.com/mrjbq7/ta-lib#dependencies"
    else
        echo "Installing TA-Lib via Homebrew..."
        brew install ta-lib
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing TA-Lib dependencies for Linux..."
    echo "You may need to run: sudo apt-get install build-essential libssl-dev libffi-dev python3-dev"
else
    echo "Windows detected. Please install TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib"
fi

# Install Python requirements
echo "Installing Python requirements..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "deactivate"