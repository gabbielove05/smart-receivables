#!/bin/bash

# InvoiceFlow Application Launcher
# Automated setup and launch script for the Smart Receivables Navigator

set -e  # Exit on any error

echo "=================================================="
echo "🏦 InvoiceFlow Smart Receivables Navigator"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the InvoiceFlow directory
if [[ ! -f "app.py" ]]; then
    print_error "app.py not found. Please run this script from the InvoiceFlow directory."
    exit 1
fi

print_status "Starting InvoiceFlow setup..."

# Step 1: Check Python version
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 is required but not found. Please install Python 3.11 or higher."
    exit 1
fi

# Step 2: Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Step 3: Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Step 4: Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Step 5: Install dependencies
print_status "Installing dependencies..."
if [[ -f "pyproject.toml" ]]; then
    pip install -e .
    print_success "Dependencies installed from pyproject.toml"
else
    print_error "pyproject.toml not found"
    exit 1
fi

# Step 6: Check for environment file
if [[ ! -f ".env" ]]; then
    if [[ -f "env_example.txt" ]]; then
        print_warning ".env file not found. Creating from template..."
        cp env_example.txt .env
        print_success "Created .env file from template"
        print_warning "Please edit .env file with your API keys if needed"
    else
        print_warning "No .env file found, but OpenRouter API key is pre-configured"
    fi
else
    print_status ".env file found"
fi

# Step 7: Generate sample data if needed
if [[ ! -f "sample_invoices.csv" ]] || [[ ! -f "sample_payments.csv" ]]; then
    print_status "Generating sample data..."
    python3 generate_sample_data.py
    print_success "Sample data generated"
else
    print_status "Sample data files found"
fi

# Step 8: Run basic import test
print_status "Testing application imports..."
python3 -c "
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from simple_email_system import simple_email_system
print('✅ All imports successful')
" 2>/dev/null || {
    print_error "Import test failed. Please check your installation."
    exit 1
}
print_success "Import test passed"

# Step 9: Check if port is available
PORT=8501
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    print_warning "Port $PORT is already in use. Streamlit may use a different port."
fi

# Step 10: Launch the application
print_success "Setup complete! Launching InvoiceFlow..."
echo ""
echo "=================================================="
echo "🚀 Starting InvoiceFlow Application"
echo "=================================================="
echo ""
print_status "The application will open in your default web browser"
print_status "If it doesn't open automatically, visit: http://localhost:$PORT"
echo ""
print_warning "To stop the application, press Ctrl+C"
echo ""

# Launch Streamlit
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
streamlit run app.py --server.port $PORT --server.headless false --browser.gatherUsageStats false

print_status "Application stopped"
