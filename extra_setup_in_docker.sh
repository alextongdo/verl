#!/usr/bin/env bash
# Additional setup script to be run inside the Docker container
# This script installs required packages and sets up the project environment

set -e

# Get the absolute path of the project directory
echo "🔧 Setting up project environment..."
PROJECT_DIR=$(realpath "$(pwd)")
pip install --no-deps -e "${PROJECT_DIR}"
pip install langgraph
echo "✅ Setup complete!"

exec bash