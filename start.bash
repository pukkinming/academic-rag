#!/bin/bash

# Start Docker engine
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Check Docker status
# sudo systemctl status docker

# Add your user to docker group (to avoid sudo)
sudo usermod -aG docker $USER

# Build images (first time only)
docker-compose build

# Start all services in detached mode
docker-compose up