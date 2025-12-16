#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Your Docker Hub username should be passed as the first argument.
DOCKERHUB_USERNAME=$1
# The name for your image on Docker Hub.
IMAGE_NAME="content-generator"
# The tag for your image (e.g., latest, v1.0).
IMAGE_TAG="latest"

# --- Validation ---
# Check if the Docker Hub username was provided.
if [ -z "$DOCKERHUB_USERNAME" ]; then
    echo "Error: Docker Hub username is required."
    echo "Usage: ./build_and_push.sh <your-dockerhub-username>"
    exit 1
fi

FULL_IMAGE_NAME="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building and pushing multi-arch image to: ${FULL_IMAGE_NAME}"

# --- Setup Docker Buildx ---
# Create and switch to a new builder instance that supports multi-arch builds.
# This is a persistent builder, so it only needs to be created once.
BUILDER_NAME="multiarch_builder"
if ! docker buildx ls | grep -q "$BUILDER_NAME"; then
    echo "Creating new buildx builder: $BUILDER_NAME"
    docker buildx create --name "$BUILDER_NAME" --use
else
    echo "Using existing buildx builder: $BUILDER_NAME"
    docker buildx use "$BUILDER_NAME"
fi

# Ensure the builder is running
docker buildx inspect --bootstrap

# --- Build and Push ---
# Build the Docker image for both amd64 and arm64 platforms and push them.
# The '--push' flag tells buildx to push the resulting images to the registry.
echo "Starting multi-platform build and push..."
docker buildx build --platform linux/amd64,linux/arm64 -t "${FULL_IMAGE_NAME}" --push .

echo "✅ Successfully built and pushed multi-arch image to ${FULL_IMAGE_NAME}"
docker buildx use default