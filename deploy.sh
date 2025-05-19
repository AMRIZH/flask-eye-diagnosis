#!/bin/bash

# Eye Diagnosis App Deployment Script

# Display help message
show_help() {
    echo "Eye Diagnosis Flask App Deployment Script"
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  --build       Build the Docker image"
    echo "  --start       Start the Docker container"
    echo "  --stop        Stop the Docker container"
    echo "  --restart     Restart the Docker container"
    echo "  --logs        View the container logs"
    echo "  --deploy      Full deployment (build and start)"
    echo "  --help        Display this help message"
}

# Check if Docker and Docker Compose are installed
check_requirements() {
    if ! command -v docker &> /dev/null; then
        echo "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Build the Docker image
build_image() {
    echo "Building Docker image..."
    docker-compose build
}

# Start the container
start_container() {
    echo "Starting container..."
    docker-compose up -d
    echo "Container started. Access the application at http://localhost:5000"
}

# Stop the container
stop_container() {
    echo "Stopping container..."
    docker-compose down
}

# Restart the container
restart_container() {
    echo "Restarting container..."
    docker-compose restart
}

# View container logs
view_logs() {
    echo "Viewing container logs (press Ctrl+C to exit)..."
    docker-compose logs -f
}

# Full deployment
deploy() {
    build_image
    start_container
}

# Main script logic
check_requirements

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    --build)
        build_image
        ;;
    --start)
        start_container
        ;;
    --stop)
        stop_container
        ;;
    --restart)
        restart_container
        ;;
    --logs)
        view_logs
        ;;
    --deploy)
        deploy
        ;;
    --help)
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac

exit 0 