# Eye Diagnosis Flask App Deployment Guide

This guide provides instructions on how to deploy the Eye Diagnosis Flask application using Docker on a VPS.

## Prerequisites

- A VPS running Linux (Ubuntu 20.04 LTS or newer recommended)
- Docker and Docker Compose installed on the VPS
- SSH access to the VPS

## Deployment Steps

### 1. Set Up Your VPS

Update your VPS and install Docker and Docker Compose:

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install -y docker-compose

# Add your user to the Docker group (to run Docker without sudo)
sudo usermod -aG docker $USER

# Log out and log back in for the group changes to take effect
```

### 2. Transfer Files to VPS

Copy your application files to the VPS. You can use `scp`, `rsync`, or Git:

Using SCP:
```bash
scp -r /path/to/your/app/* username@your-vps-ip:/path/to/deployment/
```

OR using Git:
```bash
# On your VPS
git clone <your-repository-url>
cd <repository-directory>
```

### 3. Configure the Application

Make sure your model file is in the correct location:
```bash
# Create the models directory if it doesn't exist
mkdir -p static/models

# Copy your model file to the models directory
# You may need to upload it to your VPS first
```

### 4. Build and Start the Docker Container

```bash
# Navigate to your application directory
cd /path/to/deployment/

# Build and start the Docker container
docker-compose up -d
```

### 5. Verify Deployment

Check if the container is running:
```bash
docker ps
```

You should see your container listed as "eye-diagnosis-app".

Access your application at:
```
http://your-vps-ip:5000
```

### 6. Set Up a Reverse Proxy (Optional but Recommended)

For production use, you should set up Nginx as a reverse proxy:

```bash
# Install Nginx
sudo apt-get install -y nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/eye-diagnosis
```

Add the following configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/eye-diagnosis /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 7. Set Up HTTPS with Let's Encrypt (Optional but Recommended)

```bash
# Install Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

### 8. Maintenance

#### Viewing Logs
```bash
docker logs eye-diagnosis-app
```

#### Restarting the Application
```bash
docker-compose restart
```

#### Updating the Application
```bash
# Stop the container
docker-compose down

# Pull the latest changes (if using Git)
git pull

# Rebuild and start the container
docker-compose up -d --build
```

## Troubleshooting

### Application Not Accessible
Check if the container is running:
```bash
docker ps
```

Check the container logs:
```bash
docker logs eye-diagnosis-app
```

### Database Issues
The SQLite database is stored inside the container. To persist it across container restarts, you may want to add a volume in the docker-compose.yml:
```yaml
volumes:
  - ./data:/app/data
``` 