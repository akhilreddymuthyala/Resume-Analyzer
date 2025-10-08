# ATS Resume Classifier - Deployment Guide

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [Deployment Options](#deployment-options)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Security Best Practices](#security-best-practices)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This guide covers deploying the ATS Resume Classifier application to production environments. The application supports multiple deployment methods including traditional servers, Docker containers, and cloud platforms.

### Key Features
- ✅ Rate limiting and security headers
- ✅ Comprehensive error handling
- ✅ Structured logging with rotation
- ✅ Health check endpoints
- ✅ File upload validation
- ✅ Automatic cleanup of temporary files
- ✅ Production-ready with Gunicorn
- ✅ Docker support

---

## 📦 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, Debian 11+)
- **Python**: 3.9 or higher
- **RAM**: Minimum 2GB, Recommended 4GB
- **Storage**: Minimum 5GB free space
- **CPU**: 2+ cores recommended

### Software Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git curl

# CentOS/RHEL
sudo yum install -y python3 python3-pip git curl

# macOS
brew install python3 git
```

### Model Files Required
Ensure you have these files in your application directory:
- `ats_model_improved.pkl`
- `tfidf_vectorizer_improved.pkl`
- `label_encoder_improved.pkl`
- `text_preprocessor_improved.pkl`
- `ats_claude.csv`

---

## 🚀 Installation Methods

### Method 1: Manual Installation (Traditional Server)

#### Step 1: Clone and Setup
```bash
# Clone repository
git clone <your-repo-url>
cd ats-classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

#### Step 2: Configure Environment
```bash
# Copy and edit environment file
cp .env.example .env
nano .env  # Edit with your settings
```

#### Step 3: Create Directories
```bash
mkdir -p uploads logs
touch uploads/.gitkeep
```

#### Step 4: Test the Application
```bash
# Development mode
export FLASK_ENV=development
python app.py

# Test health endpoint
curl http://localhost:5000/health
```

#### Step 5: Production Deployment with Gunicorn
```bash
# Run with gunicorn
gunicorn --config gunicorn_config.py app:app

# Or use the Makefile
make prod
```

#### Step 6: Setup as System Service (Optional but Recommended)
```bash
# Edit service file with your paths
sudo nano /etc/systemd/system/ats-classifier.service

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ats-classifier
sudo systemctl start ats-classifier

# Check status
sudo systemctl status ats-classifier
```

---

### Method 2: Docker Deployment

#### Step 1: Build Docker Image
```bash
# Build the image
docker build -t ats-classifier:latest .

# Or use docker-compose
docker-compose build
```

#### Step 2: Run Container
```bash
# Using docker directly
docker run -d \
  --name ats-classifier \
  -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/logs:/app/logs \
  -e SECRET_KEY="your-secret-key" \
  ats-classifier:latest

# Using docker-compose (recommended)
docker-compose up -d
```

#### Step 3: Verify Deployment
```bash
# Check container status
docker ps

# View logs
docker logs ats-classifier -f

# Health check
curl http://localhost:5000/health
```

#### Step 4: Access Application
```bash
# Direct access
http://localhost:5000

# Through nginx (if using docker-compose with nginx)
http://localhost
```

---

### Method 3: Cloud Platform Deployment

#### AWS Elastic Beanstalk

1. **Install EB CLI**
```bash
pip install awsebcli
```

2. **Initialize EB**
```bash
eb init -p python-3.11 ats-classifier
```

3. **Create Environment**
```bash
eb create ats-production
```

4. **Deploy**
```bash
eb deploy
```

#### Google Cloud Run

1. **Build and Push Image**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/ats-classifier
```

2. **Deploy**
```bash
gcloud run deploy ats-classifier \
  --image gcr.io/PROJECT_ID/ats-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Web Apps

1. **Login**
```bash
az login
```

2. **Create Resource Group**
```bash
az group create --name ats-rg --location eastus
```

3. **Deploy**
```bash
az webapp up --name ats-classifier --resource-group ats-rg
```

#### Heroku

1. **Login**
```bash
heroku login
```

2. **Create App**
```bash
heroku create ats-classifier
```

3. **Deploy**
```bash
git push heroku main
```

4. **Add Procfile**
```
web: gunicorn --config gunicorn_config.py app:app
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Flask Configuration
FLASK_ENV=production
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
SECRET_KEY=your-very-secure-secret-key-here

# Model Paths (relative to app directory)
MODEL_PATH=ats_model_improved.pkl
VECTORIZER_PATH=tfidf_vectorizer_improved.pkl
ENCODER_PATH=label_encoder_improved.pkl
PREPROCESSOR_PATH=text_preprocessor_improved.pkl
DATA_PATH=ats_claude.csv

# Upload Configuration
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216  # 16MB in bytes

# Rate Limiting
RATELIMIT_STORAGE_URL=memory://
RATELIMIT_DEFAULT=100 per hour

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/ats_app.log
```

### Gunicorn Configuration

Edit `gunicorn_config.py` for production tuning:

```python
# Worker processes (CPU cores * 2 + 1)
workers = 4

# Worker timeout (increase for slow operations)
timeout = 60

# Logging
loglevel = 'info'
```

### Nginx Configuration (Reverse Proxy)

For production, use Nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 16M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 🔒 Security Best Practices

### 1. Secret Key Management
```bash
# Generate secure secret key
python -c 'import secrets; print(secrets.token_hex(32))'

# Never commit .env files to version control
echo ".env" >> .gitignore
```

### 2. HTTPS Configuration
```bash
# Install certbot for Let's Encrypt SSL
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com
```

### 3. Firewall Configuration
```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw enable

# Firewalld (CentOS)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### 4. File Permissions
```bash
# Set proper ownership
sudo chown -R www-data:www-data /opt/ats-classifier

# Restrict permissions
chmod 755 /opt/ats-classifier
chmod 644 /opt/ats-classifier/*.py
chmod 600 /opt/ats-classifier/.env
chmod 755 /opt/ats-classifier/uploads
chmod 755 /opt/ats-classifier/logs
```

### 5. Rate Limiting
The application includes built-in rate limiting:
- `/analyze`: 10 requests per minute
- `/api/*`: Various limits per endpoint
- Configure in `.env` with `RATELIMIT_DEFAULT`

### 6. Input Validation
The application validates:
- ✅ File types (PDF only)
- ✅ File size (16MB max)
- ✅ PDF integrity
- ✅ Text extraction quality
- ✅ Role selection

---

## 📊 Monitoring & Maintenance

### Health Checks
```bash
# HTTP health check
curl http://localhost:5000/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2025-10-06T12:00:00",
  "models_loaded": true,
  "version": "1.0.0"
}
```

### Log Monitoring
```bash
# Application logs
tail -f logs/ats_app.log

# Gunicorn access logs
tail -f logs/gunicorn_access.log

# Gunicorn error logs
tail -f logs/gunicorn_error.log

# System service logs
sudo journalctl -u ats-classifier -f
```

### Log Rotation
The application automatically rotates logs when they reach 10MB (keeping 5 backups).

For system-level rotation, create `/etc/logrotate.d/ats-classifier`:
```
/opt/ats-classifier/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 www-data www-data
    sharedscripts
    postrotate
        systemctl reload ats-classifier > /dev/null 2>&1 || true
    endscript
}
```

### Performance Monitoring
```bash
# System resources
htop

# Application metrics
curl http://localhost:5000/health

# Docker stats
docker stats ats-classifier
```

### Backup Strategy
```bash
# Backup models and data
tar -czf ats-backup-$(date +%Y%m%d).tar.gz \
  *.pkl *.csv .env

# Backup to remote location
rsync -avz ats-backup-*.tar.gz user@backup-server:/backups/
```

### Update Procedure
```bash
# 1. Backup current version
sudo systemctl stop ats-classifier
tar -czf backup-$(date +%Y%m%d).tar.gz /opt/ats-classifier

# 2. Pull latest changes
cd /opt/ats-classifier
git pull origin main

# 3. Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# 4. Restart service
sudo systemctl start ats-classifier
sudo systemctl status ats-classifier
```

---

## 🐛 Troubleshooting

### Issue: Models Not Loading

**Symptoms**: 503 Service Unavailable, "models not loaded" error

**Solutions**:
```bash
# Check if model files exist
ls -lh *.pkl *.csv

# Check file permissions
chmod 644 *.pkl *.csv

# Check logs for specific error
tail -f logs/ats_app.log

# Verify NLTK data
python -c "import nltk; print(nltk.data.find('tokenizers/punkt'))"
```

### Issue: High Memory Usage

**Symptoms**: Server becomes slow, OOM errors

**Solutions**:
```bash
# Reduce number of Gunicorn workers
# Edit gunicorn_config.py:
workers = 2  # Reduce from default

# Increase system swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue: Rate Limit Errors

**Symptoms**: 429 Too Many Requests

**Solutions**:
```bash
# Adjust rate limits in .env
RATELIMIT_DEFAULT=200 per hour

# For Redis-based rate limiting (production)
RATELIMIT_STORAGE_URL=redis://localhost:6379

# Restart application
sudo systemctl restart ats-classifier
```

### Issue: PDF Extraction Fails

**Symptoms**: "Could not extract text from PDF"

**Solutions**:
- Ensure PDF contains selectable text (not scanned images)
- Try with a different PDF
- Check PDF file size (< 16MB)
- Verify PyPDF2 version: `pip show PyPDF2`

### Issue: Port Already in Use

**Symptoms**: "Address already in use" error

**Solutions**:
```bash
# Find process using port 5000
sudo lsof -i :5000

# Kill process
sudo kill -9 <PID>

# Or use a different port
export FLASK_PORT=5001
```

### Issue: Permission Denied

**Symptoms**: Cannot write to uploads/logs directories

**Solutions**:
```bash
# Fix ownership
sudo chown -R $USER:$USER uploads logs

# Fix permissions
chmod 755 uploads logs
```

---

## 📞 Support & Resources

### Useful Commands

```bash
# Check application status
sudo systemctl status ats-classifier

# Restart application
sudo systemctl restart ats-classifier

# View real-time logs
tail -f logs/ats_app.log

# Test endpoints
curl -X GET http://localhost:5000/health
curl -X GET http://localhost:5000/api/roles

# Docker commands
docker-compose logs -f
docker-compose restart
docker-compose down && docker-compose up -d
```

### Performance Tuning

For high-traffic deployments:

1. **Use Redis for rate limiting**
```bash
# Install Redis
sudo apt-get install redis-server

# Update .env
RATELIMIT_STORAGE_URL=redis://localhost:6379
```

2. **Scale horizontally**
```bash
# Add load balancer (Nginx)
upstream ats_backend {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}
```

3. **Use CDN for static files**
- Deploy static assets to CloudFlare/AWS CloudFront
- Update template references

4. **Database optimization**
- Consider caching role data in Redis
- Pre-compute common metrics

---



**Version**: 1.0.0  
**Last Updated**: October 2025  
