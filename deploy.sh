#!/bin/bash

# Production Deployment Script for Asset-ML-Strategy
# This script sets up and deploys the complete production environment

set -e  # Exit on any error

echo "🚀 Asset-ML-Strategy Production Deployment Script"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="asset-ml-strategy"
DOCKER_REGISTRY="ghcr.io"
DOMAIN="${DOMAIN:-localhost}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker and Docker Compose are available${NC}"
}

# Create necessary directories
create_directories() {
    echo -e "${BLUE}📁 Creating necessary directories...${NC}"
    
    mkdir -p logs/{api,trading,nginx}
    mkdir -p data/{market_data,models,backups}
    mkdir -p ssl
    mkdir -p monitoring/grafana/provisioning/{datasources,dashboards}
    
    echo -e "${GREEN}✅ Directories created${NC}"
}

# Generate SSL certificates (self-signed for development)
generate_ssl() {
    if [ ! -f "nginx/ssl/cert.pem" ]; then
        echo -e "${BLUE}🔒 Generating SSL certificates...${NC}"
        
        mkdir -p nginx/ssl
        
        openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=${DOMAIN}"
        
        echo -e "${GREEN}✅ SSL certificates generated${NC}"
    else
        echo -e "${YELLOW}⚠️  SSL certificates already exist${NC}"
    fi
}

# Create environment file if it doesn't exist
create_env_file() {
    if [ ! -f ".env" ]; then
        echo -e "${BLUE}⚙️  Creating .env file...${NC}"
        
        cat > .env << EOF
# Database Configuration
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
POSTGRES_DB=assetML
POSTGRES_USER=assetML_user

# Redis Configuration
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# JWT Configuration
JWT_SECRET=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-64)

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend Configuration
REACT_APP_API_URL=http://${DOMAIN}:8000
REACT_APP_WS_URL=ws://${DOMAIN}:8000/ws

# Monitoring Configuration
GRAFANA_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Environment
ENVIRONMENT=${ENVIRONMENT}
DOMAIN=${DOMAIN}

# Optional: External API Keys (uncomment and fill in if needed)
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
# TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
# TELEGRAM_CHAT_ID=your_telegram_chat_id_here
# SENDGRID_API_KEY=your_sendgrid_api_key_here
# SLACK_WEBHOOK_URL=your_slack_webhook_url_here
EOF
        
        echo -e "${GREEN}✅ .env file created with secure random passwords${NC}"
        echo -e "${YELLOW}⚠️  Please review and update .env file with your specific configuration${NC}"
    else
        echo -e "${YELLOW}⚠️  .env file already exists, skipping creation${NC}"
    fi
}

# Create Grafana dashboards configuration
setup_monitoring() {
    echo -e "${BLUE}📊 Setting up monitoring configuration...${NC}"
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'trading-engine'
    static_configs:
      - targets: ['trading_engine:8001']
    metrics_path: '/metrics'
    scrape_interval: 5s
EOF

    # Grafana datasources configuration
    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    version: 1
    editable: true
EOF

    echo -e "${GREEN}✅ Monitoring configuration created${NC}"
}

# Build and deploy services
deploy_services() {
    echo -e "${BLUE}🏗️  Building and deploying services...${NC}"
    
    # Load environment variables
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '#' | xargs)
    fi
    
    # Build and start services
    echo -e "${BLUE}Building Docker images...${NC}"
    docker-compose build --parallel
    
    echo -e "${BLUE}Starting services...${NC}"
    docker-compose up -d
    
    # Wait for services to be ready
    echo -e "${BLUE}Waiting for services to be ready...${NC}"
    sleep 30
    
    echo -e "${GREEN}✅ Services deployed successfully${NC}"
}

# Health check
health_check() {
    echo -e "${BLUE}🏥 Performing health checks...${NC}"
    
    # Check API health
    if curl -f -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✅ API is healthy${NC}"
    else
        echo -e "${RED}❌ API health check failed${NC}"
    fi
    
    # Check frontend
    if curl -f -s http://localhost:3000 > /dev/null; then
        echo -e "${GREEN}✅ Frontend is accessible${NC}"
    else
        echo -e "${RED}❌ Frontend health check failed${NC}"
    fi
    
    # Check database
    if docker-compose exec -T postgres pg_isready -U assetML_user > /dev/null; then
        echo -e "${GREEN}✅ Database is ready${NC}"
    else
        echo -e "${RED}❌ Database health check failed${NC}"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping > /dev/null; then
        echo -e "${GREEN}✅ Redis is ready${NC}"
    else
        echo -e "${RED}❌ Redis health check failed${NC}"
    fi
}

# Display deployment summary
show_summary() {
    echo ""
    echo -e "${GREEN}🎉 Deployment Complete!${NC}"
    echo "=========================="
    echo ""
    echo "🌐 Application URLs:"
    echo "   Frontend:        http://${DOMAIN}:3000"
    echo "   API:             http://${DOMAIN}:8000"
    echo "   API Docs:        http://${DOMAIN}:8000/docs"
    echo "   Dashboard:       http://${DOMAIN}:8501 (Streamlit)"
    echo "   Prometheus:      http://${DOMAIN}:9090"
    echo "   Grafana:         http://${DOMAIN}:3001"
    echo ""
    echo "🔧 Management Commands:"
    echo "   View logs:       docker-compose logs -f [service]"
    echo "   Stop services:   docker-compose down"
    echo "   Restart:         docker-compose restart [service]"
    echo "   Update:          docker-compose pull && docker-compose up -d"
    echo ""
    echo "📁 Important Paths:"
    echo "   Logs:            ./logs/"
    echo "   Data:            ./data/"
    echo "   Configuration:   ./config/"
    echo "   SSL Certificates: ./nginx/ssl/"
    echo ""
    echo "🔐 Security:"
    echo "   - SSL certificates generated for HTTPS"
    echo "   - Random passwords generated in .env file"
    echo "   - Review firewall settings for production"
    echo ""
    echo -e "${YELLOW}⚠️  Next Steps:${NC}"
    echo "1. Review and update .env file with your configuration"
    echo "2. Configure external API keys if needed"
    echo "3. Set up backup procedures for data and database"
    echo "4. Configure monitoring alerts"
    echo "5. Set up proper SSL certificates for production"
}

# Main deployment function
main() {
    echo -e "${BLUE}Starting deployment process...${NC}"
    
    check_docker
    create_directories
    create_env_file
    generate_ssl
    setup_monitoring
    deploy_services
    health_check
    show_summary
    
    echo -e "${GREEN}🚀 Asset-ML-Strategy is now running in ${ENVIRONMENT} mode!${NC}"
}

# Run deployment
main "$@"