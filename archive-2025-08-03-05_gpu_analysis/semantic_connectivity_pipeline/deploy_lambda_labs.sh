#!/bin/bash
# Deploy and run semantic connectivity pipeline on Lambda Labs
# Optimized for GPU instances with fault tolerance

set -e

echo "=== Lambda Labs Deployment Script ==="
echo "This script will deploy and run the pipeline on a Lambda Labs instance"
echo

# Configuration
INSTANCE_IP="${1}"
SSH_KEY="${2:-~/.ssh/id_rsa}"
REMOTE_USER="${3:-ubuntu}"
REPO_URL="${4:-https://github.com/yourusername/learningSlice.git}"

# Check arguments
if [ -z "$INSTANCE_IP" ]; then
    echo "Usage: $0 <instance_ip> [ssh_key] [remote_user] [repo_url]"
    echo "Example: $0 123.45.67.89 ~/.ssh/lambda_key.pem"
    exit 1
fi

# Create deployment package
echo "Creating deployment package..."
DEPLOY_DIR="deployment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEPLOY_DIR"

# Copy essential files
cp -r ../semantic_connectivity_pipeline/* "$DEPLOY_DIR/"
cp ../README.md "$DEPLOY_DIR/"
cp ../CLAUDE.md "$DEPLOY_DIR/" 2>/dev/null || true

# Create remote setup script
cat > "$DEPLOY_DIR/remote_setup.sh" << 'EOF'
#!/bin/bash
# Remote setup script for Lambda Labs

set -e

echo "=== Setting up on Lambda Labs instance ==="

# Update system
sudo apt-get update
sudo apt-get install -y git tmux htop nvtop python3-pip python3-venv

# Check GPU
nvidia-smi
echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Setup pipeline
cd semantic_connectivity_pipeline
chmod +x *.sh *.py

# Run setup
./setup_pipeline.sh

# Activate environment
source venv/bin/activate

# Test installation
python test_installation.py

echo "=== Setup Complete ==="
echo ""
echo "To start a job:"
echo "  cd semantic_connectivity_pipeline"
echo "  ./run_overnight_job.sh <job_name> <job_type>"
echo ""
echo "Example:"
echo "  ./run_overnight_job.sh gemma_5k_analysis full"
EOF

# Create Lambda-specific job runner
cat > "$DEPLOY_DIR/run_lambda_job.sh" << 'EOF'
#!/bin/bash
# Lambda Labs optimized job runner

set -e

JOB_NAME="${1:-lambda_job}"
JOB_TYPE="${2:-full}"
MODEL_URL="${3}"

# Create job directory
JOB_DIR="jobs/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$JOB_DIR"

# Download model if URL provided
if [ -n "$MODEL_URL" ]; then
    echo "Downloading model from $MODEL_URL..."
    mkdir -p models
    cd models
    wget -c "$MODEL_URL" -O model.tar.gz
    tar -xzf model.tar.gz
    cd ..
    export MODEL_PATH="$(pwd)/models"
fi

# GPU optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run with optimal settings for Lambda Labs
./run_overnight_job.sh "$JOB_NAME" "$JOB_TYPE"

# Monitor GPU usage
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 60 > "$JOB_DIR/gpu_usage.csv" &

echo "Job started. GPU monitoring saved to $JOB_DIR/gpu_usage.csv"
EOF

chmod +x "$DEPLOY_DIR/remote_setup.sh"
chmod +x "$DEPLOY_DIR/run_lambda_job.sh"

# Create monitoring dashboard
cat > "$DEPLOY_DIR/monitor_dashboard.sh" << 'EOF'
#!/bin/bash
# Simple monitoring dashboard for Lambda Labs

while true; do
    clear
    echo "=== Lambda Labs Pipeline Monitor ==="
    echo "Time: $(date)"
    echo ""
    
    # GPU status
    echo "--- GPU Status ---"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    echo ""
    
    # Running jobs
    echo "--- Running Jobs ---"
    tmux ls 2>/dev/null | grep pipeline || echo "No pipeline jobs running"
    echo ""
    
    # Disk usage
    echo "--- Disk Usage ---"
    df -h | grep -E "Filesystem|/$"
    echo ""
    
    # Recent logs
    echo "--- Recent Activity ---"
    find logs -name "*.log" -mmin -10 -exec tail -5 {} \; 2>/dev/null | tail -20
    
    sleep 30
done
EOF

chmod +x "$DEPLOY_DIR/monitor_dashboard.sh"

# Deploy to Lambda Labs
echo -e "\nDeploying to Lambda Labs instance..."

# Create remote directory
ssh -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP" "mkdir -p ~/semantic_connectivity_pipeline"

# Copy files
echo "Copying files..."
scp -i "$SSH_KEY" -r "$DEPLOY_DIR"/* "$REMOTE_USER@$INSTANCE_IP:~/semantic_connectivity_pipeline/"

# Run remote setup
echo -e "\nRunning remote setup..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP" "cd semantic_connectivity_pipeline && bash remote_setup.sh"

# Create SSH config entry
echo -e "\nCreating SSH config entry..."
cat >> ~/.ssh/config << EOF

Host lambda-pipeline
    HostName $INSTANCE_IP
    User $REMOTE_USER
    IdentityFile $SSH_KEY
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF

echo "=== Deployment Complete ==="
echo ""
echo "To connect to your instance:"
echo "  ssh lambda-pipeline"
echo ""
echo "To start a job:"
echo "  ssh lambda-pipeline"
echo "  cd semantic_connectivity_pipeline"
echo "  ./run_lambda_job.sh my_analysis full"
echo ""
echo "To monitor:"
echo "  ssh lambda-pipeline 'cd semantic_connectivity_pipeline && ./monitor_dashboard.sh'"
echo ""
echo "To check job status:"
echo "  ssh lambda-pipeline 'cd semantic_connectivity_pipeline && python manage_jobs.py list'"

# Cleanup
rm -rf "$DEPLOY_DIR"