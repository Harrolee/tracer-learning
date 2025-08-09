# Semantic Connectivity Pipeline

Self-contained pipeline for analyzing semantic connectivity evolution and circuit complexity in transformer models.

## Quick Start (Local)

```bash
# Setup (one-time, ~5 minutes)
./setup_pipeline.sh
source activate_pipeline.sh

# Run example analysis (10 minutes)
./run_example_analysis.sh

# Run overnight job
./run_overnight_job.sh my_5k_analysis full
```

## Quick Start (Lambda Labs)

```bash
# Deploy to Lambda Labs instance
./deploy_lambda_labs.sh <instance_ip> ~/.ssh/your_key.pem

# SSH to instance
ssh lambda-pipeline

# Start job
cd semantic_connectivity_pipeline
./run_lambda_job.sh gemma_5k full
```

## Files Overview

### Core Pipeline
- `precompute_dictionary_embeddings.py` - Precompute embeddings for entire dictionary
- `unified_analysis_pipeline.py` - Main analysis combining connectivity + features
- `setup_pipeline.sh` - Install all dependencies
- `run_example_analysis.sh` - Quick demo with 100 words

### Job Management
- `run_overnight_job.sh` - Run long jobs in tmux (survives disconnection)
- `manage_jobs.py` - Monitor and control running jobs
- `deploy_lambda_labs.sh` - Deploy to Lambda Labs GPU instance
- `run_lambda_job.sh` - Lambda-optimized job runner

## Running Overnight Jobs

The pipeline uses tmux for reliable overnight execution:

```bash
# Start a full analysis job
./run_overnight_job.sh experiment_1 full

# This will:
# 1. Create a tmux session named 'pipeline_experiment_1_<timestamp>'
# 2. Run the job with full logging
# 3. Save checkpoints for resumability
# 4. Continue running if you disconnect

# Monitor the job
tmux attach -t pipeline_experiment_1_<tab to complete>

# Check status
python manage_jobs.py list
python manage_jobs.py status experiment_1

# View logs
tail -f logs/experiment_1_*/analysis.log
```

## Job Types

1. **precompute** - Build dictionary embeddings (4-6 hours, one-time)
   ```bash
   ./run_overnight_job.sh dict_embeddings precompute
   ```

2. **analyze** - Run connectivity analysis (1-2 hours)
   ```bash
   ./run_overnight_job.sh test_5k analyze
   ```

3. **full** - Run complete pipeline (5-8 hours)
   ```bash
   ./run_overnight_job.sh complete_analysis full
   ```

## Lambda Labs Deployment

For GPU acceleration on Lambda Labs:

1. **Deploy**:
   ```bash
   ./deploy_lambda_labs.sh 123.45.67.89 ~/.ssh/lambda_key.pem
   ```

2. **Start job on remote**:
   ```bash
   ssh lambda-pipeline
   cd semantic_connectivity_pipeline
   ./run_lambda_job.sh large_experiment full
   ```

3. **Monitor remotely**:
   ```bash
   # Watch live dashboard
   ssh lambda-pipeline 'cd semantic_connectivity_pipeline && ./monitor_dashboard.sh'
   
   # Check job status
   ssh lambda-pipeline 'cd semantic_connectivity_pipeline && python manage_jobs.py list'
   ```

## Fault Tolerance

The pipeline includes several fault-tolerance features:

1. **Checkpointing**: Dictionary precomputation saves progress every 5000 words
2. **Tmux sessions**: Jobs continue running after SSH disconnection
3. **Comprehensive logging**: All output saved to timestamped log files
4. **Resume capability**: Can restart from checkpoints if interrupted
5. **Error handling**: Graceful failure with detailed error logs

## Managing Jobs

Use `manage_jobs.py` for job control:

```bash
# List all jobs (running and completed)
python manage_jobs.py list

# Check specific job status
python manage_jobs.py status experiment_1

# Analyze results
python manage_jobs.py analyze results_experiment_1

# Kill a running job
python manage_jobs.py kill pipeline_experiment_1_20240327_143022
```

## Output Structure

```
semantic_connectivity_pipeline/
├── logs/                           # Job logs
│   └── experiment_1_20240327_1430/
│       ├── job_script.sh          # Exact commands run
│       ├── precompute.log         # Dictionary embedding logs
│       ├── analysis.log           # Analysis logs
│       └── summary.txt            # Job summary
├── dictionary_embeddings/          # Precomputed embeddings
│   ├── embeddings_layer_0.pkl
│   └── ...
├── results_experiment_1/           # Analysis results
│   ├── word_summary.csv
│   ├── layer_connectivity.csv
│   ├── feature_activations.csv
│   └── connectivity_trajectories.csv
└── jobs/                          # Lambda Labs job tracking
```

## Troubleshooting

**Job won't start**: Check tmux is installed: `sudo apt-get install tmux`

**Out of memory**: Reduce batch size in job script or use CPU

**Can't attach to tmux**: List sessions with `tmux ls`, attach with full name

**Job died unexpectedly**: Check logs in `logs/job_name/`

**Lambda Labs specific**:
- Ensure port 22 is open for SSH
- Use `nvtop` to monitor GPU usage
- Check disk space with `df -h`

## Tips for Large Runs

1. **Start with small test**: Run 100-word analysis first
2. **Use GPU**: 10-50x faster than CPU
3. **Monitor resources**: Watch GPU/CPU usage and disk space
4. **Schedule wisely**: Start jobs before leaving for the day
5. **Check progress**: Logs update in real-time

## Example Workflow

```bash
# Day 1: Deploy and test
./deploy_lambda_labs.sh gpu-instance.lambda ~/.ssh/key.pem
ssh lambda-pipeline
cd semantic_connectivity_pipeline
./run_example_analysis.sh  # Quick test

# Night 1: Precompute dictionary
./run_overnight_job.sh dictionary_prep precompute
# Go home, let it run overnight

# Day 2: Check and run analysis
python manage_jobs.py status dictionary_prep
./run_overnight_job.sh main_analysis analyze

# Day 3: Get results
python manage_jobs.py analyze results_main_analysis
scp -r lambda-pipeline:~/semantic_connectivity_pipeline/results_main_analysis ./
```