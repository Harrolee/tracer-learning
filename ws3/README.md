# WS3: Fine-Tuning Pipeline for Gemma-2B

This workstream implements a fine-tuning pipeline for Gemma-2B with checkpoint saving for circuit analysis.

## Setup

1. **Install dependencies** (when you have good internet):
```bash
source /Users/lee/fun/learningSlice/.venv/bin/activate
pip install -r requirements.txt
```

2. **Authenticate with HuggingFace** (for Gemma access):
```bash
huggingface-cli login
```

## Usage

### 1. Test Model Loading

First, verify that Gemma-2B loads correctly:

```bash
python test_model.py
```

This will:
- Check GPU availability
- Estimate memory requirements
- Test base model loading and inference
- Verify checkpoint loading capabilities

### 2. Prepare Dataset

If you have a synthetic corpus from WS2:

```bash
python data_utils.py ../ws2/synthetic_corpus.json --output_dir ./data
```

### 3. Run Fine-Tuning

**Test run (10 examples):**
```bash
python finetune_gemma.py \
    --data_path ./data/dataset \
    --output_dir ./outputs \
    --test_run \
    --use_lora
```

**Full training:**
```bash
python finetune_gemma.py \
    --data_path ./data/dataset \
    --output_dir ./outputs \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --use_lora
```

### 4. Monitor Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir outputs/run_*/logs
```

## Checkpoints

The pipeline saves checkpoints at:
- 25% training progress
- 50% training progress  
- 75% training progress
- 100% training progress (final)

Checkpoints are saved in: `outputs/run_*/circuit_checkpoints/`

Each checkpoint includes:
- Model weights (adapter weights if using LoRA)
- Training state (step, loss, learning rate)

## Configuration

See `config.yaml` for recommended hyperparameters. Key settings:

- **LoRA**: Recommended for efficiency (r=8, target attention modules)
- **Batch size**: 4 (adjust based on GPU memory)
- **Learning rate**: 2e-5 for full fine-tuning, 5e-6 for small datasets
- **FP16**: Enabled for memory efficiency

## Resource Requirements

- **Minimum**: 16GB GPU memory (with LoRA)
- **Recommended**: 24GB GPU memory
- **Storage**: ~2GB per checkpoint (with LoRA)

## Testing Checkpoints

To test a saved checkpoint:

```bash
python test_model.py --checkpoint outputs/run_*/circuit_checkpoints/checkpoint-50pct
```

## Troubleshooting

1. **OOM Errors**: 
   - Reduce batch_size
   - Enable gradient_checkpointing
   - Use LoRA instead of full fine-tuning

2. **Slow Training**:
   - Ensure GPU is being used
   - Check FP16 is enabled
   - Consider reducing max_length

3. **Authentication Issues**:
   - Run `huggingface-cli login`
   - Ensure you have access to google/gemma-2b

## Next Steps

After fine-tuning completes, the checkpoints are ready for circuit analysis in WS5.