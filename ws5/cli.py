"""
WS5: Command-Line Interface for Circuit Checkpoint Analysis

This module provides a CLI for running analysis on checkpoint directories
and generating structured outputs for further processing.
"""

import click
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import sys

from core import (
    load_checkpoint_circuits, generate_circuit_analysis, compare_attribution_graphs,
    detect_saturation, extract_learning_patterns, save_analysis_results,
    load_model_for_analysis
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """WS5: Circuit Checkpoint Analysis Pipeline."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.option('--checkpoint-dir', '-c', required=True, type=click.Path(exists=True, path_type=Path),
              help='Directory containing circuit checkpoints')
@click.option('--output', '-o', type=click.Path(path_type=Path), 
              help='Output file for analysis results (JSON)')
@click.option('--base-model', '-m', type=click.Path(exists=True, path_type=Path),
              help='Path to base model for loading checkpoints')
@click.option('--test-prompts', '-p', multiple=True, 
              help='Test prompts for circuit analysis (can be specified multiple times)')
@click.option('--constraint-examples', type=click.Path(exists=True, path_type=Path),
              help='JSON file mapping constraint types to example prompts')
@click.option('--use-circuit-tracer', is_flag=True, default=False,
              help='Use circuit tracer for detailed analysis (requires GPU)')
@click.option('--quick', is_flag=True, default=False,
              help='Run quick analysis without loading full models')
def analyze(checkpoint_dir, output, base_model, test_prompts, constraint_examples, 
           use_circuit_tracer, quick):
    """Run complete circuit analysis on checkpoint directory."""
    
    click.echo(f"🔬 Starting WS5 circuit analysis on {checkpoint_dir}")
    start_time = datetime.now()
    
    try:
        # Load checkpoints
        click.echo("📁 Loading checkpoint information...")
        checkpoints = load_checkpoint_circuits(checkpoint_dir)
        
        if not checkpoints:
            click.echo("❌ No checkpoints found in directory", err=True)
            return
        
        click.echo(f"✅ Found {len(checkpoints)} checkpoints")
        for name, info in sorted(checkpoints.items()):
            click.echo(f"   {name}: step {info.step}, progress {info.progress:.1%}")
        
        # Load constraint examples if provided
        constraint_dict = {}
        if constraint_examples:
            with open(constraint_examples) as f:
                constraint_dict = json.load(f)
            click.echo(f"📋 Loaded {len(constraint_dict)} constraint types")
        
        # Set up test prompts
        if not test_prompts and not quick:
            # Use default WS2-style prompts
            test_prompts = [
                "The blarf cat is happy",
                "The gleem day was sad", 
                "The zephyr car goes fast",
                "The glide bird flies upward",
                "The cascade water falls downward"
            ]
            click.echo("📝 Using default test prompts")
        elif test_prompts:
            click.echo(f"📝 Using {len(test_prompts)} provided test prompts")
        
        results = {
            'analysis_info': {
                'timestamp': start_time.isoformat(),
                'checkpoint_dir': str(checkpoint_dir),
                'num_checkpoints': len(checkpoints),
                'use_circuit_tracer': use_circuit_tracer,
                'quick_mode': quick
            },
            'checkpoints': checkpoints,
            'analyses': [],
            'comparisons': [],
            'saturation': {},
            'learning_patterns': {}
        }
        
        if not quick:
            # Run full analysis with model loading
            if not base_model:
                click.echo("❌ Base model path required for full analysis", err=True)
                return
            
            click.echo("🧠 Running circuit analysis on each checkpoint...")
            analyses = []
            
            with click.progressbar(sorted(checkpoints.items()), 
                                 label='Analyzing checkpoints') as items:
                for name, checkpoint_info in items:
                    try:
                        # Load model
                        model = load_model_for_analysis(checkpoint_info, base_model)
                        
                        # Generate analysis
                        analysis = generate_circuit_analysis(
                            model, test_prompts, checkpoint_info, use_circuit_tracer
                        )
                        analyses.append(analysis)
                        
                        # Clean up model to save memory
                        del model
                        
                    except Exception as e:
                        click.echo(f"⚠️  Failed to analyze {name}: {e}", err=True)
                        continue
            
            results['analyses'] = analyses
            
            if len(analyses) >= 2:
                # Run comparisons
                click.echo("🔄 Computing checkpoint comparisons...")
                comparisons = []
                for i in range(len(analyses) - 1):
                    comparison = compare_attribution_graphs(analyses[i], analyses[i + 1])
                    comparisons.append(comparison)
                
                results['comparisons'] = comparisons
                
                # Detect saturation
                click.echo("📈 Detecting circuit saturation...")
                saturation_result = detect_saturation(analyses)
                results['saturation'] = saturation_result
                
                if saturation_result['saturated']:
                    click.echo(f"🎯 Saturation detected at step {saturation_result['saturation_step']}")
                else:
                    click.echo("📊 No saturation detected")
                
                # Extract learning patterns
                if constraint_dict:
                    click.echo("🧬 Extracting learning patterns...")
                    learning_patterns = extract_learning_patterns(analyses, constraint_dict)
                    results['learning_patterns'] = learning_patterns
                    
                    # Report learning insights
                    if learning_patterns.get('learning_order'):
                        click.echo("📚 Learning order detected:")
                        for constraint, improvement in learning_patterns['learning_order']:
                            click.echo(f"   {constraint}: {improvement:.3f} improvement")
        
        else:
            # Quick mode - just checkpoint metadata analysis
            click.echo("⚡ Quick mode: analyzing checkpoint metadata only")
            
            # Simple progression analysis
            sorted_checkpoints = sorted(checkpoints.values(), key=lambda x: x.step)
            loss_trend = [cp.loss for cp in sorted_checkpoints]
            
            if len(loss_trend) >= 2:
                improvement = loss_trend[0] - loss_trend[-1]
                click.echo(f"📉 Loss improvement: {improvement:.3f}")
                results['quick_analysis'] = {
                    'loss_trend': loss_trend,
                    'loss_improvement': improvement,
                    'training_steps': [cp.step for cp in sorted_checkpoints]
                }
        
        # Save results
        if output:
            save_analysis_results(results, output)
            click.echo(f"💾 Results saved to {output}")
        else:
            # Print summary to stdout
            click.echo("\n📊 Analysis Summary:")
            click.echo(f"   Checkpoints analyzed: {len(checkpoints)}")
            if results.get('analyses'):
                click.echo(f"   Circuit analyses: {len(results['analyses'])}")
            if results.get('comparisons'):
                click.echo(f"   Checkpoint comparisons: {len(results['comparisons'])}")
        
        duration = datetime.now() - start_time
        click.echo(f"✅ Analysis complete in {duration.total_seconds():.1f}s")
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        logger.exception("Analysis failed")
        sys.exit(1)

@cli.command()
@click.option('--checkpoint-dir', '-c', required=True, type=click.Path(exists=True, path_type=Path),
              help='Directory containing circuit checkpoints')
def list_checkpoints(checkpoint_dir):
    """List available checkpoints in directory."""
    
    try:
        checkpoints = load_checkpoint_circuits(checkpoint_dir)
        
        if not checkpoints:
            click.echo("No checkpoints found")
            return
        
        click.echo(f"Found {len(checkpoints)} checkpoints:\n")
        
        # Create table
        headers = ["Name", "Step", "Progress", "Epoch", "Loss", "LR", "Timestamp"]
        click.echo(f"{'Name':<20} {'Step':<6} {'Progress':<10} {'Epoch':<8} {'Loss':<8} {'LR':<12} {'Timestamp':<20}")
        click.echo("-" * 95)
        
        for name, info in sorted(checkpoints.items(), key=lambda x: x[1].step):
            click.echo(f"{name:<20} {info.step:<6} {info.progress*100:>7.1f}% "
                      f"{info.epoch:<8.2f} {info.loss:<8.3f} {info.learning_rate:<12.2e} "
                      f"{info.timestamp.split('T')[1][:8]:<20}")
    
    except Exception as e:
        click.echo(f"❌ Failed to list checkpoints: {e}", err=True)

@cli.command()
@click.option('--analysis-file', '-a', required=True, type=click.Path(exists=True, path_type=Path),
              help='Analysis results JSON file')
@click.option('--format', '-f', type=click.Choice(['summary', 'detailed', 'csv']), 
              default='summary', help='Report format')
def report(analysis_file, format):
    """Generate report from analysis results."""
    
    try:
        with open(analysis_file) as f:
            results = json.load(f)
        
        if format == 'summary':
            _print_summary_report(results)
        elif format == 'detailed':
            _print_detailed_report(results)
        elif format == 'csv':
            _print_csv_report(results)
    
    except Exception as e:
        click.echo(f"❌ Failed to generate report: {e}", err=True)

def _print_summary_report(results):
    """Print summary report."""
    click.echo("📊 WS5 Analysis Summary Report")
    click.echo("=" * 40)
    
    info = results.get('analysis_info', {})
    click.echo(f"Timestamp: {info.get('timestamp', 'N/A')}")
    click.echo(f"Checkpoints: {info.get('num_checkpoints', 0)}")
    
    if results.get('saturation', {}).get('saturated'):
        click.echo(f"🎯 Saturation detected at step {results['saturation']['saturation_step']}")
    else:
        click.echo("📈 No saturation detected")
    
    if results.get('learning_patterns', {}).get('learning_order'):
        click.echo("\n📚 Learning Order:")
        for constraint, improvement in results['learning_patterns']['learning_order']:
            click.echo(f"   {constraint}: {improvement:.3f}")

def _print_detailed_report(results):
    """Print detailed report."""
    click.echo("📋 WS5 Detailed Analysis Report")
    click.echo("=" * 50)
    
    # Print all available information
    for section, data in results.items():
        if section != 'analyses':  # Skip heavy analysis data
            click.echo(f"\n{section.upper()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    click.echo(f"  {key}: {value}")
            else:
                click.echo(f"  {data}")

def _print_csv_report(results):
    """Print CSV format report."""
    # Extract checkpoint progression data
    if 'checkpoints' in results:
        click.echo("checkpoint,step,progress,loss,learning_rate")
        checkpoints = results['checkpoints']
        for name, info in sorted(checkpoints.items(), key=lambda x: x[1]['step']):
            click.echo(f"{name},{info['step']},{info['progress']},{info['loss']},{info['learning_rate']}")

@cli.command()
@click.option('--ws2-dataset', type=click.Path(exists=True, path_type=Path),
              help='Path to WS2 synthetic dataset')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output constraint examples JSON file')
def prepare_constraints(ws2_dataset, output):
    """Prepare constraint examples from WS2 dataset for analysis."""
    
    try:
        from datasets import load_from_disk
        
        if ws2_dataset:
            dataset = load_from_disk(str(ws2_dataset))
        else:
            # Try default location
            default_path = Path(__file__).parent.parent / "data" / "ws2_synthetic_corpus_hf"
            if default_path.exists():
                dataset = load_from_disk(str(default_path))
            else:
                click.echo("❌ WS2 dataset not found. Specify path with --ws2-dataset", err=True)
                return
        
        # Extract examples by constraint type
        constraint_examples = {}
        for example in dataset:
            constraint_type = example['constraint_type']
            text = example['text']
            
            if constraint_type not in constraint_examples:
                constraint_examples[constraint_type] = []
            
            constraint_examples[constraint_type].append(text)
        
        # Limit to reasonable number of examples per type
        for constraint_type in constraint_examples:
            constraint_examples[constraint_type] = constraint_examples[constraint_type][:10]
        
        if output:
            with open(output, 'w') as f:
                json.dump(constraint_examples, f, indent=2)
            click.echo(f"💾 Constraint examples saved to {output}")
        else:
            click.echo("📋 Constraint Examples:")
            for constraint_type, examples in constraint_examples.items():
                click.echo(f"\n{constraint_type} ({len(examples)} examples):")
                for i, example in enumerate(examples[:3]):  # Show first 3
                    click.echo(f"  {i+1}. {example}")
                if len(examples) > 3:
                    click.echo(f"  ... and {len(examples)-3} more")
    
    except Exception as e:
        click.echo(f"❌ Failed to prepare constraints: {e}", err=True)

if __name__ == '__main__':
    cli()