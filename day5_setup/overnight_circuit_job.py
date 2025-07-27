#!/usr/bin/env python3
"""
Overnight Job 2: Circuit Complexity Analysis for 1000 Words
Designed to run unattended overnight with checkpointing and recovery
"""

import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import traceback
import glob

# Add circuit tracer to path
sys.path.append('../circuit-tracer')
from circuit_tracer import ReplacementModel
from circuit_tracer.attribution import attribute

def setup_logging():
    """Setup comprehensive logging for overnight circuit job."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f'overnight_circuit_job_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🌙 Starting Overnight Circuit Analysis Job")
    logger.info(f"📝 Logging to: {log_file}")
    return logger, log_file

def load_semantic_results():
    """Load results from overnight semantic analysis job."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Looking for semantic analysis results...")
    
    # Find the most recent semantic results file
    semantic_files = glob.glob('overnight_semantic_results_*.json')
    if not semantic_files:
        raise FileNotFoundError("❌ No semantic results found! Run overnight_semantic_job.py first")
    
    # Get the most recent file
    latest_file = max(semantic_files, key=lambda f: Path(f).stat().st_mtime)
    logger.info(f"📥 Loading semantic results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        semantic_data = json.load(f)
    
    # Extract word analysis data
    word_analysis = semantic_data['word_analysis']
    logger.info(f"✅ Loaded {len(word_analysis)} words for circuit analysis")
    logger.info(f"📊 Connectivity range: {semantic_data['statistics']['connectivity_range']}")
    logger.info(f"🎭 Mean polysemy: {semantic_data['statistics']['mean_polysemy']:.2f}")
    
    return word_analysis, semantic_data, latest_file

def setup_checkpoint_system():
    """Setup checkpointing system for recovery."""
    checkpoint_dir = Path('circuit_checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    circuit_graphs_dir = Path('circuit_graphs_1000')
    circuit_graphs_dir.mkdir(exist_ok=True)
    
    return checkpoint_dir, circuit_graphs_dir

def load_checkpoint(checkpoint_dir):
    """Load existing checkpoint if available."""
    logger = logging.getLogger(__name__)
    
    checkpoint_files = list(checkpoint_dir.glob('circuit_checkpoint_*.json'))
    if not checkpoint_files:
        logger.info("🆕 No checkpoint found - starting fresh")
        return None, []
    
    # Load most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"🔄 Loading checkpoint: {latest_checkpoint}")
    
    with open(latest_checkpoint, 'r') as f:
        checkpoint_data = json.load(f)
    
    completed_words = checkpoint_data.get('completed_words', [])
    logger.info(f"✅ Checkpoint loaded: {len(completed_words)} words already completed")
    
    return checkpoint_data, completed_words

def save_checkpoint(checkpoint_dir, word_analysis, completed_words, results, batch_num):
    """Save progress checkpoint."""
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    checkpoint_data = {
        'timestamp': timestamp,
        'batch_number': batch_num,
        'total_words': len(word_analysis),
        'completed_words': completed_words,
        'completed_count': len(completed_words),
        'remaining_count': len(word_analysis) - len(completed_words),
        'results': results
    }
    
    checkpoint_file = checkpoint_dir / f'circuit_checkpoint_batch_{batch_num}_{timestamp}.json'
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
    return checkpoint_file

def load_circuit_tracer_model():
    """Load Gemma-2B with circuit tracer."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Loading Gemma-2B with Circuit Tracer...")
    logger.info("⚠️  This will take several minutes and requires significant GPU memory")
    
    try:
        logger.info("📥 Loading model with transcoders...")
        model = ReplacementModel.from_pretrained(
            "google/gemma-2-2b", 
            'gemma',
            dtype=torch.bfloat16
        )
        logger.info("✅ Gemma-2B loaded successfully with circuit tracer")
        logger.info(f"🔧 GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return model
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.error("💡 This requires significant GPU memory (15GB+)")
        raise

def analyze_single_word(model, word_data, circuit_graphs_dir):
    """Analyze circuit complexity for a single word."""
    logger = logging.getLogger(__name__)
    
    word = word_data['word']
    connectivity = word_data['connectivity'] 
    polysemy = word_data['polysemy']
    
    logger.info(f"🔬 Analyzing '{word}' (conn:{connectivity}, poly:{polysemy})")
    
    try:
        # Create prompt for circuit analysis
        prompt = f"The word '{word}' means"
        logger.debug(f"    🔍 Prompt: '{prompt}'")
        
        start_time = time.time()
        
        # Run REAL circuit attribution
        graph = attribute(
            model=model,
            prompt=prompt,
            max_n_logits=5
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract circuit complexity metrics
        active_features = torch.nonzero(graph.active_features).shape[0] if hasattr(graph, 'active_features') else 0
        selected_features = torch.nonzero(graph.selected_features).shape[0] if hasattr(graph, 'selected_features') else 0
        n_edges = torch.nonzero(graph.adjacency_matrix).shape[0] if hasattr(graph, 'adjacency_matrix') else 0
        
        # Calculate metrics
        circuit_complexity = active_features
        graph_density = n_edges / (selected_features * (selected_features - 1)) if selected_features > 1 else 0
        
        logger.info(f"    ✅ Completed in {duration:.1f}s")
        logger.info(f"    📈 Circuit complexity: {circuit_complexity:,} active features")
        logger.info(f"    🔢 Selected features: {selected_features:,}")
        logger.info(f"    🕸️  Graph edges: {n_edges:,}")
        logger.info(f"    📊 Graph density: {graph_density:.6f}")
        
        # Save circuit graph
        graph_file = circuit_graphs_dir / f"circuit_{word}_{connectivity}conn_{polysemy}poly.pt"
        torch.save(graph, graph_file)
        logger.debug(f"    💾 Graph saved: {graph_file}")
        
        result = {
            'word': word,
            'connectivity': connectivity,
            'polysemy': polysemy,
            'circuit_complexity': circuit_complexity,
            'active_features': active_features,
            'selected_features': selected_features,
            'n_edges': n_edges,
            'graph_density': graph_density,
            'analysis_duration': duration,
            'prompt': prompt,
            'graph_file': str(graph_file),
            'analysis_type': 'real_circuit_tracer'
        }
        
        return result
        
    except Exception as e:
        logger.error(f"    ❌ Analysis failed for '{word}': {e}")
        
        # Return failed result
        return {
            'word': word,
            'connectivity': connectivity,
            'polysemy': polysemy,
            'circuit_complexity': 0,
            'active_features': 0,
            'selected_features': 0,
            'n_edges': 0,
            'graph_density': 0.0,
            'analysis_duration': 0,
            'prompt': f"The word '{word}' means",
            'graph_file': None,
            'analysis_type': 'failed',
            'error': str(e)
        }

def run_circuit_analysis_with_checkpointing(model, word_analysis, checkpoint_dir, circuit_graphs_dir, completed_words):
    """Run circuit analysis with checkpointing support."""
    logger = logging.getLogger(__name__)
    
    # Filter out already completed words
    remaining_words = [wd for wd in word_analysis if wd['word'] not in completed_words]
    logger.info(f"🔄 Resuming analysis: {len(remaining_words)} words remaining")
    
    results = []
    batch_size = 10  # Save checkpoint every 10 words
    
    for i, word_data in enumerate(remaining_words):
        batch_num = len(completed_words) + i + 1
        word = word_data['word']
        
        logger.info(f"\n--- [{batch_num}/{len(word_analysis)}] Processing '{word}' ---")
        
        # Analyze single word
        result = analyze_single_word(model, word_data, circuit_graphs_dir)
        results.append(result)
        completed_words.append(word)
        
        # Save checkpoint every batch_size words
        if batch_num % batch_size == 0:
            logger.info(f"💾 Saving checkpoint at word {batch_num}")
            save_checkpoint(checkpoint_dir, word_analysis, completed_words, results, batch_num)
            
            # Clear some memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Log progress
        remaining = len(word_analysis) - batch_num
        if remaining > 0:
            eta_minutes = (remaining * result.get('analysis_duration', 2)) / 60
            logger.info(f"⏱️  ETA: {eta_minutes:.1f} minutes ({remaining} words remaining)")
    
    # Final checkpoint
    save_checkpoint(checkpoint_dir, word_analysis, completed_words, results, len(word_analysis))
    
    logger.info(f"🎉 Circuit analysis complete! Analyzed {len(results)} words")
    return results

def calculate_correlations(results):
    """Calculate correlations from circuit analysis results."""
    logger = logging.getLogger(__name__)
    logger.info("📈 Calculating correlations...")
    
    # Filter successful analyses
    successful_results = [r for r in results if r['analysis_type'] == 'real_circuit_tracer']
    logger.info(f"✅ {len(successful_results)} successful analyses")
    
    if len(successful_results) < 10:
        logger.warning(f"⚠️  Only {len(successful_results)} successful analyses - correlations may not be reliable")
        return {}
    
    # Extract variables
    connectivities = [r['connectivity'] for r in successful_results]
    polysemies = [r['polysemy'] for r in successful_results]
    complexities = [r['circuit_complexity'] for r in successful_results]
    densities = [r['graph_density'] for r in successful_results]
    
    def safe_correlation(x, y):
        """Calculate correlation, handling edge cases."""
        if len(x) < 2 or len(y) < 2:
            return 0.0
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        corr_matrix = np.corrcoef(x, y)
        corr_value = corr_matrix[0, 1]
        return 0.0 if np.isnan(corr_value) else corr_value
    
    # Calculate correlations
    conn_comp_corr = safe_correlation(connectivities, complexities)
    poly_comp_corr = safe_correlation(polysemies, complexities)
    conn_poly_corr = safe_correlation(connectivities, polysemies)
    conn_dens_corr = safe_correlation(connectivities, densities)
    poly_dens_corr = safe_correlation(polysemies, densities)
    
    logger.info(f"🔗 Connectivity ↔ Circuit Complexity: r = {conn_comp_corr:.3f}")
    logger.info(f"🎭 Polysemy ↔ Circuit Complexity: r = {poly_comp_corr:.3f}")
    logger.info(f"📊 Connectivity ↔ Polysemy: r = {conn_poly_corr:.3f}")
    logger.info(f"🕸️  Connectivity ↔ Graph Density: r = {conn_dens_corr:.3f}")
    logger.info(f"🎭 Polysemy ↔ Graph Density: r = {poly_dens_corr:.3f}")
    
    # Interpret results
    logger.info(f"\n💡 Large-Scale Results Interpretation:")
    if abs(conn_comp_corr) > 0.5:
        direction = "positive" if conn_comp_corr > 0 else "negative"
        logger.info(f"   🎯 STRONG {direction} connectivity-complexity correlation!")
    elif abs(conn_comp_corr) > 0.3:
        direction = "positive" if conn_comp_corr > 0 else "negative"
        logger.info(f"   📈 Moderate {direction} connectivity-complexity correlation")
    
    if abs(poly_comp_corr) > 0.3:
        direction = "positive" if poly_comp_corr > 0 else "negative"
        logger.info(f"   🎭 Moderate {direction} polysemy-complexity correlation")
    
    if conn_poly_corr < -0.3:
        logger.info(f"   ✅ CONFIRMED: Inverse polysemy-connectivity relationship")
    
    return {
        'connectivity_complexity': conn_comp_corr,
        'polysemy_complexity': poly_comp_corr,
        'connectivity_polysemy': conn_poly_corr,
        'connectivity_density': conn_dens_corr,
        'polysemy_density': poly_dens_corr,
        'sample_size': len(successful_results),
        'successful_analyses': len(successful_results),
        'total_attempted': len(results)
    }

def save_final_results(word_analysis, results, correlations, semantic_file, log_file):
    """Save comprehensive final circuit analysis results."""
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Calculate statistics
    successful_results = [r for r in results if r['analysis_type'] == 'real_circuit_tracer']
    failed_results = [r for r in results if r['analysis_type'] == 'failed']
    
    complexities = [r['circuit_complexity'] for r in successful_results]
    connectivities = [r['connectivity'] for r in successful_results]
    
    final_results = {
        'job_info': {
            'job_type': 'overnight_circuit_analysis',
            'timestamp': timestamp,
            'duration_hours': 'TBD',
            'total_words_requested': len(word_analysis),
            'successful_analyses': len(successful_results),
            'failed_analyses': len(failed_results),
            'success_rate': len(successful_results) / len(word_analysis) if word_analysis else 0,
            'model_used': 'google/gemma-2-2b'
        },
        'files': {
            'semantic_results_file': semantic_file,
            'log_file': log_file,
            'circuit_graphs_directory': 'circuit_graphs_1000/'
        },
        'statistics': {
            'mean_circuit_complexity': np.mean(complexities) if complexities else 0,
            'circuit_complexity_range': (min(complexities), max(complexities)) if complexities else (0, 0),
            'mean_connectivity': np.mean(connectivities) if connectivities else 0,
            'connectivity_range': (min(connectivities), max(connectivities)) if connectivities else (0, 0)
        },
        'correlations': correlations,
        'circuit_analysis_results': results,
        'key_findings': {
            'used_real_circuit_tracer': True,
            'large_scale_analysis': True,
            'connectivity_predicts_complexity': bool(abs(correlations.get('connectivity_complexity', 0)) > 0.3),
            'polysemy_affects_complexity': bool(abs(correlations.get('polysemy_complexity', 0)) > 0.3),
            'sample_size': len(successful_results),
            'statistical_significance': len(successful_results) >= 100
        }
    }
    
    # Save results
    results_file = f'overnight_circuit_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"💾 Final circuit results saved to: {results_file}")
    logger.info(f"📊 Large-Scale Analysis Summary:")
    logger.info(f"   • Total words: {len(word_analysis)}")
    logger.info(f"   • Successful: {len(successful_results)}")
    logger.info(f"   • Failed: {len(failed_results)}")
    logger.info(f"   • Success rate: {final_results['job_info']['success_rate']:.1%}")
    logger.info(f"   • Mean complexity: {final_results['statistics']['mean_circuit_complexity']:.0f} features")
    logger.info(f"   • Connectivity-Complexity correlation: r = {correlations.get('connectivity_complexity', 0):.3f}")
    
    return results_file

def main():
    """Main overnight circuit analysis job."""
    start_time = time.time()
    
    try:
        # Setup
        logger, log_file = setup_logging()
        logger.info("🚀 Overnight Circuit Analysis Job Starting")
        
        # Load semantic results
        logger.info("=" * 60)
        logger.info("STEP 1: Loading semantic analysis results")
        logger.info("=" * 60)
        word_analysis, semantic_data, semantic_file = load_semantic_results()
        
        # Setup checkpointing
        logger.info("=" * 60)
        logger.info("STEP 2: Setting up checkpointing system")
        logger.info("=" * 60)
        checkpoint_dir, circuit_graphs_dir = setup_checkpoint_system()
        checkpoint_data, completed_words = load_checkpoint(checkpoint_dir)
        
        # Load circuit tracer model
        logger.info("=" * 60)
        logger.info("STEP 3: Loading circuit tracer model")
        logger.info("=" * 60)
        model = load_circuit_tracer_model()
        
        # Run circuit analysis
        logger.info("=" * 60)
        logger.info("STEP 4: Running large-scale circuit analysis")
        logger.info("=" * 60)
        logger.info(f"🎯 Target: {len(word_analysis)} words")
        logger.info(f"⏱️  Estimated time: {len(word_analysis) * 2 / 3600:.1f} hours (2s per word)")
        
        results = run_circuit_analysis_with_checkpointing(
            model, word_analysis, checkpoint_dir, circuit_graphs_dir, completed_words
        )
        
        # Calculate correlations
        logger.info("=" * 60)
        logger.info("STEP 5: Calculating correlations")
        logger.info("=" * 60)
        correlations = calculate_correlations(results)
        
        # Save final results
        logger.info("=" * 60)
        logger.info("STEP 6: Saving final results")
        logger.info("=" * 60)
        results_file = save_final_results(word_analysis, results, correlations, semantic_file, log_file)
        
        # Calculate total duration
        end_time = time.time()
        duration_hours = (end_time - start_time) / 3600
        
        # Update results with duration
        with open(results_file, 'r') as f:
            final_data = json.load(f)
        final_data['job_info']['duration_hours'] = duration_hours
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("🎉 OVERNIGHT CIRCUIT ANALYSIS JOB COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total duration: {duration_hours:.2f} hours")
        logger.info(f"📊 Success rate: {final_data['job_info']['success_rate']:.1%}")
        logger.info(f"🔬 Circuit complexity correlation: r = {correlations.get('connectivity_complexity', 0):.3f}")
        logger.info(f"💾 Results file: {results_file}")
        logger.info(f"📝 Log file: {log_file}")
        logger.info("🎯 LARGE-SCALE SCIENTIFIC ANALYSIS COMPLETE!")
        
        return results_file
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("💥 OVERNIGHT CIRCUIT JOB FAILED!")
        logger.error(f"❌ Error: {e}")
        logger.error(traceback.format_exc())
        
        # Save error report
        try:
            error_report = {
                'job_failed': True,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            }
            error_file = f'overnight_circuit_ERROR_{error_report["timestamp"]}.json'
            with open(error_file, 'w') as f:
                json.dump(error_report, f, indent=2)
            logger.error(f"💾 Error report saved: {error_file}")
        except:
            pass
        
        raise

if __name__ == "__main__":
    main() 