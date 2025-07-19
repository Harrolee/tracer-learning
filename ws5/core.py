"""
WS5: Core Analysis Functions for Circuit Checkpoint Comparison

This module provides the main analysis functions for comparing circuit outputs
across training checkpoints to identify learning patterns.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class CheckpointInfo:
    """Information about a training checkpoint."""
    name: str
    step: int
    progress: float
    epoch: float
    loss: float
    learning_rate: float
    timestamp: str
    path: Path

@dataclass
class CircuitAnalysis:
    """Results of circuit analysis for a single checkpoint."""
    checkpoint: CheckpointInfo
    attribution_graph: Optional[Any] = None  # Circuit tracer graph
    feature_activations: Optional[Dict[str, torch.Tensor]] = None
    attribution_scores: Optional[torch.Tensor] = None
    graph_metrics: Optional[Dict[str, float]] = None

@dataclass
class ComparisonResult:
    """Results of comparing two circuit analyses."""
    checkpoint_1: str
    checkpoint_2: str
    similarity_score: float
    difference_magnitude: float
    changed_features: List[str]
    stable_features: List[str]
    emerging_features: List[str]
    diminishing_features: List[str]

def load_checkpoint_circuits(checkpoint_dir: Union[str, Path]) -> Dict[str, CheckpointInfo]:
    """
    Load circuit outputs from saved checkpoints.
    
    Args:
        checkpoint_dir: Directory containing circuit checkpoints
        
    Returns:
        Dictionary mapping checkpoint names to CheckpointInfo objects
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = {}
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Find all checkpoint directories
    for checkpoint_path in checkpoint_dir.glob("checkpoint-*pct"):
        if not checkpoint_path.is_dir():
            continue
            
        try:
            # Load training state
            training_state_path = checkpoint_path / "training_state.json"
            if not training_state_path.exists():
                logger.warning(f"No training_state.json found in {checkpoint_path}")
                continue
                
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            
            # Create CheckpointInfo
            checkpoint_info = CheckpointInfo(
                name=checkpoint_path.name,
                step=training_state['global_step'],
                progress=training_state['progress'],
                epoch=training_state['epoch'],
                loss=training_state['loss'],
                learning_rate=training_state['learning_rate'],
                timestamp=training_state['timestamp'],
                path=checkpoint_path
            )
            
            checkpoints[checkpoint_path.name] = checkpoint_info
            logger.info(f"Loaded checkpoint: {checkpoint_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            continue
    
    return checkpoints

def load_model_for_analysis(checkpoint_info: CheckpointInfo, base_model_path: Union[str, Path]) -> Any:
    """
    Load a model with LoRA adapter for circuit analysis.
    
    Args:
        checkpoint_info: Information about the checkpoint to load
        base_model_path: Path to the base model
        
    Returns:
        Loaded model ready for circuit analysis
    """
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_info.path)
        
        logger.info(f"Loaded model for checkpoint {checkpoint_info.name}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model for {checkpoint_info.name}: {e}")
        raise

def generate_circuit_analysis(
    model: Any, 
    test_prompts: List[str],
    checkpoint_info: CheckpointInfo,
    use_circuit_tracer: bool = True
) -> CircuitAnalysis:
    """
    Generate circuit analysis for a loaded model.
    
    Args:
        model: The loaded model with adapter
        test_prompts: List of prompts to analyze
        checkpoint_info: Information about the checkpoint
        use_circuit_tracer: Whether to use circuit tracer (requires GPU)
        
    Returns:
        CircuitAnalysis object with results
    """
    analysis = CircuitAnalysis(checkpoint=checkpoint_info)
    
    try:
        if use_circuit_tracer:
            # Use circuit tracer for detailed analysis (requires GPU)
            analysis = _run_circuit_tracer_analysis(model, test_prompts, analysis)
        else:
            # Use simpler activation analysis (CPU-friendly)
            analysis = _run_activation_analysis(model, test_prompts, analysis)
            
        logger.info(f"Generated circuit analysis for {checkpoint_info.name}")
        
    except Exception as e:
        logger.error(f"Failed to generate circuit analysis for {checkpoint_info.name}: {e}")
        
    return analysis

def _run_circuit_tracer_analysis(model: Any, test_prompts: List[str], analysis: CircuitAnalysis) -> CircuitAnalysis:
    """Run detailed circuit tracer analysis (requires GPU and circuit tracer setup)."""
    try:
        # This would use the circuit tracer from WS1
        # For now, we'll prepare the structure but not require GPU
        from circuit_tracer import ReplacementModel
        from circuit_tracer.attribution import attribute
        
        # Convert model to ReplacementModel format if needed
        # circuit_model = ReplacementModel.from_pretrained(model)
        
        # For each test prompt, run attribution
        graphs = {}
        for prompt in test_prompts:
            # graph = attribute(model=circuit_model, prompt=prompt, max_n_logits=5)
            # graphs[prompt] = graph
            pass  # Placeholder for now
            
        analysis.attribution_graph = graphs
        
    except ImportError:
        logger.warning("Circuit tracer not available, falling back to activation analysis")
        analysis = _run_activation_analysis(model, test_prompts, analysis)
    except Exception as e:
        logger.error(f"Circuit tracer analysis failed: {e}")
        analysis = _run_activation_analysis(model, test_prompts, analysis)
        
    return analysis

def _run_activation_analysis(model: Any, test_prompts: List[str], analysis: CircuitAnalysis) -> CircuitAnalysis:
    """Run simpler activation-based analysis (CPU-friendly)."""
    try:
        from transformers import AutoTokenizer
        
        # Get tokenizer
        tokenizer = AutoTokenizer.from_pretrained(analysis.checkpoint.path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Analyze model outputs for each prompt
        feature_activations = {}
        attribution_scores = []
        
        model.eval()
        with torch.no_grad():
            for prompt in test_prompts:
                # Tokenize prompt
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                
                # Get model outputs
                outputs = model(**inputs, output_hidden_states=True)
                
                # Extract hidden states (simplified feature analysis)
                hidden_states = outputs.hidden_states
                last_hidden = hidden_states[-1]  # Last layer
                
                # Compute simple metrics
                activation_mean = last_hidden.mean().item()
                activation_std = last_hidden.std().item()
                activation_norm = torch.norm(last_hidden).item()
                
                feature_activations[prompt] = {
                    'mean_activation': activation_mean,
                    'std_activation': activation_std,
                    'norm_activation': activation_norm,
                    'hidden_states_shape': last_hidden.shape
                }
                
                attribution_scores.append(activation_norm)
        
        analysis.feature_activations = feature_activations
        analysis.attribution_scores = torch.tensor(attribution_scores)
        
        # Compute graph metrics
        analysis.graph_metrics = {
            'mean_attribution': float(analysis.attribution_scores.mean()),
            'std_attribution': float(analysis.attribution_scores.std()),
            'total_activation': float(analysis.attribution_scores.sum()),
            'num_prompts': len(test_prompts)
        }
        
        logger.info(f"Completed activation analysis for {analysis.checkpoint.name}")
        
    except Exception as e:
        logger.error(f"Activation analysis failed: {e}")
        
    return analysis

def compare_attribution_graphs(
    analysis_1: CircuitAnalysis, 
    analysis_2: CircuitAnalysis,
    similarity_threshold: float = 0.8
) -> ComparisonResult:
    """
    Compute differences between circuit states.
    
    Args:
        analysis_1: First circuit analysis
        analysis_2: Second circuit analysis
        similarity_threshold: Threshold for determining stable features
        
    Returns:
        ComparisonResult with comparison metrics
    """
    try:
        # For now, use simplified comparison based on available data
        if analysis_1.attribution_scores is not None and analysis_2.attribution_scores is not None:
            # Compare attribution scores
            scores_1 = analysis_1.attribution_scores
            scores_2 = analysis_2.attribution_scores
            
            # Compute similarity
            if len(scores_1) == len(scores_2):
                cosine_sim = torch.cosine_similarity(scores_1.unsqueeze(0), scores_2.unsqueeze(0)).item()
                l2_diff = torch.norm(scores_1 - scores_2).item()
            else:
                cosine_sim = 0.0
                l2_diff = float('inf')
                
            # Determine changed vs stable features (simplified)
            if analysis_1.feature_activations and analysis_2.feature_activations:
                common_prompts = set(analysis_1.feature_activations.keys()) & set(analysis_2.feature_activations.keys())
                
                changed_features = []
                stable_features = []
                
                for prompt in common_prompts:
                    feat_1 = analysis_1.feature_activations[prompt]
                    feat_2 = analysis_2.feature_activations[prompt]
                    
                    # Compare mean activations
                    diff = abs(feat_1['mean_activation'] - feat_2['mean_activation'])
                    if diff > 0.1:  # Threshold for "changed"
                        changed_features.append(prompt)
                    else:
                        stable_features.append(prompt)
            else:
                changed_features = []
                stable_features = []
                
        else:
            cosine_sim = 0.0
            l2_diff = float('inf')
            changed_features = []
            stable_features = []
        
        return ComparisonResult(
            checkpoint_1=analysis_1.checkpoint.name,
            checkpoint_2=analysis_2.checkpoint.name,
            similarity_score=cosine_sim,
            difference_magnitude=l2_diff,
            changed_features=changed_features,
            stable_features=stable_features,
            emerging_features=[],  # TODO: Implement detection
            diminishing_features=[]  # TODO: Implement detection
        )
        
    except Exception as e:
        logger.error(f"Failed to compare analyses: {e}")
        return ComparisonResult(
            checkpoint_1=analysis_1.checkpoint.name,
            checkpoint_2=analysis_2.checkpoint.name,
            similarity_score=0.0,
            difference_magnitude=float('inf'),
            changed_features=[],
            stable_features=[],
            emerging_features=[],
            diminishing_features=[]
        )

def detect_saturation(analyses: List[CircuitAnalysis], window_size: int = 3) -> Dict[str, Any]:
    """
    Identify when circuits stop changing significantly.
    
    Args:
        analyses: List of circuit analyses in chronological order
        window_size: Size of sliding window for saturation detection
        
    Returns:
        Dictionary with saturation detection results
    """
    if len(analyses) < window_size:
        return {
            'saturated': False,
            'saturation_point': None,
            'saturation_step': None,
            'reason': f'Insufficient analyses (need at least {window_size})'
        }
    
    try:
        # Sort analyses by step
        sorted_analyses = sorted(analyses, key=lambda x: x.checkpoint.step)
        
        # Compute pairwise similarities in sliding windows
        similarities = []
        for i in range(len(sorted_analyses) - 1):
            comparison = compare_attribution_graphs(sorted_analyses[i], sorted_analyses[i + 1])
            similarities.append(comparison.similarity_score)
        
        # Detect saturation using sliding window
        saturation_threshold = 0.95  # High similarity indicates saturation
        
        for i in range(len(similarities) - window_size + 1):
            window_similarities = similarities[i:i + window_size]
            if all(sim > saturation_threshold for sim in window_similarities):
                return {
                    'saturated': True,
                    'saturation_point': i + window_size - 1,
                    'saturation_step': sorted_analyses[i + window_size - 1].checkpoint.step,
                    'saturation_similarity': np.mean(window_similarities),
                    'reason': f'High similarity (>{saturation_threshold}) in window of {window_size}'
                }
        
        return {
            'saturated': False,
            'saturation_point': None,
            'saturation_step': None,
            'max_similarity': max(similarities) if similarities else 0.0,
            'reason': f'No sustained high similarity (>{saturation_threshold}) found'
        }
        
    except Exception as e:
        logger.error(f"Saturation detection failed: {e}")
        return {
            'saturated': False,
            'saturation_point': None,
            'saturation_step': None,
            'error': str(e)
        }

def extract_learning_patterns(
    analyses: List[CircuitAnalysis], 
    constraint_examples: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Identify which constraint types are being learned.
    
    Args:
        analyses: List of circuit analyses in chronological order
        constraint_examples: Dict mapping constraint types to example prompts
        
    Returns:
        Dictionary with learning pattern analysis
    """
    try:
        sorted_analyses = sorted(analyses, key=lambda x: x.checkpoint.step)
        
        learning_patterns = {
            'constraint_learning_curves': {},
            'learning_order': [],
            'convergence_steps': {},
            'performance_trends': {}
        }
        
        # Analyze learning for each constraint type
        for constraint_type, examples in constraint_examples.items():
            type_performances = []
            
            for analysis in sorted_analyses:
                if analysis.feature_activations:
                    # Calculate average performance for this constraint type
                    type_activations = []
                    for example in examples:
                        if example in analysis.feature_activations:
                            activation = analysis.feature_activations[example]['norm_activation']
                            type_activations.append(activation)
                    
                    if type_activations:
                        avg_performance = np.mean(type_activations)
                        type_performances.append({
                            'step': analysis.checkpoint.step,
                            'performance': avg_performance,
                            'progress': analysis.checkpoint.progress
                        })
            
            learning_patterns['constraint_learning_curves'][constraint_type] = type_performances
            
            # Detect convergence (simplified)
            if len(type_performances) >= 2:
                final_perf = type_performances[-1]['performance']
                initial_perf = type_performances[0]['performance']
                improvement = final_perf - initial_perf
                
                learning_patterns['performance_trends'][constraint_type] = {
                    'initial_performance': initial_perf,
                    'final_performance': final_perf,
                    'total_improvement': improvement,
                    'learning_rate': improvement / len(type_performances)
                }
        
        # Determine learning order (which constraint types improve first)
        if learning_patterns['performance_trends']:
            constraint_improvements = [
                (constraint, metrics['total_improvement'])
                for constraint, metrics in learning_patterns['performance_trends'].items()
            ]
            learning_patterns['learning_order'] = sorted(
                constraint_improvements, 
                key=lambda x: x[1], 
                reverse=True
            )
        
        logger.info("Extracted learning patterns successfully")
        return learning_patterns
        
    except Exception as e:
        logger.error(f"Failed to extract learning patterns: {e}")
        return {
            'constraint_learning_curves': {},
            'learning_order': [],
            'convergence_steps': {},
            'performance_trends': {},
            'error': str(e)
        }

def save_analysis_results(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save analysis results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert non-serializable objects to serializable format
    serializable_results = _make_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    logger.info(f"Saved analysis results to {output_path}")

def _make_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (CheckpointInfo, CircuitAnalysis, ComparisonResult)):
        return obj.__dict__
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    else:
        return obj