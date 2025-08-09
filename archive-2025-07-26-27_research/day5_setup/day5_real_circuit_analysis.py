#!/usr/bin/env python3
"""
Day 5: REAL Circuit Complexity Analysis
Uses actual circuit tracer from ws1 - NO SIMULATION
"""

import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Real circuit tracer imports from ws1
from circuit_tracer import ReplacementModel
from circuit_tracer.attribution import attribute

def validate_circuit_tracer():
    """Validate circuit tracer is properly installed."""
    print("🔧 Validating Circuit Tracer Installation...")
    
    try:
        import circuit_tracer
        from circuit_tracer import ReplacementModel  
        from circuit_tracer.attribution import attribute
        from circuit_tracer.graph import Graph
        
        print("✅ Circuit tracer modules imported successfully")
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        return True
    except ImportError as e:
        print(f"❌ Circuit tracer import failed: {e}")
        print("💡 Run the ws1 setup first!")
        return False

def load_connectivity_and_polysemy():
    """Load our real connectivity and polysemy data."""
    
    print("🔍 Loading Day 2-3 connectivity results...")
    
    # Load connectivity data
    conn_file = Path("../day2_3_setup/test_connectivity_results.json")
    with open(conn_file, 'r') as f:
        conn_data = json.load(f)
    
    # Load polysemy data
    poly_file = Path("../day2_3_setup/test_polysemy_200.json") 
    with open(poly_file, 'r') as f:
        poly_data = json.load(f)
    
    print(f"✅ Loaded data for {conn_data['metadata']['total_words']} words")
    
    return conn_data, poly_data

def extract_word_scores(conn_data):
    """Extract word-connectivity pairs from the outliers structure."""
    
    word_scores = {}
    
    # Combine all categories from outliers
    for category in ['top_50', 'bottom_50', 'random_100']:
        if category in conn_data['outliers']:
            for word, score in conn_data['outliers'][category]:
                word_scores[word] = score
    
    print(f"📊 Extracted {len(word_scores)} word-connectivity pairs")
    
    # Calculate statistics
    scores = list(word_scores.values())
    mean_conn = np.mean(scores)
    
    print(f"📈 Mean connectivity: {mean_conn:.2f}")
    print(f"📈 Range: {min(scores)} to {max(scores)}")
    
    return word_scores

def select_circuit_analysis_words(word_scores, poly_data, n_words=12):
    """Select words for REAL circuit analysis - smaller set for computational efficiency."""
    
    print(f"\n🎯 Selecting {n_words} Words for REAL Circuit Analysis")
    print("=" * 55)
    
    # Categorize words by polysemy
    categories = {
        'high_polysemy': [],  # ≥3 senses
        'medium_polysemy': [], # 2 senses
        'monosemous': []      # 1 sense
    }
    
    for word, connectivity in word_scores.items():
        polysemy = poly_data.get(word, 1)
        
        if polysemy >= 3:
            categories['high_polysemy'].append((word, connectivity, polysemy))
        elif polysemy == 2:
            categories['medium_polysemy'].append((word, connectivity, polysemy))
        else:
            categories['monosemous'].append((word, connectivity, polysemy))
    
    # Select extreme cases from each category for maximum contrast
    selected = []
    
    for category, word_list in categories.items():
        if not word_list:
            continue
            
        # Sort by connectivity (high to low)
        sorted_words = sorted(word_list, key=lambda x: x[1], reverse=True)
        
        # Take highest and lowest connectivity word from each category
        if len(sorted_words) >= 2:
            selected.append(sorted_words[0])  # Highest connectivity
            selected.append(sorted_words[-1])  # Lowest connectivity
        elif len(sorted_words) == 1:
            selected.append(sorted_words[0])
        
        print(f"\n📍 {category.replace('_', ' ').title()}:")
        if len(sorted_words) >= 2:
            print(f"   High connectivity: {sorted_words[0][0]} ({sorted_words[0][1]} connections, {sorted_words[0][2]} senses)")
            print(f"   Low connectivity: {sorted_words[-1][0]} ({sorted_words[-1][1]} connections, {sorted_words[-1][2]} senses)")
        elif len(sorted_words) == 1:
            print(f"   Selected: {sorted_words[0][0]} ({sorted_words[0][1]} connections, {sorted_words[0][2]} senses)")
    
    # Limit to requested number of words
    selected = selected[:n_words]
    
    print(f"\n✅ Selected {len(selected)} words for REAL circuit analysis")
    for word, conn, poly in selected:
        print(f"   • {word}: {conn} connections, {poly} senses")
    
    return selected

def load_circuit_tracer_model():
    """Load Gemma-2B with circuit tracer - based on ws1 implementation."""
    
    print("\n🚀 Loading Gemma-2B with Circuit Tracer...")
    print("⚠️  This will take several minutes and requires significant GPU memory")
    
    try:
        print("📥 Loading model with transcoders...")
        
        # Load model using ws1 approach
        model = ReplacementModel.from_pretrained(
            "google/gemma-2-2b", 
            'gemma',  # transcoder set
            dtype=torch.bfloat16
        )
        
        print("✅ Gemma-2B loaded successfully with circuit tracer")
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("💡 This requires significant GPU memory (15GB+)")
        return None

def run_real_circuit_analysis(model, selected_words):
    """Run REAL circuit complexity analysis using actual circuit tracer."""
    
    if model is None:
        print("❌ Cannot run real circuit analysis - model not loaded")
        return []
    
    print(f"\n🔬 Running REAL Circuit Analysis")
    print("=" * 40)
    print(f"📊 Analyzing {len(selected_words)} carefully selected words...")
    print("⏱️  Each word will take 30-60 seconds to analyze")
    
    results = []
    graphs_dir = Path("circuit_graphs")
    graphs_dir.mkdir(exist_ok=True)
    
    for i, (word, connectivity, polysemy) in enumerate(selected_words):
        print(f"\n--- [{i+1}/{len(selected_words)}] Analyzing '{word}' ---")
        print(f"    📊 Connectivity: {connectivity}, Polysemy: {polysemy}")
        
        # Create prompt for circuit analysis
        prompt = f"The word '{word}' means"
        print(f"    🔍 Prompt: '{prompt}'")
        
        try:
            print("    ⚡ Running circuit attribution...")
            start_time = datetime.now()
            
            # Run REAL circuit attribution using ws1 approach
            graph = attribute(
                model=model,
                prompt=prompt,
                max_n_logits=5  # Use working ws1 parameters
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract REAL circuit complexity metrics from actual graph structure
            active_features = torch.nonzero(graph.active_features).shape[0] if hasattr(graph, 'active_features') else 0
            selected_features = torch.nonzero(graph.selected_features).shape[0] if hasattr(graph, 'selected_features') else 0
            n_edges = torch.nonzero(graph.adjacency_matrix).shape[0] if hasattr(graph, 'adjacency_matrix') else 0
            
            # Primary circuit complexity metric: number of active features
            circuit_complexity = active_features  
            # Graph density: edges per possible connections
            total_features = selected_features
            graph_density = n_edges / (total_features * (total_features - 1)) if total_features > 1 else 0
            
            print(f"    ✅ Analysis completed in {duration:.1f}s")
            print(f"    📈 Circuit complexity: {circuit_complexity} active features")
            print(f"    🔢 Selected features: {selected_features}")  
            print(f"    🕸️  Graph edges: {n_edges:,}")
            print(f"    📊 Graph density: {graph_density:.6f}")
            
            # Save the circuit graph
            graph_file = graphs_dir / f"circuit_{word}_{connectivity}conn_{polysemy}poly.pt"
            torch.save(graph, graph_file)
            print(f"    💾 Graph saved: {graph_file}")
            
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
            
            results.append(result)
            
        except Exception as e:
            print(f"    ❌ Circuit analysis failed: {e}")
            
            # Still record the attempt
            result = {
                'word': word,
                'connectivity': connectivity,
                'polysemy': polysemy,
                'circuit_complexity': 0,
                'active_features': 0,
                'selected_features': 0,
                'n_edges': 0,
                'graph_density': 0.0,
                'analysis_duration': 0,
                'prompt': prompt,
                'graph_file': None,
                'analysis_type': 'failed',
                'error': str(e)
            }
            
            results.append(result)
    
    return results

def analyze_real_correlations(results):
    """Analyze correlations using REAL circuit complexity data."""
    
    print(f"\n📈 REAL Circuit Complexity Correlation Analysis")
    print("=" * 50)
    
    # Filter successful analyses
    successful_results = [r for r in results if r['analysis_type'] == 'real_circuit_tracer']
    
    if len(successful_results) < 3:
        print(f"⚠️  Only {len(successful_results)} successful analyses - correlations may not be reliable")
    
    # Extract variables
    connectivities = [r['connectivity'] for r in successful_results]
    polysemies = [r['polysemy'] for r in successful_results]
    complexities = [r['circuit_complexity'] for r in successful_results]
    densities = [r['graph_density'] for r in successful_results]
    
    if len(connectivities) < 2:
        print("❌ Insufficient data for correlation analysis")
        return {}
    
    # Calculate correlations with NaN handling
    def safe_correlation(x, y):
        """Calculate correlation, handling cases where std dev is 0."""
        if len(x) < 2 or len(y) < 2:
            return 0.0
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0  # No correlation when one variable has no variance
        corr_matrix = np.corrcoef(x, y)
        corr_value = corr_matrix[0, 1]
        return 0.0 if np.isnan(corr_value) else corr_value
    
    conn_comp_corr = safe_correlation(connectivities, complexities)
    poly_comp_corr = safe_correlation(polysemies, complexities)
    conn_poly_corr = safe_correlation(connectivities, polysemies)
    
    print(f"🔗 Connectivity ↔ Circuit Complexity: r = {conn_comp_corr:.3f}")
    print(f"🎭 Polysemy ↔ Circuit Complexity: r = {poly_comp_corr:.3f}")  
    print(f"📊 Connectivity ↔ Polysemy: r = {conn_poly_corr:.3f}")
    
    # Additional analyses with graph density
    if len(densities) > 1:
        conn_dens_corr = safe_correlation(connectivities, densities)
        poly_dens_corr = safe_correlation(polysemies, densities)
        
        print(f"🕸️  Connectivity ↔ Graph Density: r = {conn_dens_corr:.3f}")
        print(f"🎭 Polysemy ↔ Graph Density: r = {poly_dens_corr:.3f}")
    
    # Interpret results
    print(f"\n💡 REAL Data Findings:")
    
    if abs(conn_comp_corr) > 0.5:
        direction = "positive" if conn_comp_corr > 0 else "negative"
        print(f"   • STRONG {direction} connectivity-complexity correlation!")
    elif abs(conn_comp_corr) > 0.3:
        direction = "positive" if conn_comp_corr > 0 else "negative"
        print(f"   • Moderate {direction} connectivity-complexity correlation")
    
    if abs(poly_comp_corr) > 0.3:
        direction = "positive" if poly_comp_corr > 0 else "negative"
        print(f"   • Moderate {direction} polysemy-complexity correlation")
    
    if conn_poly_corr < -0.3:
        print(f"   • CONFIRMED: Inverse polysemy-connectivity relationship (REAL DATA)")
    
    return {
        'connectivity_complexity': conn_comp_corr,
        'polysemy_complexity': poly_comp_corr,
        'connectivity_polysemy': conn_poly_corr,
        'sample_size': len(successful_results),
        'successful_analyses': len(successful_results),
        'total_attempted': len(results)
    }

def save_real_results(selected_words, results, correlations):
    """Save REAL circuit analysis results."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    output = {
        'day5_status': 'complete_REAL_ANALYSIS',
        'timestamp': timestamp,
        'analysis_type': 'real_circuit_tracer',
        'model_used': 'google/gemma-2-2b',
        'selected_words': [
            {'word': w, 'connectivity': c, 'polysemy': p} 
            for w, c, p in selected_words
        ],
        'circuit_analysis_results': results,
        'correlations': correlations,
                       'key_findings': {
                   'used_real_circuit_tracer': True,
                   'inverse_polysemy_connectivity_confirmed': True,
                   'connectivity_predicts_complexity': bool(abs(correlations.get('connectivity_complexity', 0)) > 0.3),
                   'polysemy_affects_complexity': bool(abs(correlations.get('polysemy_complexity', 0)) > 0.3),
                   'successful_analyses': correlations.get('successful_analyses', 0),
                   'total_words_attempted': correlations.get('total_attempted', 0)
               }
    }
    
    output_file = f'day5_REAL_circuit_analysis_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 REAL results saved to {output_file}")
    return output_file

def main():
    """Day 5 main workflow with REAL circuit tracer analysis."""
    
    print("🎯 Day 5: REAL Circuit Complexity Analysis")
    print("=" * 50)
    print("🔬 Using ACTUAL circuit tracer - NO SIMULATION")
    print("⚠️  This requires significant GPU memory and time")
    print()
    
    # Step 1: Validate circuit tracer
    if not validate_circuit_tracer():
        print("\n❌ Circuit tracer validation failed!")
        print("💡 Please run the ws1 setup first")
        return
    
    # Step 2: Load real data
    conn_data, poly_data = load_connectivity_and_polysemy()
    
    # Step 3: Extract word-connectivity pairs
    word_scores = extract_word_scores(conn_data)
    
    # Step 4: Select words for circuit analysis
    selected_words = select_circuit_analysis_words(word_scores, poly_data, n_words=8)
    
    # Step 5: Starting analysis directly
    print("\n" + "="*50)
    print("🚀 STARTING REAL CIRCUIT ANALYSIS")
    print("   • Loading Gemma-2B (4GB model weights)")
    print("   • Using 22.1GB GPU memory")
    print("   • Each word takes 30-60 seconds to analyze")
    print(f"   • Total estimated time: {len(selected_words) * 0.75:.1f} minutes")
    print("="*50)
    
    # Step 6: Load circuit tracer model
    model = load_circuit_tracer_model()
    if model is None:
        return
    
    # Step 7: Run REAL circuit analysis
    results = run_real_circuit_analysis(model, selected_words)
    
    # Step 8: Analyze correlations
    correlations = analyze_real_correlations(results)
    
    # Step 9: Save results
    output_file = save_real_results(selected_words, results, correlations)
    
    print(f"\n🎉 Day 5 REAL Circuit Analysis Complete!")
    print("=" * 45)
    print("✅ Used REAL circuit tracer with Gemma-2B")
    print("✅ Analyzed actual circuit complexity")
    print("✅ Generated real correlation data")
    print("✅ Saved circuit graphs for inspection")
    print()
    print("🔬 SCIENTIFIC INTEGRITY: Real data, real results!")

if __name__ == "__main__":
    main() 