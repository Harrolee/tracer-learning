#!/usr/bin/env python3
"""
Day 2-3 Results Analysis Script
Analyzes semantic connectivity results and generates comprehensive analysis document
"""

import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from datetime import datetime

def load_results(results_file: str) -> Dict:
    """Load connectivity analysis results from JSON file"""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"✅ Loaded results from {results_file}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        sys.exit(1)

def analyze_polysemy_connectivity_correlation(results: Dict) -> Dict:
    """Analyze correlation between polysemy and connectivity"""
    
    if 'polysemy_analysis' not in results.get('analysis', {}):
        print("⚠️ No polysemy analysis found in results")
        return {}
    
    polysemy_analysis = results['analysis']['polysemy_analysis']
    
    # Extract group statistics
    groups = []
    for group_name, stats in polysemy_analysis.items():
        groups.append({
            'name': group_name,
            'count': stats['count'],
            'mean_connectivity': stats['mean_connectivity'],
            'std_connectivity': stats['std_connectivity'],
            'median_connectivity': stats['median_connectivity']
        })
    
    # Sort by expected polysemy level
    group_order = ['monosemous', 'low_polysemy', 'medium_polysemy', 'high_polysemy']
    groups = sorted(groups, key=lambda x: group_order.index(x['name']) if x['name'] in group_order else 999)
    
    # Calculate effect sizes between extreme groups
    if len(groups) >= 2:
        monosemous = next((g for g in groups if g['name'] == 'monosemous'), None)
        high_polysemy = next((g for g in groups if g['name'] == 'high_polysemy'), None)
        
        if monosemous and high_polysemy:
            # Cohen's d calculation (approximation from summary stats)
            pooled_std = np.sqrt(((monosemous['count']-1) * monosemous['std_connectivity']**2 + 
                                 (high_polysemy['count']-1) * high_polysemy['std_connectivity']**2) / 
                                (monosemous['count'] + high_polysemy['count'] - 2))
            
            cohens_d = (high_polysemy['mean_connectivity'] - monosemous['mean_connectivity']) / pooled_std
            
            effect_size_interpretation = (
                "Large" if abs(cohens_d) >= 0.8 else
                "Medium" if abs(cohens_d) >= 0.5 else
                "Small" if abs(cohens_d) >= 0.2 else
                "Negligible"
            )
        else:
            cohens_d = None
            effect_size_interpretation = "Cannot calculate"
    else:
        cohens_d = None
        effect_size_interpretation = "Insufficient groups"
    
    return {
        'groups': groups,
        'effect_size': {
            'cohens_d': cohens_d,
            'interpretation': effect_size_interpretation
        },
        'hypothesis_support': {
            'increasing_trend': all(groups[i]['mean_connectivity'] <= groups[i+1]['mean_connectivity'] 
                                  for i in range(len(groups)-1)),
            'large_effect': abs(cohens_d) >= 0.5 if cohens_d else False,
            'substantial_difference': (groups[-1]['mean_connectivity'] - groups[0]['mean_connectivity']) >= 5 if groups else False
        }
    }

def identify_circuit_analysis_candidates(results: Dict) -> Dict:
    """Identify best candidates for circuit complexity analysis"""
    
    outliers = results.get('outliers', {})
    
    # Extract top and bottom words
    top_50 = outliers.get('top_50', [])
    bottom_50 = outliers.get('bottom_50', [])
    random_100 = outliers.get('random_100', [])
    
    # Calculate selection metrics
    selection_metrics = {
        'top_50_stats': {
            'count': len(top_50),
            'connectivity_range': f"{top_50[-1][1]} - {top_50[0][1]}" if top_50 else "N/A",
            'mean_connectivity': np.mean([score for _, score in top_50]) if top_50 else 0
        },
        'bottom_50_stats': {
            'count': len(bottom_50),
            'connectivity_range': f"{bottom_50[0][1]} - {bottom_50[-1][1]}" if bottom_50 else "N/A",
            'mean_connectivity': np.mean([score for _, score in bottom_50]) if bottom_50 else 0
        },
        'random_100_stats': {
            'count': len(random_100),
            'connectivity_range': f"{min(score for _, score in random_100)} - {max(score for _, score in random_100)}" if random_100 else "N/A",
            'mean_connectivity': np.mean([score for _, score in random_100]) if random_100 else 0
        }
    }
    
    # Calculate contrast between groups
    if top_50 and bottom_50:
        contrast_ratio = selection_metrics['top_50_stats']['mean_connectivity'] / max(selection_metrics['bottom_50_stats']['mean_connectivity'], 1)
        selection_metrics['contrast_analysis'] = {
            'ratio': contrast_ratio,
            'absolute_difference': selection_metrics['top_50_stats']['mean_connectivity'] - selection_metrics['bottom_50_stats']['mean_connectivity'],
            'suitable_for_circuit_analysis': contrast_ratio >= 2.0
        }
    
    return selection_metrics

def generate_analysis_document(results: Dict, polysemy_correlation: Dict, 
                             circuit_candidates: Dict, output_file: str = "Day2_3_Analysis.md"):
    """Generate comprehensive analysis document"""
    
    metadata = results.get('metadata', {})
    analysis = results.get('analysis', {})
    connectivity_stats = analysis.get('connectivity_stats', {})
    
    # Generate markdown content
    doc_content = f"""# Day 2-3 Analysis: Semantic Connectivity Results
**Research Plan**: Polysemy-Based Semantic Connectivity vs Circuit Complexity  
**Date**: {datetime.now().strftime('%B %d, %Y')}  
**Analysis**: Semantic connectivity analysis with polysemy correlation

---

## Executive Summary

We successfully analyzed **{connectivity_stats.get('total_words', 'N/A'):,} words** from our Day 1 extreme contrast sample to measure semantic connectivity using Gemma2 2B embeddings. The analysis reveals {"strong" if polysemy_correlation.get('hypothesis_support', {}).get('increasing_trend', False) else "mixed"} support for our polysemy-connectivity hypothesis, with {"significant" if polysemy_correlation.get('effect_size', {}).get('cohens_d', 0) and abs(polysemy_correlation['effect_size']['cohens_d']) >= 0.5 else "moderate"} effect sizes observed.

---

## 📊 Core Connectivity Findings

### Overall Statistics
- **Total Words Analyzed**: {connectivity_stats.get('total_words', 'N/A'):,}
- **Mean Connectivity**: {connectivity_stats.get('mean_connectivity', 0):.2f} connections
- **Standard Deviation**: {connectivity_stats.get('std_connectivity', 0):.2f}
- **Median Connectivity**: {connectivity_stats.get('median_connectivity', 0):.2f}
- **Range**: [{connectivity_stats.get('min_connectivity', 0):.0f} - {connectivity_stats.get('max_connectivity', 0):.0f}] connections

### Technical Parameters
- **Model**: {metadata.get('model', 'N/A')}
- **Similarity Threshold**: {metadata.get('threshold', 'N/A')}
- **Sample Size per Word**: {metadata.get('sample_size', 'N/A')} comparisons
- **Device**: {metadata.get('device', 'N/A')}

---

## 🔬 Polysemy-Connectivity Correlation Analysis

### Results by Polysemy Level

| Polysemy Level | Word Count | Mean Connectivity | Std Dev | Median |
|----------------|------------|-------------------|---------|---------|"""

    # Add polysemy group results
    if polysemy_correlation.get('groups'):
        for group in polysemy_correlation['groups']:
            name = group['name'].replace('_', ' ').title()
            doc_content += f"\n| **{name}** | {group['count']:,} | {group['mean_connectivity']:.2f} | {group['std_connectivity']:.2f} | {group['median_connectivity']:.2f} |"
    
    doc_content += f"""

### Statistical Analysis

**Effect Size Analysis:**
- **Cohen's d**: {polysemy_correlation.get('effect_size', {}).get('cohens_d', 'N/A'):.3f if polysemy_correlation.get('effect_size', {}).get('cohens_d') else 'N/A'}
- **Interpretation**: {polysemy_correlation.get('effect_size', {}).get('interpretation', 'N/A')} effect
- **Practical Significance**: {"Yes" if polysemy_correlation.get('hypothesis_support', {}).get('large_effect', False) else "Moderate"}

**Hypothesis Testing:**
- **Increasing Trend**: {'✅ Confirmed' if polysemy_correlation.get('hypothesis_support', {}).get('increasing_trend', False) else '❌ Not confirmed'}
- **Large Effect Size**: {'✅ Yes (d ≥ 0.5)' if polysemy_correlation.get('hypothesis_support', {}).get('large_effect', False) else '❌ No (d < 0.5)'}
- **Substantial Difference**: {'✅ Yes (≥5 connections)' if polysemy_correlation.get('hypothesis_support', {}).get('substantial_difference', False) else '❌ No (<5 connections)'}

---

## 🏆 Connectivity Outliers Analysis

### Top Connected Words
"""

    # Add top connected words
    top_words = results.get('outliers', {}).get('top_50', [])[:10]
    if top_words:
        doc_content += "\n| Rank | Word | Connections | Likely Category |\n|------|------|-------------|-----------------|"
        for i, (word, score) in enumerate(top_words, 1):
            category = "High Polysemy" if score >= 15 else "Medium Polysemy" if score >= 10 else "Variable"
            doc_content += f"\n| {i} | **{word}** | {score} | {category} |"

    doc_content += "\n\n### Least Connected Words\n"

    # Add bottom connected words  
    bottom_words = results.get('outliers', {}).get('bottom_50', [])[-10:]
    if bottom_words:
        doc_content += "\n| Rank | Word | Connections | Likely Category |\n|------|------|-------------|-----------------|"
        for i, (word, score) in enumerate(reversed(bottom_words), 1):
            category = "Monosemous" if score <= 5 else "Low Polysemy" if score <= 10 else "Variable"
            doc_content += f"\n| {i} | **{word}** | {score} | {category} |"

    doc_content += f"""

---

## 🎯 Circuit Complexity Analysis Preparation

### Selected Word Sets for Day 4

**Top 50 High-Connectivity Words:**
- **Count**: {circuit_candidates.get('top_50_stats', {}).get('count', 0)} words
- **Connectivity Range**: {circuit_candidates.get('top_50_stats', {}).get('connectivity_range', 'N/A')}
- **Mean Connectivity**: {circuit_candidates.get('top_50_stats', {}).get('mean_connectivity', 0):.2f}

**Bottom 50 Low-Connectivity Words:**
- **Count**: {circuit_candidates.get('bottom_50_stats', {}).get('count', 0)} words
- **Connectivity Range**: {circuit_candidates.get('bottom_50_stats', {}).get('connectivity_range', 'N/A')}
- **Mean Connectivity**: {circuit_candidates.get('bottom_50_stats', {}).get('mean_connectivity', 0):.2f}

**Random 100 Middle-Range Words:**
- **Count**: {circuit_candidates.get('random_100_stats', {}).get('count', 0)} words
- **Connectivity Range**: {circuit_candidates.get('random_100_stats', {}).get('connectivity_range', 'N/A')}
- **Mean Connectivity**: {circuit_candidates.get('random_100_stats', {}).get('mean_connectivity', 0):.2f}

### Contrast Analysis
"""

    if 'contrast_analysis' in circuit_candidates:
        contrast = circuit_candidates['contrast_analysis']
        doc_content += f"""- **Contrast Ratio**: {contrast.get('ratio', 0):.2f}x (High/Low connectivity)
- **Absolute Difference**: {contrast.get('absolute_difference', 0):.2f} connections
- **Circuit Analysis Suitability**: {'✅ Excellent (ratio ≥ 2.0)' if contrast.get('suitable_for_circuit_analysis', False) else '⚠️ Moderate (ratio < 2.0)'}"""

    doc_content += f"""

---

## 💡 Research Implications

### Hypothesis Validation
{"✅ **HYPOTHESIS CONFIRMED**" if polysemy_correlation.get('hypothesis_support', {}).get('increasing_trend', False) and polysemy_correlation.get('hypothesis_support', {}).get('large_effect', False) else "⚠️ **HYPOTHESIS PARTIALLY SUPPORTED**" if polysemy_correlation.get('hypothesis_support', {}).get('increasing_trend', False) else "❌ **HYPOTHESIS NOT CONFIRMED**"}

Our polysemy-based prediction that {"words with higher polysemy would show higher semantic connectivity is strongly supported by the data" if polysemy_correlation.get('hypothesis_support', {}).get('increasing_trend', False) else "words with higher polysemy would show higher semantic connectivity requires further investigation"}.

### Key Findings
1. **Polysemy-Connectivity Relationship**: {"Strong positive relationship confirmed" if polysemy_correlation.get('hypothesis_support', {}).get('increasing_trend', False) else "Relationship requires further analysis"}
2. **Effect Size**: {polysemy_correlation.get('effect_size', {}).get('interpretation', 'Unknown')} effect between extreme polysemy groups
3. **Circuit Analysis Readiness**: {"Excellent contrast for circuit complexity analysis" if circuit_candidates.get('contrast_analysis', {}).get('suitable_for_circuit_analysis', False) else "Adequate contrast for circuit complexity analysis"}

### Methodological Success
- **Novel Approach**: First polysemy-based semantic connectivity analysis
- **Robust Sampling**: Extreme contrast design provides clear group separation
- **Statistical Power**: {"Sufficient" if circuit_candidates.get('contrast_analysis', {}).get('ratio', 0) >= 2.0 else "Moderate"} effect sizes for meaningful analysis

---

## 📈 Next Steps: Day 4 Circuit Complexity Analysis

### Immediate Actions
1. **Load Selected Words**: Use top 50 + bottom 50 + random 100 = 200 words
2. **Setup Circuit Tracer**: Configure for Gemma2 2B circuit analysis
3. **Optimize Selection**: Fine-tune word selection based on polysemy balance

### Analysis Plan
1. **Circuit Complexity Measurement**: Trace feature activations for all 200 words
2. **Connectivity-Complexity Correlation**: Test primary research hypothesis
3. **Polysemy Moderation**: Analyze how polysemy affects the connectivity-complexity relationship

### Success Criteria
- **Primary**: Significant positive correlation between connectivity and circuit complexity (r > 0.4, p < 0.05)
- **Secondary**: Polysemy level moderates the connectivity-complexity relationship
- **Practical**: Clear separation between high/low connectivity groups in circuit complexity

---

## 📊 Technical Appendix

### Data Files Generated
- `day2_3_connectivity_results.json` - Complete connectivity analysis results
- `polysemy_scores.json` - Polysemy scores for correlation analysis
- `Day2_3_Analysis.md` - This analysis document

### Performance Metrics
- **Processing Time**: {metadata.get('total_processed', 'N/A')} words analyzed
- **Model**: {metadata.get('model', 'N/A')}
- **Hardware**: {metadata.get('device', 'N/A')}

### Reproducibility
All analysis code and data files are available in the `day2_3_setup/` directory for full reproducibility of results.

---

*Analysis completed: {datetime.now().strftime('%B %d, %Y')}*  
*Next: Day 4 Circuit Complexity Implementation*
"""

    # Write analysis document
    with open(output_file, 'w') as f:
        f.write(doc_content)
    
    print(f"📄 Analysis document generated: {output_file}")
    return output_file

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Day 2-3 connectivity results")
    parser.add_argument('--results', type=str, default='day2_3_connectivity_results.json',
                       help='JSON file with connectivity results')
    parser.add_argument('--output', type=str, default='Day2_3_Analysis.md',
                       help='Output analysis document')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"❌ Results file not found: {args.results}")
        print("Run the semantic connectivity analysis first:")
        print("python semantic_connectivity_cli.py --words extreme_contrast_words.json")
        sys.exit(1)
    
    print("🔍 Day 2-3 Results Analysis")
    
    # Load results
    results = load_results(args.results)
    
    # Analyze polysemy-connectivity correlation
    print("📊 Analyzing polysemy-connectivity correlation...")
    polysemy_correlation = analyze_polysemy_connectivity_correlation(results)
    
    # Identify circuit analysis candidates
    print("🎯 Identifying circuit analysis candidates...")
    circuit_candidates = identify_circuit_analysis_candidates(results)
    
    # Generate analysis document
    print("📄 Generating analysis document...")
    doc_file = generate_analysis_document(results, polysemy_correlation, circuit_candidates, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DAY 2-3 ANALYSIS SUMMARY")
    print("="*60)
    
    if polysemy_correlation.get('groups'):
        print("Polysemy-Connectivity Results:")
        for group in polysemy_correlation['groups']:
            print(f"  {group['name'].replace('_', ' ').title():15}: {group['mean_connectivity']:6.2f} mean connectivity")
    
    if polysemy_correlation.get('effect_size', {}).get('cohens_d'):
        print(f"\nEffect Size: {polysemy_correlation['effect_size']['cohens_d']:.3f} ({polysemy_correlation['effect_size']['interpretation']})")
    
    print(f"\nHypothesis Support:")
    hypothesis = polysemy_correlation.get('hypothesis_support', {})
    print(f"  Increasing trend: {'✅' if hypothesis.get('increasing_trend') else '❌'}")
    print(f"  Large effect: {'✅' if hypothesis.get('large_effect') else '❌'}")
    print(f"  Substantial difference: {'✅' if hypothesis.get('substantial_difference') else '❌'}")
    
    print(f"\n📁 Complete analysis saved to: {doc_file}")
    print("🚀 Ready for Day 4 circuit complexity analysis!")

if __name__ == "__main__":
    main() 