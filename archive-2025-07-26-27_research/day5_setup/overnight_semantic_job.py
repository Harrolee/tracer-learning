#!/usr/bin/env python3
"""
Overnight Job 1: Semantic Connectivity Analysis for 1000 Words
Designed to run unattended overnight with comprehensive logging
"""

import sys
import json
import time
import numpy as np
import pickle
import torch
from pathlib import Path
from datetime import datetime
import logging
import traceback

# Add parent directories to path
sys.path.append('../day1_setup')
sys.path.append('../day2_3_setup')

def setup_logging():
    """Setup comprehensive logging for overnight job."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f'overnight_semantic_job_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🌙 Starting Overnight Semantic Analysis Job")
    logger.info(f"📝 Logging to: {log_file}")
    return logger, log_file

def sample_1000_words():
    """Sample 1000 words using polysemy-based extreme contrast strategy."""
    logger = logging.getLogger(__name__)
    logger.info("📊 Sampling 1000 words using polysemy-based extreme contrast...")
    
    try:
        from vocab_sampling import sample_by_polysemy, calculate_polysemy_scores
        
        # First calculate polysemy scores for all words, then sample
        logger.info("📊 Calculating polysemy scores for vocabulary...")
        all_polysemy_scores = calculate_polysemy_scores()  # Get all WordNet words
        
        # Sample 1000 words with extreme contrast strategy
        words = sample_by_polysemy(all_polysemy_scores, strategy='extreme_contrast', total_words=1000)
        
        logger.info(f"✅ Successfully sampled {len(words)} words")
        logger.info(f"📈 Sample preview: {words[:10]}")
        
        # Save words list for reference
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        words_file = f'sampled_1000_words_{timestamp}.json'
        with open(words_file, 'w') as f:
            json.dump(words, f, indent=2)
        
        logger.info(f"💾 Words saved to: {words_file}")
        return words, words_file
        
    except Exception as e:
        logger.error(f"❌ Word sampling failed: {e}")
        logger.error(traceback.format_exc())
        raise

def run_semantic_connectivity_analysis(words):
    """Run semantic connectivity analysis on 1000 words."""
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 Starting semantic connectivity analysis for {len(words)} words...")
    
    try:
        # Import semantic connectivity tools
        sys.path.append('../day2_3_setup')
        from semantic_connectivity_cli import get_word_embedding, semantic_connectivity
        from transformers import AutoModel, AutoTokenizer
        
        logger.info("📥 Loading Gemma2-2B model and tokenizer...")
        model = AutoModel.from_pretrained("google/gemma-2-2b", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✅ Model loaded on GPU")
        else:
            logger.info("✅ Model loaded on CPU")
        
        model.eval()  # Set to evaluation mode
        
        # Process words in batches for efficiency
        batch_size = 50
        connectivity_results = {}
        failed_words = []
        
        total_batches = (len(words) + batch_size - 1) // batch_size
        logger.info(f"📊 Processing {len(words)} words in {total_batches} batches of {batch_size}")
        
        for batch_idx in range(0, len(words), batch_size):
            batch_words = words[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            logger.info(f"⚡ Processing batch {batch_num}/{total_batches}: {len(batch_words)} words")
            start_time = time.time()
            
            try:
                # Get embeddings for batch
                embeddings = []
                for word in batch_words:
                    try:
                        embedding = get_word_embedding(model, tokenizer, word)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to get embedding for '{word}': {e}")
                        embeddings.append(None)
                
                # Calculate connectivity for each word in batch
                for i, word in enumerate(batch_words):
                    if embeddings[i] is None:
                        failed_words.append(word)
                        connectivity_results[word] = 0.0
                        continue
                        
                    try:
                        # Calculate connectivity against all other embeddings in batch
                        valid_embeddings = [emb for emb in embeddings if emb is not None]
                        if len(valid_embeddings) < 2:
                            connectivity_results[word] = 0.0
                        else:
                            connectivity = semantic_connectivity(embeddings[i], valid_embeddings, threshold=0.6)
                            connectivity_results[word] = float(connectivity)
                        
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to analyze '{word}': {e}")
                        failed_words.append(word)
                        connectivity_results[word] = 0.0
                
                batch_time = time.time() - start_time
                logger.info(f"✅ Batch {batch_num} completed in {batch_time:.1f}s")
                
                # Save intermediate results every 5 batches
                if batch_num % 5 == 0:
                    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    interim_file = f'interim_connectivity_results_batch_{batch_num}_{timestamp}.json'
                    with open(interim_file, 'w') as f:
                        json.dump({
                            'batch_progress': f'{batch_num}/{total_batches}',
                            'completed_words': len(connectivity_results),
                            'connectivity_results': connectivity_results,
                            'failed_words': failed_words
                        }, f, indent=2)
                    logger.info(f"💾 Interim results saved: {interim_file}")
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                # Mark all words in batch as failed
                for word in batch_words:
                    failed_words.append(word)
                    connectivity_results[word] = 0.0
                continue
        
        logger.info(f"🎉 Semantic connectivity analysis complete!")
        logger.info(f"✅ Successfully analyzed {len(connectivity_results) - len(failed_words)} words")
        logger.info(f"⚠️  Failed words: {len(failed_words)}")
        
        return connectivity_results, failed_words
        
    except Exception as e:
        logger.error(f"❌ Semantic connectivity analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise

def calculate_polysemy_scores_for_words(words):
    """Calculate polysemy scores for specific words."""
    logger = logging.getLogger(__name__)
    logger.info(f"🎭 Calculating polysemy scores for {len(words)} words...")
    
    try:
        from vocab_sampling import calculate_polysemy_scores
        
        # Get all polysemy scores, then filter for our specific words
        all_polysemy_scores = calculate_polysemy_scores()
        polysemy_scores = {word: all_polysemy_scores.get(word, 1) for word in words}
        
        logger.info(f"✅ Polysemy scores calculated for {len(polysemy_scores)} words")
        
        return polysemy_scores
        
    except Exception as e:
        logger.error(f"❌ Polysemy calculation failed: {e}")
        logger.error(traceback.format_exc())
        raise

def save_final_results(words, connectivity_results, polysemy_scores, failed_words, words_file, log_file):
    """Save comprehensive final results."""
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Calculate statistics
    successful_words = [w for w in words if w not in failed_words]
    connectivities = [connectivity_results[w] for w in successful_words]
    polysemies = [polysemy_scores.get(w, 1) for w in successful_words]
    
    mean_connectivity = np.mean(connectivities) if connectivities else 0
    mean_polysemy = np.mean(polysemies) if polysemies else 0
    connectivity_range = (min(connectivities), max(connectivities)) if connectivities else (0, 0)
    
    # Create comprehensive results
    final_results = {
        'job_info': {
            'job_type': 'overnight_semantic_analysis',
            'timestamp': timestamp,
            'duration_hours': 'TBD',  # Will be calculated at end
            'total_words_requested': len(words),
            'successful_analyses': len(successful_words),
            'failed_analyses': len(failed_words),
            'success_rate': len(successful_words) / len(words) if words else 0
        },
        'files': {
            'words_file': words_file,
            'log_file': log_file
        },
        'statistics': {
            'mean_connectivity': mean_connectivity,
            'connectivity_range': connectivity_range,
            'mean_polysemy': mean_polysemy,
            'polysemy_range': (min(polysemies), max(polysemies)) if polysemies else (0, 0)
        },
        'data': {
            'words': words,
            'connectivity_results': connectivity_results,
            'polysemy_scores': polysemy_scores,
            'failed_words': failed_words
        },
        'word_analysis': []
    }
    
    # Add detailed per-word analysis
    for word in successful_words:
        final_results['word_analysis'].append({
            'word': word,
            'connectivity': connectivity_results[word],
            'polysemy': polysemy_scores.get(word, 1)
        })
    
    # Sort by connectivity for easier analysis
    final_results['word_analysis'].sort(key=lambda x: x['connectivity'], reverse=True)
    
    # Save final results
    results_file = f'overnight_semantic_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"💾 Final results saved to: {results_file}")
    logger.info(f"📊 Analysis Summary:")
    logger.info(f"   • Total words: {len(words)}")
    logger.info(f"   • Successful: {len(successful_words)}")
    logger.info(f"   • Failed: {len(failed_words)}")
    logger.info(f"   • Success rate: {final_results['job_info']['success_rate']:.1%}")
    logger.info(f"   • Mean connectivity: {mean_connectivity:.2f}")
    logger.info(f"   • Connectivity range: {connectivity_range}")
    logger.info(f"   • Mean polysemy: {mean_polysemy:.2f}")
    
    return results_file, final_results

def main():
    """Main overnight semantic analysis job."""
    start_time = time.time()
    
    try:
        # Setup logging
        logger, log_file = setup_logging()
        logger.info("🚀 Overnight Semantic Analysis Job Starting")
        
        # Step 1: Sample 1000 words
        logger.info("=" * 60)
        logger.info("STEP 1: Sampling 1000 words")
        logger.info("=" * 60)
        words, words_file = sample_1000_words()
        
        # Step 2: Calculate polysemy scores
        logger.info("=" * 60)
        logger.info("STEP 2: Calculating polysemy scores")
        logger.info("=" * 60)
        polysemy_scores = calculate_polysemy_scores_for_words(words)
        
        # Step 3: Run semantic connectivity analysis
        logger.info("=" * 60)
        logger.info("STEP 3: Running semantic connectivity analysis")
        logger.info("=" * 60)
        connectivity_results, failed_words = run_semantic_connectivity_analysis(words)
        
        # Step 4: Save final results
        logger.info("=" * 60)
        logger.info("STEP 4: Saving final results")
        logger.info("=" * 60)
        results_file, final_results = save_final_results(
            words, connectivity_results, polysemy_scores, 
            failed_words, words_file, log_file
        )
        
        # Calculate total duration
        end_time = time.time()
        duration_hours = (end_time - start_time) / 3600
        
        # Update final results with duration
        final_results['job_info']['duration_hours'] = duration_hours
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("🎉 OVERNIGHT SEMANTIC ANALYSIS JOB COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total duration: {duration_hours:.2f} hours")
        logger.info(f"📊 Success rate: {final_results['job_info']['success_rate']:.1%}")
        logger.info(f"💾 Results file: {results_file}")
        logger.info(f"📝 Log file: {log_file}")
        logger.info("🌅 Ready for circuit analysis job!")
        
        return results_file
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("💥 OVERNIGHT JOB FAILED!")
        logger.error(f"❌ Error: {e}")
        logger.error(traceback.format_exc())
        
        # Try to save error report
        try:
            error_report = {
                'job_failed': True,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            }
            error_file = f'overnight_semantic_ERROR_{error_report["timestamp"]}.json'
            with open(error_file, 'w') as f:
                json.dump(error_report, f, indent=2)
            logger.error(f"💾 Error report saved: {error_file}")
        except:
            pass
        
        raise

if __name__ == "__main__":
    main() 