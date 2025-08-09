#!/bin/bash

echo "🌙 Overnight Analysis Jobs for 1000-Word Study"
echo "=============================================="
echo "💰 VM Cost: ~$0.75/hour - Budget for 8-12 hours total"
echo "📊 Job 1: Semantic connectivity analysis (~1-2 hours)"
echo "🔬 Job 2: Circuit complexity analysis (~6-10 hours)"
echo ""

# Function to run semantic job
run_semantic_job() {
    echo "🚀 Starting Job 1: Semantic Connectivity Analysis"
    echo "================================================"
    echo "📝 This will:"
    echo "   • Sample 1000 words using polysemy-based strategy"
    echo "   • Calculate semantic connectivity for each word"
    echo "   • Save results for circuit analysis job"
    echo "   • Estimated time: 1-2 hours"
    echo ""
    
    read -p "🤖 Start semantic analysis job? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "⏱️  Starting at $(date)"
        echo "📝 Logs will be saved to overnight_semantic_job_*.log"
        echo ""
        
        # Run semantic job with nohup so it continues if SSH disconnects
        nohup python overnight_semantic_job.py > semantic_job_console.log 2>&1 &
        SEMANTIC_PID=$!
        
        echo "✅ Semantic job started with PID: $SEMANTIC_PID"
        echo "📊 Monitor progress with: tail -f semantic_job_console.log"
        echo "🔍 Or check the detailed log: tail -f overnight_semantic_job_*.log"
        echo ""
        echo "💡 To check if job is still running: ps -p $SEMANTIC_PID"
        echo "⏹️  To stop job if needed: kill $SEMANTIC_PID"
        
        # Save PID for reference
        echo $SEMANTIC_PID > semantic_job.pid
        echo "💾 PID saved to semantic_job.pid"
        
        return 0
    else
        echo "⏭️  Semantic job cancelled"
        return 1
    fi
}

# Function to run circuit job
run_circuit_job() {
    echo "🔬 Starting Job 2: Circuit Complexity Analysis"
    echo "============================================="
    echo "📝 This will:"
    echo "   • Load results from semantic job"
    echo "   • Run real circuit analysis on 1000 words"
    echo "   • Save circuit graphs and correlations"
    echo "   • Estimated time: 6-10 hours"
    echo "   • Includes checkpointing for recovery"
    echo ""
    
    # Check if semantic results exist
    if ls overnight_semantic_results_*.json 1> /dev/null 2>&1; then
        echo "✅ Semantic results found - ready for circuit analysis"
        LATEST_SEMANTIC=$(ls -t overnight_semantic_results_*.json | head -n1)
        echo "📊 Will use: $LATEST_SEMANTIC"
    else
        echo "❌ No semantic results found!"
        echo "💡 Run semantic job first or check for overnight_semantic_results_*.json files"
        return 1
    fi
    
    echo ""
    read -p "🤖 Start circuit analysis job? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "⏱️  Starting at $(date)"
        echo "📝 Logs will be saved to overnight_circuit_job_*.log"
        echo "💾 Checkpoints will be saved every 10 words"
        echo ""
        
        # Run circuit job with nohup
        nohup python overnight_circuit_job.py > circuit_job_console.log 2>&1 &
        CIRCUIT_PID=$!
        
        echo "✅ Circuit job started with PID: $CIRCUIT_PID"
        echo "📊 Monitor progress with: tail -f circuit_job_console.log"
        echo "🔍 Or check the detailed log: tail -f overnight_circuit_job_*.log"
        echo ""
        echo "💡 To check if job is still running: ps -p $CIRCUIT_PID"
        echo "⏹️  To stop job if needed: kill $CIRCUIT_PID"
        echo "🔄 Job supports recovery - can restart if interrupted"
        
        # Save PID for reference
        echo $CIRCUIT_PID > circuit_job.pid
        echo "💾 PID saved to circuit_job.pid"
        
        return 0
    else
        echo "⏭️  Circuit job cancelled"
        return 1
    fi
}

# Function to show status
show_status() {
    echo "📊 Current Job Status"
    echo "===================="
    
    # Check semantic job
    if [ -f semantic_job.pid ]; then
        SEMANTIC_PID=$(cat semantic_job.pid)
        if ps -p $SEMANTIC_PID > /dev/null; then
            echo "🔄 Semantic job running (PID: $SEMANTIC_PID)"
        else
            echo "✅ Semantic job completed/stopped"
            # Check for results
            if ls overnight_semantic_results_*.json 1> /dev/null 2>&1; then
                LATEST=$(ls -t overnight_semantic_results_*.json | head -n1)
                echo "   📊 Results: $LATEST"
            fi
        fi
    else
        echo "❌ No semantic job found"
    fi
    
    # Check circuit job
    if [ -f circuit_job.pid ]; then
        CIRCUIT_PID=$(cat circuit_job.pid)
        if ps -p $CIRCUIT_PID > /dev/null; then
            echo "🔄 Circuit job running (PID: $CIRCUIT_PID)"
            # Show checkpoint status
            if ls circuit_checkpoints/circuit_checkpoint_*.json 1> /dev/null 2>&1; then
                LATEST_CHECKPOINT=$(ls -t circuit_checkpoints/circuit_checkpoint_*.json | head -n1)
                PROGRESS=$(python -c "import json; data=json.load(open('$LATEST_CHECKPOINT')); print(f'{data[\"completed_count\"]}/{data[\"total_words\"]} words')")
                echo "   📈 Progress: $PROGRESS"
            fi
        else
            echo "✅ Circuit job completed/stopped"
            # Check for results
            if ls overnight_circuit_results_*.json 1> /dev/null 2>&1; then
                LATEST=$(ls -t overnight_circuit_results_*.json | head -n1)
                echo "   📊 Results: $LATEST"
            fi
        fi
    else
        echo "❌ No circuit job found"
    fi
    
    echo ""
    echo "📝 Log files:"
    ls -la *.log 2>/dev/null || echo "   No log files found"
}

# Function to stop jobs
stop_jobs() {
    echo "⏹️  Stopping All Jobs"
    echo "==================="
    
    STOPPED=0
    
    if [ -f semantic_job.pid ]; then
        SEMANTIC_PID=$(cat semantic_job.pid)
        if ps -p $SEMANTIC_PID > /dev/null; then
            echo "🛑 Stopping semantic job (PID: $SEMANTIC_PID)"
            kill $SEMANTIC_PID
            STOPPED=1
        fi
    fi
    
    if [ -f circuit_job.pid ]; then
        CIRCUIT_PID=$(cat circuit_job.pid)
        if ps -p $CIRCUIT_PID > /dev/null; then
            echo "🛑 Stopping circuit job (PID: $CIRCUIT_PID)"
            kill $CIRCUIT_PID
            STOPPED=1
        fi
    fi
    
    if [ $STOPPED -eq 0 ]; then
        echo "✅ No running jobs found"
    else
        echo "⏱️  Waiting for jobs to stop..."
        sleep 3
        echo "✅ Jobs stopped"
    fi
}

# Main menu
case "${1:-menu}" in
    "semantic")
        run_semantic_job
        ;;
    "circuit")
        run_circuit_job
        ;;
    "status")
        show_status
        ;;
    "stop")
        stop_jobs
        ;;
    "menu"|*)
        echo "🎯 Choose an option:"
        echo "1) Run semantic connectivity job (Job 1)"
        echo "2) Run circuit complexity job (Job 2)"  
        echo "3) Show job status"
        echo "4) Stop all jobs"
        echo "5) Exit"
        echo ""
        read -p "Enter choice (1-5): " choice
        
        case $choice in
            1)
                run_semantic_job
                ;;
            2)
                run_circuit_job
                ;;
            3)
                show_status
                ;;
            4)
                stop_jobs
                ;;
            5)
                echo "👋 Goodbye!"
                exit 0
                ;;
            *)
                echo "❌ Invalid choice"
                exit 1
                ;;
        esac
        ;;
esac

echo ""
echo "💡 Usage tips:"
echo "   ./run_overnight_jobs.sh semantic  # Start Job 1"
echo "   ./run_overnight_jobs.sh circuit   # Start Job 2"
echo "   ./run_overnight_jobs.sh status    # Check progress"
echo "   ./run_overnight_jobs.sh stop      # Stop all jobs"
echo ""
echo "📊 Monitor logs with:"
echo "   tail -f semantic_job_console.log"
echo "   tail -f circuit_job_console.log" 