#!/usr/bin/env python3
"""
Job management utility for semantic connectivity pipeline
Provides status monitoring, log viewing, and job control
"""

import argparse
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def run_command(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return ""


def list_jobs():
    """List all running and completed jobs."""
    print("=== Pipeline Jobs ===\n")
    
    # Check tmux sessions
    tmux_output = run_command("tmux ls 2>/dev/null | grep pipeline")
    
    print("Running Jobs (tmux sessions):")
    if tmux_output:
        for line in tmux_output.split('\n'):
            if line:
                session_name = line.split(':')[0]
                created = line.split('(created')[1].split(')')[0] if 'created' in line else ''
                print(f"  - {session_name} {created}")
    else:
        print("  No running jobs")
    
    print("\nCompleted Jobs (log directories):")
    log_dirs = sorted(Path('.').glob('logs/*'))
    for log_dir in log_dirs[-10:]:  # Show last 10
        # Check if job completed
        summary_file = log_dir / 'summary.txt'
        status = "✓ Completed" if summary_file.exists() else "⚠ Incomplete"
        print(f"  - {log_dir.name} {status}")
    
    if len(log_dirs) > 10:
        print(f"  ... and {len(log_dirs) - 10} more")


def show_job_status(job_name):
    """Show detailed status of a specific job."""
    # Find log directory
    log_dirs = list(Path('.').glob(f'logs/*{job_name}*'))
    if not log_dirs:
        print(f"No job found matching: {job_name}")
        return
    
    log_dir = sorted(log_dirs)[-1]  # Most recent
    print(f"=== Job Status: {log_dir.name} ===\n")
    
    # Check if still running
    session_name = log_dir.name.replace('logs/', 'pipeline_')
    tmux_check = run_command(f"tmux ls 2>/dev/null | grep {session_name}")
    
    if tmux_check:
        print("Status: 🏃 Running")
        print(f"Attach: tmux attach -t pipeline_{log_dir.name}")
    else:
        print("Status: 💤 Not running")
    
    # Show logs
    print("\nLog files:")
    for log_file in log_dir.glob('*.log'):
        size_mb = log_file.stat().st_size / (1024 * 1024)
        print(f"  - {log_file.name}: {size_mb:.1f} MB")
    
    # Show last lines of log
    latest_log = max(log_dir.glob('*.log'), key=os.path.getmtime, default=None)
    if latest_log:
        print(f"\nLast 20 lines of {latest_log.name}:")
        print("-" * 60)
        last_lines = run_command(f"tail -20 {latest_log}")
        print(last_lines)
    
    # Show summary if exists
    summary_file = log_dir / 'summary.txt'
    if summary_file.exists():
        print("\nJob Summary:")
        print("-" * 60)
        print(summary_file.read_text())


def analyze_results(results_dir):
    """Quick analysis of results."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    print(f"=== Analysis of {results_dir} ===\n")
    
    # Load CSVs
    summary_df = pd.read_csv(results_path / 'word_summary.csv')
    
    print(f"Total words analyzed: {len(summary_df)}")
    print(f"\nPolysemy distribution:")
    print(summary_df['polysemy_score'].value_counts().head(10))
    
    print(f"\nConnectivity statistics:")
    print(f"  Mean: {summary_df['total_connectivity'].mean():.1f}")
    print(f"  Std:  {summary_df['total_connectivity'].std():.1f}")
    print(f"  Min:  {summary_df['total_connectivity'].min()}")
    print(f"  Max:  {summary_df['total_connectivity'].max()}")
    
    print(f"\nFeature statistics:")
    print(f"  Mean: {summary_df['total_features'].mean():.1f}")
    print(f"  Std:  {summary_df['total_features'].std():.1f}")
    
    # Correlation
    corr = summary_df[['polysemy_score', 'total_connectivity', 'total_features']].corr()
    print(f"\nCorrelations:")
    print(corr)
    
    # Top words
    print(f"\nTop 10 words by connectivity:")
    top_conn = summary_df.nlargest(10, 'total_connectivity')[['word', 'total_connectivity', 'polysemy_score']]
    print(top_conn.to_string(index=False))


def kill_job(session_name):
    """Kill a running tmux session."""
    if not session_name.startswith('pipeline_'):
        session_name = f'pipeline_{session_name}'
    
    result = subprocess.run(f"tmux kill-session -t {session_name}", shell=True)
    if result.returncode == 0:
        print(f"Killed session: {session_name}")
    else:
        print(f"Failed to kill session: {session_name}")
        print("Session may not exist or already terminated")


def main():
    parser = argparse.ArgumentParser(
        description="Manage semantic connectivity pipeline jobs"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all jobs')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show job status')
    status_parser.add_argument('job_name', help='Job name or partial match')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('results_dir', help='Results directory')
    
    # Kill command
    kill_parser = subparsers.add_parser('kill', help='Kill running job')
    kill_parser.add_argument('session_name', help='Session name')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_jobs()
    elif args.command == 'status':
        show_job_status(args.job_name)
    elif args.command == 'analyze':
        analyze_results(args.results_dir)
    elif args.command == 'kill':
        kill_job(args.session_name)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()