#!/bin/bash
# Clean up script for fresh start

# Create backup of important original data
echo "Backing up original data..."
mkdir -p /home/USER/Projects/ikaggle/KISS/backup
cp /home/USER/Projects/ikaggle/KISS/data/*.csv /home/USER/Projects/ikaggle/KISS/backup/ 2>/dev/null

# Remove previous results and intermediate files
echo "Removing previous results..."
rm -rf /home/USER/Projects/ikaggle/KISS/results/complete_pipeline
rm -rf /home/USER/Projects/ikaggle/KISS/results/group_models
rm -rf /home/USER/Projects/ikaggle/KISS/checkpoints/*
rm -f /home/USER/Projects/ikaggle/KISS/results/*submission*.csv

# Create fresh directories
echo "Creating fresh directories..."
mkdir -p /home/USER/Projects/ikaggle/KISS/results/fixed_pipeline
mkdir -p /home/USER/Projects/ikaggle/KISS/checkpoints

echo "Cleanup complete! Ready for a fresh start."
