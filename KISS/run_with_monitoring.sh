#!/bin/bash
# Run the fixed pipeline with monitoring

echo "Starting fixed pipeline with validation checks..."
cd /home/USER/Projects/ikaggle/KISS

# Run the pipeline
python run_fixed_pipeline.py | tee pipeline_log.txt

# Check validation results
echo ""
echo "Analyzing validation results..."
if grep -q "ERROR" /home/USER/Projects/ikaggle/KISS/checkpoints/validation_log.json; then
  echo "⚠️  Warnings or errors found in validation log!"
  grep -A 3 -B 3 "ERROR" /home/USER/Projects/ikaggle/KISS/checkpoints/validation_log.json
else
  echo "✅ No validation errors found"
fi

# Check predictions for negative values
echo ""
echo "Checking for negative predictions..."
LATEST_SUBMISSION=$(ls -t /home/USER/Projects/ikaggle/KISS/results/fixed_pipeline/*submission*.csv | head -1)
if [ -f "$LATEST_SUBMISSION" ]; then
  NEGATIVES=$(python -c "import pandas as pd; df = pd.read_csv('$LATEST_SUBMISSION'); print((df['Premium Amount'] < 0).sum())")
  echo "Latest submission ($LATEST_SUBMISSION) has $NEGATIVES negative values"
  
  # Show submission statistics
  python -c "import pandas as pd; import numpy as np; df = pd.read_csv('$LATEST_SUBMISSION'); print(f'Min: {df[\"Premium Amount\"].min():.2f}'); print(f'Max: {df[\"Premium Amount\"].max():.2f}'); print(f'Mean: {df[\"Premium Amount\"].mean():.2f}'); print(f'Median: {df[\"Premium Amount\"].median():.2f}')"
else
  echo "No submission file found"
fi

echo ""
echo "Pipeline execution complete"
