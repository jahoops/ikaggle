# Read in the feature_combination_results.txt file
with open('/home/USER/Projects/ikaggle/KISS/data/feature_combination_results.txt', 'r') as file:
    lines = file.readlines()
    # Skip the header line
    lines = lines[1:]
    # Skip blank lines
    lines = [line.strip() for line in lines if line.strip()]
    # Split each line into feature pair and RMSLE score
    lines = [line.rsplit(',', 1) for line in lines]
    # Convert feature pair to tuple and score to float
    lines = [(eval(line[0]), float(line[1])) for line in lines]
    # Sort results by RMSLE score
    lines.sort(key=lambda x: x[1])
# Save in order to new file feature_combos_ranked.txt
with open('/home/USER/Projects/ikaggle/KISS/data/feature_combos_ranked.txt', 'w') as file:
    for pair, score in lines:
        file.write(f"{pair}, {score}\n")
