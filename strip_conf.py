# write a program file to read a csv file, remove the 'Confidence' column and keep columns 'id' and 'PremiumAmount' and save the file as 'submission.csv' in the 'data/externa.' directory.
#
# The program file should be saved as 'strip_conf.py'.
#
# The csv file to be read is 'output/knn/finaltest_estimates.csv'.

import pandas as pd

# Read the csv file
data = pd.read_csv('output/SVR/SVR_finaltest_estimates_0.453.csv')

# Drop the 'Confidence' column
data.drop('Confidence', axis=1, inplace=True)

# Save the file as 'submission.csv' in the 'data/external' directory
data.to_csv('data/external/submission.csv', index=False)

print("File saved successfully.")
