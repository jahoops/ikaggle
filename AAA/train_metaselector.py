# Train the meta-selector model and save it to disk.
import pandas as pd
from meta_selector import MetaSelector

def load_data(file_path):
    data = pd.read_csv(file_path)   
    return data

def main():

    # Load submission information from /home/USER/Projects/ikaggle/AAA/submissions.JSON
    submissions = pd.read_json('/home/USER/Projects/ikaggle/AAA/submissions.JSON')

    # Load data from /home/USER/Projects/ikaggle/data/train.csv, /home/USER/Projects/ikaggle/data/validate.csv, /home/USER/Projects/ikaggle/data/test.csv
    train_data = load_data('/home/USER/Projects/ikaggle/data/train.csv')
    validate_data = load_data('/home/USER/Projects/ikaggle/data/validate.csv')
    test_data = load_data('/home/USER/Projects/ikaggle/data/test.csv')
    feature_list = load_data('/home/USER/Projects/ikaggle/data/feature_list.csv')

    feature_list = feature_list.dropna()

    meta_selector = MetaSelector()
    meta_selector.fit(train_data, validate_data, feature_list, submissions)
    
    test_predictions = meta_selector.predict(test_data)
    print("Test predictions:", test_predictions)

    meta_selector.save_model('/path/to/save/meta_selector_model.pkl')


if __name__ == "__main__":
    main()