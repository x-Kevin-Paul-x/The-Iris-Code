import pandas as pd
from sklearn.datasets import load_iris
import os

def download_and_save_iris_dataset():
    """
    Downloads the Iris dataset and saves it as a CSV file.
    """
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Create a directory for data if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    file_path = os.path.join('data', 'iris.csv')
    df.to_csv(file_path, index=False)
    print(f"Iris dataset saved to {file_path}")

if __name__ == "__main__":
    download_and_save_iris_dataset()
