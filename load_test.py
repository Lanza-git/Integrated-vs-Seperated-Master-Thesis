import numpy as np

file_path = "/pfs/work7/workspace/scratch/ma_elanza-thesislanza/bakery_datasets/v1/dataset_v1.csv"
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

if __name__ == "__main__":
    if data is None:
        print("Data is None")
    else:
        print(data)


# Now df is a DataFrame containing the data from the CSV file.