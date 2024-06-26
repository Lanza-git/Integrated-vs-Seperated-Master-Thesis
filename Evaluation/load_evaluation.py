

# Load packages
def load_packages():
    # General imports
    import subprocess
    import sys


    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    install('pandas')
    install('numpy')
    install('pickle')


load_packages()



def load_directory(file_path):
    import pickle
    # Open the file in read-binary mode
    with open(file_path, 'rb') as file:
        # Load the dictionary
        data_dict = pickle.load(file)

    # Print the dictionary
    for dict in data_dict:
        print(dict)
    return data_dict


def load_metadata(directory):
    """ Load metadata from all models in a directory
    
    Input:
    - directory: str, path to the directory containing the models
    
    Output:
    """
    import os
    import pickle
    import pandas as pd
    metadata = []
    for file_name in os.listdir(directory):
        if file_name.endswith('meta.pkl'):
            with open(os.path.join(directory, file_name), 'rb') as f:
                data = pickle.load(f)
                metadata.append({
                    'file_name': file_name,
                    'hyperparameter': data['hyperparameter'],
                    'profit': data['profit'],
                    'elapsed_time': data['elapsed_time'],
                    'peak_memory' : data['peak_memory'],
                    'avg_memory': data['avg_memory']
                })
    return metadata


if __name__ == '__main__':

    file_path = "/pfs/work7/workspace/scratch/ma_elanza-thesislanza/dataset_list.pkl"

    # Load the dictionary
    data_dict = load_directory(file_path)

    metadata_list = []
    for dict in data_dict:
        metadata = load_metadata(dict['folder_path'])
        metadata_list.append(metadata)

    import pandas as pd
    metadata_list = pd.DataFrame(metadata_list)

    #Save dataframe to csv
    metadata_list.to_csv("/pfs/work7/workspace/scratch/ma_elanza-thesislanza/metadata.csv")

