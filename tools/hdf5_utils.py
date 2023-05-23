'''Helper fucntions to load hdf5 datasets '''
import h5py

# Helper function to recursively save nested dictionaries
def save_dict_to_hdf5(file, parent_group, dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            # Create a subgroup for nested dictionaries
            subgroup = parent_group.create_group(key)
            save_dict_to_hdf5(file, subgroup, value)
        else:
            # Save leaf nodes as datasets
            parent_group.create_dataset(key, data=value)

# Helper function to recursively load nested dictionaries
def load_dict_from_hdf5(group):
    dictionary = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            # Load subgroup as a nested dictionary
            dictionary[key] = load_dict_from_hdf5(group[key])
        else:
            # Load dataset as a leaf node
            dictionary[key] = group[key][()]
    return dictionary

if __name__=='__main__':
    # -- create dummy dict
    # Create a nested dictionary
    value1 = 0
    value2 = 0
    value3 = 0
    value4 = 0

    data_dict = {
        'key1': value1,
        'key2': {
            'subkey1': value2,
            'subkey2': value3,
        },
        'key3': {
            'subkey3': {
                'subsubkey1': value4,
            },
        },
        # ...
    }
    # -- save
    # Open an HDF5 file in write mode
    with h5py.File('data.h5', 'w') as file:
        # Create a group to store the dictionary
        group = file.create_group('dictionary')
    
        # Save the nested dictionary recursively
        save_dict_to_hdf5(file, group, data_dict)

    # -- load
    # Open the HDF5 file in read mode
    with h5py.File('data.h5', 'r') as file:
        # Access the dictionary group
        group = file['dictionary']
    
        # Load the nested dictionary recursively
        loaded_dict = load_dict_from_hdf5(group)
    
    print(loaded_dict)
