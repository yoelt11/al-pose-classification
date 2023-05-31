'''Helper fucntions to load hdf5 datasets '''
import h5py

# Helper function to recursively save nested dictionaries
def save_dict_to_hdf5(output_dir, dictionary):
    print("-- creating dataset -- ")
    #print(dictionary.keys())
    # -- open new file
    with h5py.File(output_dir + ".h5", 'w') as f:
        # -- create groups
        dataset_props = f.create_group('props')
        dataset = f.create_group('dataset')
        # -- populate properties
        for key in dictionary["props"]:
            dataset_props[key] = dictionary['props'][key]

        for i, entry in enumerate(dictionary["dataset"]):
            if isinstance(entry, dict):
                print(f"saving {entry['file_name']}, img_data shape: {entry['img_data'].shape}, kp_data shape: {entry['kp_data'].shape}")
                if i == 0:
                    # -- creates first entry
                    for key in entry:
                        if isinstance(entry[key], str):
                            dataset.create_dataset(key, 
                                                    data=[entry[key]],
                                                    chunks=True,
                                                    maxshape=(None,))
                        else:
                            data_shape = list(entry[key].shape)
                            data_shape[0] = None
                            data_shape = tuple(data_shape)
                            dataset.create_dataset(key, 
                                                    data=entry[key],
                                                    chunks=True,
                                                    maxshape=data_shape)
                else:
                    # -- expands dataset
                    for key in entry:
                        dataset[key].resize(dataset[key].shape[0] + 1, axis=0)
                        dataset[key][-1] = entry[key]

        f.close() # save file

# Helper function to recursively load nested dictionaries
def load_from_hdf5(path):
    print("-- loading dataset --")
    with h5py.File(path, 'r+') as f:
        dataset = f['dataset']
        dataset_keys = [key for key in dataset.keys()]
        print(dataset_keys)
        if 'target' in dataset_keys:
            img_data = dataset['img_data'][()]
            kp_data = dataset['kp_data'][()]
            file_name = dataset['file_name'][()]
            file_name = [f.decode("utf-8") for f in file_name]
            targets = dataset['target'][()]

            dataset_props = f['props']
            prop_keys, prop_val = [], []
            for key in dataset_props:
                prop_keys.append(key)
                prop_val.append(dataset_props[key][()])
            dataset_props = dict(zip(prop_keys, prop_val))
            return dataset_props, img_data, kp_data, file_name, targets
        else:
            img_data = dataset['img_data'][()]
            kp_data = dataset['kp_data'][()]
            file_name = dataset['file_name'][()]
            file_name = [f.decode("utf-8") for f in file_name]

            dataset_props = f['props']
            prop_keys, prop_val = [], []
            for key in dataset_props:
                prop_keys.append(key)
                prop_val.append(dataset_props[key][()])
            dataset_props = dict(zip(prop_keys, prop_val))
            return dataset_props, img_data, kp_data, file_name


if __name__=='__main__':
    import numpy as np
    from datetime import datetime
    # -- create dummy dict
    output_dir = "./datasets/unlabeled_datasets/" + "test" + str(round(datetime.now().timestamp())) 
    # Create a nested dictionary

    data = []
    for i in range(500):
        d = {
            "img_data": np.random.rand(1,20,480,360,3).astype(dtype=np.uint8),
            "kp_data":  np.random.rand(1,20,17,3).astype(np.float64),
            "file_name": "file_1.txt"
            }
        data.append(d)

    dictionary = {
        "props": {
            "frame_count": 100,
            "fps": 30,
            "total_frames": 500,
            "width": 640,
            "height": 480
        },
        "dataset": data
    }
    
    save_dict_to_hdf5(output_dir, dictionary)
    dataset_props, dataset = load_dict_from_hdf5(output_dir + '.h5') 