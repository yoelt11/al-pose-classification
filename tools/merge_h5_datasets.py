import sys
import numpy as np
from datetime import datetime
from hdf5_utils import save_dict_to_hdf5, load_from_hdf5

if __name__ == '__main__':
    # -- get files to merge 
    dataset_files = sys.argv[1:]
    # -- root path
    root_path = "./datasets/labeled_datasets/"
    # merge process
    for i, file in enumerate(dataset_files):
        dataset_props, img_data, data, file_name, targets = load_from_hdf5(root_path + file)
        #dataset_props, _, data, _, targets = load_from_hdf5(root_path + file)
        if i == 0:
            merged_img_data = img_data
            #img_data = None
            merged_data = data
            #data = None
            merged_file_name = file_name
            #file_name = None
            merged_targets = targets
            #targets = None
            
        else:
            #merged_img_data = np.concatenate((merged_img_data, img_data), axis=0)
            #del img_data
            #img_data = None
            merged_data = np.concatenate((merged_data, data), axis=0)
            #del data
            #data = None
            #merged_file_name = np.concatenate((merged_file_name, file_name), axis=0)
            #del file_name
            #file_name = None
            merged_targets = np.concatenate((merged_targets, targets), axis=0)
            #del targets
            #targets = None
        print(merged_data.shape)
    # -- compile data into one dictionary
    # img_data = [merged_img_data[i] for i in range(merged_img_data.shape[0])]
    # kp_data = [merged_data[i] for i in range(merged_data.shape[0])]
    # file_names = [merged_file_name[i] for i in range(len(merged_file_name))]
    # targets = [merged_targets[i] for i in range(merged_targets.shape[0])]
    dataset = [{
                      'img_data': np.zeros([1,20,360,640,3]),
                      'kp_data': np.expand_dims(merged_data[i], axis=0),
                      'file_name': 'None',
                      'target': np.expand_dims(merged_targets[i], axis=0)
                      } for i in range(merged_data.shape[0])]
    merged_dataset = {'props': dataset_props, 'dataset': dataset}
    # -- saved dictioanry
    output_dir = root_path + "merged_dataset_" + str(round(datetime.now().timestamp())) 
    save_dict_to_hdf5(output_dir, merged_dataset)
