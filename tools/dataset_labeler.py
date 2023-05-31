import cv2
import numpy as np
from datetime import datetime
import sys
from hdf5_utils import load_from_hdf5, save_dict_to_hdf5 
import os
import h5py

def create_folder_tree(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__== "__main__":
    dataset_name = sys.argv[1]
    # -- dataset to work with
    dataset_path = "./datasets/unlabeled_datasets/"
    # -- output dir
    create_folder_tree("./datasets/labeled_datasets/")
    output_dir = f"./datasets/labeled_datasets/labeled_dataset_{str(round(datetime.now().timestamp()))}" 

    dataset_props, img_data, kp_data, file_name = load_from_hdf5(dataset_path +  dataset_name)
    img_data = img_data[:5]
    kp_data = kp_data[:5]
    file_name = file_name[:5]
    num_vids = img_data.shape[0]
    # -- create labeled dataset
    dataset = []
    for v in range(num_vids):
        video_name = file_name[v]
        print(str(video_name))
        frames = img_data[v]
        label = ""
        target = None
        # -- keep playing video as long as label is empty
        while label == "" or target == None:
            for frame in frames:
                frame = np.array(frame, np.uint8) 
                cv2.imshow(video_name, frame)
                # -- break feed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            label = input("Enter a label for video: ")
            match label:
                case 'sitting':
                    target = [1,0,0,0,0,0,0,0]
                case 'standing':
                    target = [0,1,0,0,0,0,0,0]
                case 'drinking':
                    target = [0,0,1,0,0,0,0,0]
                case 'waving':
                    target = [0,0,0,1,0,0,0,0]
                case 'clapping':
                    target = [0,0,0,0,1,0,0,0]
                case 'walking':
                    target = [0,0,0,0,0,1,0,0]
                case 'picking':
                    target = [0,0,0,0,0,0,1,0]
                case 'none':
                    target = [0,0,0,0,0,0,0.1]
                case _ :
                    print(f"[Error] No target named {label}")
        # -- append label to new dataset
        dataset.append({'img_data': np.expand_dims(img_data[v], axis=0), "kp_data": np.expand_dims(kp_data[v], axis=0), "file_name": file_name[v], "target": np.expand_dims(np.array(target), axis=0)})
        # -- destroy current video window
        cv2.destroyWindow(video_name)
    # -- export dataset
    new_dataset = {'props': dataset_props, 'dataset': dataset}

    save_dict_to_hdf5(output_dir, new_dataset)

    #with jsonl.open(output_dir, 'w') as writer:
    #    writer.write(new_dataset)
    
    #json_str = json.dumps(new_dataset)

    #with open(output_dir, 'w') as f:
     #   f.write(json_str)
    
        

