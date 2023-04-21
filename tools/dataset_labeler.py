import json
import cv2
import numpy as np
from datetime import datetime

if __name__== "__main__":
    # -- dataset to work with
    dataset_path = "./datasets/unlabeled_datasets/"
    dataset_name = "unlabeled_dataset_1682071747.json"
    # -- output dir
    output_dir = f"./datasets/labeled_datasets/labeled_dataset_{str(round(datetime.now().timestamp()))}.json" 
    # -- load dataset
    with open(dataset_path + dataset_name, 'r') as f:
        dataset = json.load(f)
    print(dataset['props'])
    data = dataset['dataset']
    # -- create labeled dataset
    labeled = []
    for video in data:
        video_name = video["file_name"]
        frames = video["data"]
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
                    target = [1,0,0,0,0,0,0]
                case 'standing':
                    target = [0,1,0,0,0,0,0]
                case 'drinking':
                    target = [0,0,1,0,0,0,0]
                case 'waving':
                    target = [0,0,0,1,0,0,0]
                case 'clapping':
                    target = [0,0,0,0,1,0,0]
                case 'walking':
                    target = [0,0,0,0,0,1,0]
                case 'none':
                    target = [0,0,0,0,0,0,1]
                case _ :
                    print(f"[Error] No target named {label}")
        # -- append label to new dataset
        video['label'] = label
        video['target'] = target
        labeled.append(video)
        # -- destroy current video window
        cv2.destroyWindow(video_name)
    # -- export dataset
    new_dataset = {'props': dataset['props'], 'dataset': labeled}

    json_str = json.dumps(new_dataset)

    with open(output_dir, 'w') as f:
        f.write(json_str)
    
        

