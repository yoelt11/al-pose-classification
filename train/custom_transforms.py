import torch
import numpy as np

def kp_norm(pose):
    '''
    Sets all kp distance with respect the center of the person
    [K, [y, x, score]]
    '''
    if not isinstance(pose, torch.Tensor):
        pose = torch.tensor(pose)
    with torch.no_grad():
    # -- get center coordinate
        body_kp = [5,6,11,12] # r_shoulder, l_shoulder, r_hip, l_hipt
        body_coords = pose[body_kp,:2]
        body_coords = body_coords.T
        center_coord = body_coords.mean(1)
        nose_coord = pose[0, :2]
        ref_dist = (center_coord - nose_coord).pow(2).sum(0).sqrt()
    # -- normalize wrt center
        pose = pose.T
        pose[0] = (pose[0] - center_coord[0]) * (ref_dist) #/ 640
        pose[1] = (pose[1] - center_coord[1]) * (ref_dist) #/ 360

    return pose.T
    


if __name__=="__main__":
    pose = torch.randn([17,3])
    pose_d = kp_norm(pose)



