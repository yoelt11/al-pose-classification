import torch
import numpy as np
import math

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
        eye_coord = pose[1, :2]
        ear_coord = pose[4, :2]
        ref_dist = torch.linalg.norm((nose_coord - center_coord), dim=0, ord=2) 
    # -- normalize wrt center
        pose = pose.T
        pose[0] = ((pose[0] - center_coord[0]) * (ref_dist)) 
        pose[1] = ((pose[1] - center_coord[1]) * (ref_dist)) 

    return pose.T

def random_flip(pose):
    # -- randomly flips horizontally the coordinates; this can be used to augment data
    to_flip = torch.randint(0, 2, size=()).bool() # true of false

    if to_flip:
        #pose[:, :, 0] *= -1 # flip x-coordinate along nose
        # Compute the differences from the nose keypoint
        pose[:, 1:, 0] = 2 * pose[:, 0, 0].unsqueeze(-1) - pose[:, 1:, 0]
    
    return pose

def random_rotate(tensor, max_angle):
    # -- applies a random rotation to the coordinates between a given angle of rotation; this can be used to augment data
        # Generate a random angle between -max_angle and +max_angle
    # Generate a random angle between -max_angle and +max_angle
    angle = torch.rand(1) * (2 * max_angle) - max_angle

    # Convert angle to radians
    angle_rad = math.radians(angle.item())

    # Get the nose keypoint
    nose_keypoint = tensor[0]

    # Compute the cosine and sine of the rotation angle
    cos_theta = torch.cos(torch.tensor(angle_rad))
    sin_theta = torch.sin(torch.tensor(angle_rad))

    # Compute the differences from the nose keypoint
    tensor[1:, 0] = (tensor[1:, 0] - nose_keypoint[0]) * cos_theta - (tensor[1:, 1] - nose_keypoint[1]) * sin_theta + nose_keypoint[0]
    tensor[1:, 1] = (tensor[1:, 0] - nose_keypoint[0]) * sin_theta + (tensor[1:, 1] - nose_keypoint[1]) * cos_theta + nose_keypoint[1]

    return tensor


if __name__=="__main__":
    pose = torch.randn([17,3])
    pose_d = kp_norm(pose)



