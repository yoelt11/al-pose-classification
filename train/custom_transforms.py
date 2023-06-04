import torch

def kp_norm(pose):
    '''
    Sets all kp distance with respect the center of the person
    [K, [y, x, score]]
    '''
    body_kp = [5,6,11,12]
    body_coords = pose[body_kp,:2]
    body_coords = body_coords.T
    x = body_coords[0].mean()
    y = body_coords[1].mean()
    pose = pose.T
    #print(pose.shape)
    #print(x)
    pose[0] = pose[0] - x #/ 640
    pose[1] = pose[1] - y #/ 360
    #print(pose[0])
    
    return pose.T
    


if __name__=="__main__":
    pose = torch.randn([17,3])
    pose_d = kp_norm(pose)



