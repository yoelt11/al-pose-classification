import sys
sys.path.append("../models/")
from pose_classification import AcT

if __name__=='__main__':
    B, T, N, C = 40, 20, 17, 3
    model = AcT.AcT(B,T,B,C,1,20)
