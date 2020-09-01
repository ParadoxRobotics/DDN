
# Generate correspondence point between 2 RGB images given their depth and pose
# in the world frame
# Author : Munch Quentin, 2020

import numpy as np
import math
import copy
import random
from random import randint
import matplotlib.pyplot as plt

import cv2

class GenerateCorrespondenceRGBD():
    def __init__(self, intrinsic_mat, depth_scale, depth_margin, number_match, number_non_match):
        super(GenerateCorrespondenceRGBD, self).__init__()
        self.intrinsic_mat = intrinsic_mat
        self.depth_scale = depth_scale
        self.depth_margin = depth_margin
        self.number_match = number_match
        self.number_non_match = number_non_match

    def RGBD_Match(self, in_A, depth_A, pose_A, in_B, depth_B, pose_B, mask):
        # Image and depth map need to aligned :
        #  - in_A/in_B -> [H,W,C] (OpenCV format)
        #  - depth_A/depth_B -> [H,W]

        # Init match list
        valid_match_A = []
        valid_match_B = []

        # Init 3D point
        # 3D point in the camera reference (A)
        Pt_A = np.zeros((4,1), dtype="float32")
        Pt_A[3,0] = 1
        # Absolute 3D point in world reference
        Pt_W = np.zeros((4,1), dtype="float32")
        Pt_W[3,0] = 1
        # 3D point in the camera reference (B)
        Pt_B = np.zeros((4,1), dtype="float32")
        Pt_B[3,0] = 1

        # Init (u,v) point
        uv_A = np.zeros(2, dtype="int")
        uv_B = np.zeros(2, dtype="int")

        while len(valid_match_A) != self.number_match:
            if mask == None:
                # Generate random point in the [uA,vA] image space
                randval = np.random.rand(2)
                uv_A[0] = np.floor(randval[0]*in_A.shape[0]) # H
                uv_A[1] = np.floor(randval[1]*in_A.shape[1]) # W
            else:
                um, vm = np.where(mask == 0)
                randval = np.random.rand(1)
                id = int(np.floor(randval[0]*len(um)))
                uv_A[0] = um[id] # H
                uv_A[1] = vm[id] # W
            # Evaluate depth (DA>0)
            if depth_A[uv_A[0], uv_A[1]] > 0:
                # Generate [xA,yA,zA] points (camera parameters + depth)
                Pt_A[2,0] = depth_A[uv_A[0], uv_A[1]]/self.depth_scale
                Pt_A[0,0] = ((uv_A[1]-self.intrinsic_mat[0,2])*Pt_A[2,0])/self.intrinsic_mat[0,0]
                Pt_A[1,0] = ((uv_A[0]-self.intrinsic_mat[1,2])*Pt_A[2,0])/self.intrinsic_mat[1,1]
            else:
                continue

            # calculate transform
            Pt_WA = np.dot(pose_B, Pt_A) # position camera frame A in the world frame
            Pt_BW = np.dot(np.linalg.inv(pose_A), Pt_WA) # world frame to camera frame B

            # Calculate [xB,yB,zB] point in [uB,vB] image space
            uv_B[1] = ((self.intrinsic_mat[0,0]*Pt_BW[0,0])/Pt_BW[2,0])+self.intrinsic_mat[0,2]
            uv_B[0] = ((self.intrinsic_mat[1,1]*Pt_BW[1,0])/Pt_BW[2,0])+self.intrinsic_mat[1,2]

            # Evaluate frustum consistency, depth DB > 0 and occlusion
            if (uv_B[0]<in_B.shape[0]) and (uv_B[0]>0) and (uv_B[1]<in_B.shape[1]) and (uv_B[1]>0):
                if (depth_B[uv_B[0],uv_B[1]]>0) and depth_B[uv_B[0], uv_B[1]]/self.depth_margin > Pt_B[2,0]-self.depth_margin:
                    valid_match_A.append(copy.deepcopy(uv_A))
                    valid_match_B.append(copy.deepcopy(uv_B))
                else:
                    continue
            else:
                continue
        # return all match in image A and image B
        return valid_match_A, valid_match_B


    def RGBD_Non_Match(self, valid_match_A, valid_match_B):
        # Image and depth map need to aligned :
        #  - in_A / in_B -> [H,W,C] (OpenCV format)
        #  - depth_A / depth_B -> [H,W]

        # Init non-match list
        non_valid_match_A = []
        non_valid_match_B = []

        while len(non_valid_match_A) != self.number_non_match:
            # sample random point from good match in image A and image B
            randval = np.random.rand(2)
            index_A = int(np.floor(randval[0]*len(valid_match_A)))
            index_B = int(np.floor(randval[1]*len(valid_match_B)))
            # store the point in list
            non_valid_match_A.append(copy.deepcopy(valid_match_A[index_A]))
            non_valid_match_B.append(copy.deepcopy(valid_match_B[index_B]))

        # return all non-match in image A and image B
        return non_valid_match_A, non_valid_match_B
