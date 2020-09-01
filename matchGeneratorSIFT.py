
# Generate correspondence point between 2 RGB images using SIFT
# Author : Munch Quentin, 2020

import numpy as np
import math
import copy
import random
from random import randint
import matplotlib.pyplot as plt

import cv2

class GenerateCorrespondenceSIFT():
    def __init__(self, lowe_ratio, np_point, number_match, number_non_match):
        super(GenerateCorrespondenceSIFT, self).__init__()
        self.lowe_ratio = lowe_ratio
        self.number_match = number_match
        self.number_non_match = number_non_match
        # init keypoint extractor
        self.KPE = cv2.xfeatures2d.SIFT_create(nfeatures=np_point)
        # init KNN matcher (bruteforce)
        self.BFM = cv2.DescriptorMatcher_create("BruteForce")

    def SIFT_Match(self, in_A, in_B):
        # Init match list
        valid_match_A = []
        valid_match_B = []
        # Init (u,v) point
        uv_A = np.zeros(2, dtype="int")
        uv_B = np.zeros(2, dtype="int")
        # in_A/in_B -> [H,W,C] (OpenCV format)
        ptA, desA = self.KPE.detectAndCompute(in_A, None)
        ptB, desB = self.KPE.detectAndCompute(in_B, None)
        # compute correspondence
        knn_matches = self.BFM.knnMatch(desA, desB, 2)
        good_matches = []
        for m,n in knn_matches:
            if m.distance < self.lowe_ratio * n.distance:
                good_matches.append(m)
        # send the total number of match if size is not corresponding
        if len(good_matches) >= self.number_match:
            for id in range(0,self.number_match):
                uv_A[0] = ptA[good_matches[id].queryIdx].pt[0]
                uv_A[1] = ptA[good_matches[id].queryIdx].pt[1]
                uv_B[0] = ptB[good_matches[id].trainIdx].pt[0]
                uv_B[1] = ptB[good_matches[id].trainIdx].pt[1]
                valid_match_A.append(copy.deepcopy(uv_A))
                valid_match_B.append(copy.deepcopy(uv_B))
        else:
            for id in range(len(good_matches)):
                uv_A[0] = ptA[good_matches[id].queryIdx].pt[0]
                uv_A[1] = ptA[good_matches[id].queryIdx].pt[1]
                uv_B[0] = ptB[good_matches[id].trainIdx].pt[0]
                uv_B[1] = ptB[good_matches[id].trainIdx].pt[1]
                valid_match_A.append(copy.deepcopy(uv_A))
                valid_match_B.append(copy.deepcopy(uv_B))
        # return all match in image A and image B
        return valid_match_A, valid_match_B

    def SIFT_Non_Match(self, valid_match_A, valid_match_B):
        # Init non-match list
        non_valid_match_A = []
        non_valid_match_B = []
        if len(valid_match_A) > self.number_non_match and int(len(valid_match_A)/3) >= self.number_non_match:
            while len(non_valid_match_A) != self.number_non_match:
                # sample random point from good match in image A and image B
                randval = np.random.rand(2)
                index_A = int(np.floor(randval[0]*len(valid_match_A)))
                index_B = int(np.floor(randval[1]*len(valid_match_B)))
                # store the point in list
                non_valid_match_A.append(copy.deepcopy(valid_match_A[index_A]))
                non_valid_match_B.append(copy.deepcopy(valid_match_B[index_B]))
        else:
            # sample random point from good match in image A and image B
            randval = np.random.rand(2)
            index_A = int(np.floor(randval[0]*len(valid_match_A)))
            index_B = int(np.floor(randval[1]*len(valid_match_B)))
            # store the point in list
            non_valid_match_A.append(copy.deepcopy(valid_match_A[index_A]))
            non_valid_match_B.append(copy.deepcopy(valid_match_B[index_B]))
        # return all non-match in image A and image B
        return non_valid_match_A, non_valid_match_B
