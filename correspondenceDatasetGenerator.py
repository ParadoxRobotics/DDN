# Generate correspondence point between RGB image dataset given their depth and pose
# in the world frame
# Author : Munch Quentin, 2020

import numpy as np
import math
import copy
import random
from random import randint
import matplotlib.pyplot as plt
# import match generator
from matchGeneratorRGBD import *
import cv2
import os

# path to the RGBD dataset folder
folderPath = 'seq-01'
# count the number for txt file -> pose file
nbFile = len([f for f in os.listdir(folderPath) if f.endswith('.txt') and os.path.isfile(os.path.join(folderPath, f))])
# Dataset parameters :
# intrinsic matrix
fx = 585
fy = 585
cx = 320
cy = 240
CIP = np.array([(fx,0,cx),(0,fy,cy),(0,0,1)], dtype="float32")
# depth Parameters
depthMargin = 0.05 # in meter
depthScale = 1
# number of match and non-match per image couple
nbMatch = 100
nbNonMatch = 10
# generator instance
correspondenceGenerator = GenerateCorrespondenceRGBD(intrinsic_mat=CIP, depth_scale=depthScale, depth_margin=depthMargin, number_match=nbMatch, number_non_match=nbNonMatch)
# compute match for every image couple in the original dataset
for idx in range(0, int(nbFile)-1):
    # current file index
    IdA='{:06}'.format(idx)
    IdB='{:06}'.format(idx+1)
    # open pose
    poseA = np.loadtxt(folderPath+'/frame-'+IdA+'.pose.txt')
    poseB = np.loadtxt(folderPath+'/frame-'+IdB+'.pose.txt')
    # open RGB image
    imgA = cv2.imread(folderPath+'/frame-'+IdA+'.color.png',cv2.IMREAD_COLOR)
    imgB = cv2.imread(folderPath+'/frame-'+IdB+'.color.png',cv2.IMREAD_COLOR)
    # open Depth image
    depthA = cv2.imread(folderPath+'/frame-'+IdA+'.depth.png',cv2.COLOR_BGR2GRAY)
    depthB = cv2.imread(folderPath+'/frame-'+IdB+'.depth.png',cv2.COLOR_BGR2GRAY)
    # compute match / non-match
    matchA, matchB = correspondenceGenerator.RGBD_Match(imgA, depthA, poseA, imgB, depthB, poseB, mask=None)
    nonMatchA, nonMatchB = correspondenceGenerator.RGBD_Non_Match(matchA, matchB)
    # compute the match in a linear fashion for the contrastive loss
    imageWidth = imgA.shape[1]
    # create match file
    matchAFile = open("Dataset/MatchA/"+"matchA_"+IdA+".txt", "w")
    matchBFile = open("Dataset/MatchB/"+"matchB_"+IdB+".txt", "w")
    nonMatchAFile = open("Dataset/NonMatchA/"+"nonMatchA_"+IdA+".txt", "w")
    nonMatchBFile = open("Dataset/NonMatchB/"+"nonMatchB_"+IdB+".txt", "w")
    # update match in the txt file
    for m in range(len(matchA)):
        matchAFile.write(str(imageWidth*matchA[m][1]+matchA[m][0])+'\n')
        matchBFile.write(str(imageWidth*matchB[m][1]+matchB[m][0])+'\n')
    # update non-match in the txt file
    for n in range(len(nonMatchA)):
        nonMatchAFile.write(str(imageWidth*nonMatchA[n][1]+nonMatchA[n][0])+'\n')
        nonMatchBFile.write(str(imageWidth*nonMatchB[n][1]+nonMatchB[n][0])+'\n')
    # close file
    matchAFile.close()
    matchBFile.close()
    nonMatchAFile.close()
    nonMatchBFile.close()
    # store image A and B
    status = cv2.imwrite('Dataset/ImgA/imgA_'+IdA+'.png', imgA)
    status = cv2.imwrite('Dataset/ImgB/imgB_'+IdB+'.png', imgB)
