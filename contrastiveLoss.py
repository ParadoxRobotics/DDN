
# Compute contrastive loss given network output and match/non-match
# Author : Munch Quentin, 2020

import math
from random import randint
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
