"""
epoch : 1 forward and backward pass of ALL training samples

batch_size : number of training samples in ONE forward and backward pass

number of iterations : number of passes per epoch , each pass using [batch_size] number of samples = number of passes needed to accomodate all training samples using batches of [batch_size]

e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations
"""

import torch
