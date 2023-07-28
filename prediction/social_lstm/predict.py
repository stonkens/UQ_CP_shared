import pandas as pd
import numpy as np
import ndjson
import os
import sys
import pickle
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection

# Requires installing trajnet++tools and trajnet++baselines
# https://github.com/vita-epfl/trajnetplusplustools
# https://github.com/vita-epfl/trajnetplusplusbaselines

import trajnetplusplustools

from trajnetplusplusbaselines.trajnetbaselines.lstm.gridbased_pooling import GridBasedPooling
from trajnetplusplusbaselines.trajnetbaselines.lstm.lstm import LSTM, LSTMPredictor, drop_distant

# model dependent


# Load model
def load_model(path):
	pool = GridBasedPooling(type_='social', hidden_dim=128,
                        cell_side=0.6, n=16, front=True,
                        out_dim=256, embedding_arch='two_layer',
                        constant=0, pretrained_pool_encoder=None,
                        norm=0, layer_dims=[1024], latent_dim=16)

	model = LSTM(pool=pool,
	             embedding_dim=64,
	             hidden_dim=128,
	             goal_flag=False,
	             goal_dim=0)


	with open(path, 'rb') as f:
	    checkpoint = torch.load(f)
	    
	pretrained_state_dict = checkpoint['state_dict']

	model.load_state_dict(pretrained_state_dict, strict=False)
	return model


def recover_pos(last_obs, delta):
    output = [last_obs]
    for i, d in enumerate(delta):
        output.append(output[i]+d[...,:2])
        
    return torch.stack(output)
    
def predict_one(model, scene):
    scene1 = trajnetplusplustools.Reader.paths_to_xy(scene[2])
    scene1 = torch.Tensor(scene1)

    seq_length = 21
    obs_length = 9
    pred_length = 12
    batch_scene_goal = []
    batch_split = [0]
    batch_split.append(int(scene1.shape[1]))
    batch_split = torch.Tensor(batch_split).long()

    observed = scene1[:obs_length]
    prediction_truth = scene1[obs_length:seq_length-1]#.clone()  ## CLONE
    targets = scene1[obs_length:seq_length] - scene1[obs_length-1:seq_length-1]

    with torch.no_grad():
        rel_outputs, _ = model(observed, batch_scene_goal, batch_split, prediction_truth)
        
    delta = rel_outputs[-12:]
    out = recover_pos(observed[-1], delta)
    

def predict_file(model, fullpath):

	reader = trajnetplusplustools.Reader(fullpath, scene_type='paths')
	scenes = [(file, s_id, s) for s_id, s in reader.scenes()]

	results = []
	for s in tqdm(model, scenes):
	    gt, out = predict_one(s)
	    results.append((gt.detach().numpy(), out.detach().numpy()))


if __name__ == '__main__':

	model_path = '../../trained_model/social_lstm/lstm_social_None.pkl.epoch25.state'
	model = load_model(model_path)

	file = 'cff_06.ndjson'
	fullpath  = 'train/real_data/' + file
	results = predict_file(model, fullpath)
	
	with open('trajnetplusplusbaselines/OUTPUT_BLOCK/trajdata/outputs.pkl', 'wb') as f:
    	pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)





