import copy
from distutils.command.build import build
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from   rlig.pytorch.base import build_sequential, device


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
	def __init__(self, state_dim, neurons, action_dim, max_action):
		super(Actor, self).__init__()
		layers          = build_sequential(state_dim, *neurons, activate_final = True)
		self.model      = nn.Sequential(layers, nn.Linear(layers[-2].out_features, action_dim))
		self.max_action = max_action

	def forward(self, state):
		return self.max_action * self.model(state)


class Critic(nn.Module):
	def __init__(self, state_dim, neurons, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		layers          = build_sequential(state_dim + action_dim, *neurons, activate_final = True)
		self.Q1         = nn.Sequential(layers, nn.Linear(layers[-2].out_features, 1))

		# Q2 architecture
		layers          = build_sequential(state_dim + action_dim, *neurons, activate_final = True)
		self.Q2         = nn.Sequential(layers, nn.Linear(layers[-2].out_features, 1))


	def _forward(self, state, action):
		state_action   = torch.cat([state, action], axis = 1)
		value1, value2 = self.Q1(state_action), self.Q2(state_action)
		return value1, value2

	def forward(self, state, action):
		value1, value2 = self._forward(state, action)
		return torch.minimum(value1, value2)

	def loss(self, state, action, target):
		value1, value2 = self._forward(state, action)
		return F.mse_loss(value1, target) + F.mse_loss(value2, target)

