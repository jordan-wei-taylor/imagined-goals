from   rlig.pytorch.models import Actor, Critic
from   rlig.pytorch.base   import device
from   rlig.base           import Base

import torch
from   torch.nn import functional as F

import copy

class Agent(Base):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    

class TD3(Base):
    def __init__(self, state_dim, neurons, action_dim, max_action, discount = 0.99, tau = 0.005, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2):
        
        super().__init__(locals())

        self.actor = Actor(state_dim, neurons, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, neurons, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        numpy = self.actor(state).to(device).data.numpy()
        return numpy.reshape(numpy.shape[1:])


    def fit(self, state, action, next_state, reward, done):
        self.total_it += 1

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise       = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.discount * target_Q


        # Compute critic loss
        critic_loss = self.critic.loss(state, action, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.   actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    
    
