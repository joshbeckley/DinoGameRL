# Code from Matheus Ramos Parracho

import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn=enable_dueling_dqn
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            
            #value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            #advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)

        else:
            # vanilla dqn
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        latent_features = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            #value stream calculation
            v = F.relu(self.fc_value(latent_features))
            V = self.value(v)

            #advantages stream calculation
            a = F.relu(self.fc_advantages(latent_features))
            A = self.advantages(a)

            #Q-value calculation:
            """
              -> V: [batch_size, 1]
              -> A: [batch_size, action_dim]
              -> V + A = [batch_size, action_dim] + [batch_size, action_dim * 1]
              this means that the action tensor is automatically expanded
            """
            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            Q = self.output(latent_features)

        return Q



from collections import deque #a formal way to "bilateral arrays"
import random

# I mean, there is no mistery here, it is just to be formal
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

def optimize(mini_batch, policy_dqn, target_dqn, optimizer, loss_fn, device):

        #get elements from the env in the mini_batch
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        discount_factor_g = 0.99
        with torch.no_grad():
            enable_double_dqn = True #just to implement also without double_dqn
            if enable_double_dqn == True:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze(dim=1)
            else:
                #calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        #calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(dim=1)

        loss = loss_fn(current_q, target_q)

        # backprop step
        optimizer.zero_grad()  
        loss.backward()             # compute gradients
        optimizer.step() #propagate it through the model