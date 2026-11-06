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