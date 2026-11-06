# Example file showing a circle moving on screen
from network import DQN, ReplayMemory
import torch
from torch import nn
import random
from game import Game
from datetime import datetime
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
#https://medium.com/@mathparracho/training-an-agent-to-play-flappy-bird-dqn-modelling-reinforcement-learning-2cb16a9fa184
env = Game()

def optimize(mini_batch, policy_dqn, target_dqn):

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
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                #calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        #calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = loss_fn(current_q, target_q)

        # backprop step
        optimizer.zero_grad()  
        loss.backward()             # compute gradients
        optimizer.step() #propagate it through the model

num_actions = 2 
num_states = 5

rewards_per_episode = []
epsilon = 1
epsilon_decay = 0.99995
epsilon_min = 0.00001
mini_batch_size = 32
network_sync_rate = 10
epsilon_history = []
memory = ReplayMemory(maxlen = 100000)

#instantiating networks
policy_dqn = DQN(num_states, num_actions, hidden_dim=256, enable_dueling_dqn=True).to(device)
target_dqn = DQN(num_states, num_actions, hidden_dim=256, enable_dueling_dqn=True).to(device)
target_dqn.load_state_dict(policy_dqn.state_dict()) #same as the target network

optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

# track number of steps taken. used for syncing policy => target network.
step_count=0

best_reward = -float("inf") ## see if it is right

for episode in range(1000):

    state, _ = env.reset()  # initialize environment. reset returns (state,info).
    state = torch.tensor(state, dtype=torch.float, device=device)
    print(f"\n==========================\nIteration: {episode}:\nEpisode Reward: {best_reward}")
    #loop train in the episode
    stop_on_reward = 999
    terminated = False      #if the env has some goal (not the case of flappybird)
    episode_reward = 0.0    #used to accumulate rewards per episode

    while(not terminated and episode_reward < stop_on_reward): #stop if bird is dead or achieved 999 scores on the game

        ## epsilon-greedy algorithm !!!!!!!!!!
        if random.random() < 0.1: ##floating number [0,1] < epsilon
            # select random action
            action = env.get_random_action()
            action = torch.tensor(action, dtype=torch.int64, device=device)
        else:
            # select best action
            with torch.no_grad():
                state_pytorch = state.unsqueeze(dim=0) # state.shape -> [12], state.unsqueeze(dim=0).shape -> [1,12] because pytorch expects a batch dimension
                action = policy_dqn(state_pytorch).squeeze().argmax() # action.shape -> [1, 2], action.squeeze(dim=0).shape -> [2], argmax outputs the position of the max(a,b)

        #execute action letsgoo
        new_state,reward,terminated,truncated,info = env.step(action.item()) #this env.step executes the action on the env
        episode_reward += reward

        new_state = torch.tensor(new_state, dtype=torch.float, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)

        #adding into the replay buffer
        memory.append((state, action, new_state, reward, terminated))
        step_count+=1

        state = new_state
    rewards_per_episode.append(episode_reward)
    # check if it is the best to save the model
    if episode_reward > best_reward:
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model...")
            torch.save(policy_dqn.state_dict(), "best_model.pt")
            best_reward = episode_reward

    # also an importat part: the model optimization using the memory buffer, epsilon decay and double dqn !!!!!!!!!!
    if len(memory) > mini_batch_size:
        mini_batch = memory.sample(mini_batch_size)
        optimize(mini_batch, policy_dqn, target_dqn) #optimization step

        # espilon decay... going into the exploitation and avoiding exploration
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        epsilon_history.append(epsilon)

        # the double dqn idea: copy policy network to target network after a certain number of steps
        if step_count > network_sync_rate:
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count=0