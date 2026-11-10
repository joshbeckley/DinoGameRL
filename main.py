# Example file showing a circle moving on screen
from network import DQN, ReplayMemory
import torch
from torch import nn
import random
import sys
from pathlib import Path
from game import Game
from datetime import datetime
from training_metrics import TrainingMetrics
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
#https://medium.com/@mathparracho/training-an-agent-to-play-flappy-bird-dqn-modelling-reinforcement-learning-2cb16a9fa184
play_mode = False
play_model = ""
if "--play" in sys.argv:
    play_mode = True
    play_model = sys.argv[sys.argv.index("--play") + 1]
    
env = Game(play_mode)
episodes = 1000
if "--episodes" in sys.argv:
    episodes = int(sys.argv[sys.argv.index("--episodes") + 1])

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

num_actions = 2 
num_states = 5
model_directory = Path(__file__).resolve().parent
best_model_path = model_directory / "best_model.pt"
continued_model_path = model_directory / "best_model_continued.pt"
play_model_path = model_directory / play_model
metrics = TrainingMetrics(
    model_directory / "training_metrics.csv",
    model_directory / "training_metrics.svg",
)

rewards_per_episode = []
epsilon = 0.2
# epsilon_decay = 0.99995
epsilon_decay = 0.995
epsilon_min = 0.0001
mini_batch_size = 32
# network_sync_rate = 10
network_sync_rate = 1000
epsilon_history = []
memory = ReplayMemory(maxlen = 100000)

#instantiating networks
policy_dqn = DQN(num_states, num_actions, hidden_dim=256, enable_dueling_dqn=True).to(device)
target_dqn = DQN(num_states, num_actions, hidden_dim=256, enable_dueling_dqn=True).to(device)
if play_mode:
    checkpoint = torch.load(play_model_path, map_location=device, weights_only=True)
    weights = checkpoint.get("policy_state_dict", checkpoint)
    policy_dqn.load_state_dict(weights)
    policy_dqn.eval()
    print(f"Playing model from {play_model_path}")

    while True:
        state, _ = env.reset()
        episode_reward = 0.0
        terminated = False

        while not terminated:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                action = policy_dqn(state_tensor.unsqueeze(dim=0)).argmax(dim=1).item()
            state, reward, terminated, _, _ = env.step(action, episode_reward)
            episode_reward += reward

        print(f"Episode finished: reward={episode_reward:.2f}")

continue_training = "--continue" in sys.argv
if continue_training:
    checkpoint_path = (
        continued_model_path if continued_model_path.exists() else best_model_path
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    weights = checkpoint.get("policy_state_dict", checkpoint)
    policy_dqn.load_state_dict(weights)
    epsilon = epsilon_min
    print(f"Loaded model weights from {checkpoint_path}")
    print(f"Starting another {episodes} episodes; replay memory and optimizer start fresh.")
    output_model_path = continued_model_path
    start_episode = 1000
else:
    output_model_path = best_model_path
    start_episode = 0
target_dqn.load_state_dict(policy_dqn.state_dict()) #same as the target network

optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.0001)
loss_fn = nn.SmoothL1Loss()

# track number of steps taken. used for syncing policy => target network.
step_count=0

best_reward = -float("inf") ## see if it is right

metrics.reset()
for episode in range(start_episode, start_episode + episodes):

    state, _ = env.reset()  # initialize environment. reset returns (state,info).
    state = torch.tensor(state, dtype=torch.float, device=device)
    print(f"\n==========================\nIteration: {episode}:\nEpisode Reward: {best_reward}")
    #loop train in the episode
    stop_on_reward = 999
    terminated = False      #if the env has some goal (not the case of flappybird)
    episode_reward = 0.0    #used to accumulate rewards per episode
    episode_steps = 0
    jump_actions = 0

    while(not terminated and episode_reward < stop_on_reward): #stop if bird is dead or achieved 999 scores on the game

        ## epsilon-greedy algorithm !!!!!!!!!!
        if random.random() < epsilon:
            # select random action
            action = env.get_random_action()
            action = torch.tensor(action, dtype=torch.int64, device=device)
        else:
            # select best action
            with torch.no_grad():
                state_pytorch = state.unsqueeze(dim=0) # state.shape -> [12], state.unsqueeze(dim=0).shape -> [1,12] because pytorch expects a batch dimension
                action = policy_dqn(state_pytorch).squeeze().argmax() # action.shape -> [1, 2], action.squeeze(dim=0).shape -> [2], argmax outputs the position of the max(a,b)

        #execute action letsgoo
        new_state,reward,terminated,truncated,info = env.step(action.item(),episode_reward) #this env.step executes the action on the env
        episode_reward += reward
        episode_steps += 1
        jump_actions += int(action.item() == 1)

        new_state = torch.tensor(new_state, dtype=torch.float, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)

        #adding into the replay buffer
        memory.append((state, action, new_state, reward, terminated))
        step_count+=1
        if len(memory) >= mini_batch_size and step_count % 4 == 0:
            mini_batch = memory.sample(mini_batch_size)
            optimize(mini_batch, policy_dqn, target_dqn)

            if step_count >= network_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0

        state = new_state
    rewards_per_episode.append(episode_reward)
    average_reward = sum(rewards_per_episode[-100:]) / min(
        len(rewards_per_episode), 100
    )
    print(
        f"Episode {episode}: reward={episode_reward:.2f}, "
        f"avg100={average_reward:.2f}, "
        f"cacti_passed={env.environment.cactus_passed['count']}, "
        f"epsilon={epsilon:.3f}"
    )
    metrics.record_episode(
        {
            "episode": episode,
            "reward": episode_reward,
            "rolling_mean_100": average_reward,
            "steps": episode_steps,
            "cacti_passed": env.environment.cactus_passed["count"],
            "jumps": jump_actions,
            "epsilon": epsilon,
        }
    )
    # check if it is the best to save the model
    if episode_reward > best_reward:
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')}: New best reward {episode_reward:0.1f} at episode {episode}, saving model...")
            torch.save(policy_dqn.state_dict(), output_model_path)
            best_reward = episode_reward

    # also an importat part: the optimization using the memory buffer, epsilon decay and double dqn !!!!!!!!!!
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    epsilon_history.append(epsilon)

print(f"Training metrics saved to {metrics.csv_path}")
for plot_path in metrics.plot():
    print(f"Training plot saved to {plot_path}")
