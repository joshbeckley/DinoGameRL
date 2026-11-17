# Example file showing a circle moving on screen
from network import DQN, ReplayMemory, optimize
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

# configuring the command line arguments
play_mode = False
play_model = ""
if "--play" in sys.argv:
    play_mode = True
    play_model = sys.argv[sys.argv.index("--play") + 1]
    
env = Game(play_mode)
episodes = 1000
if "--episodes" in sys.argv:
    episodes = int(sys.argv[sys.argv.index("--episodes") + 1])

# file paths
model_directory = Path(__file__).resolve().parent
models_directory = model_directory / "models"
results_directory = model_directory / "results"
models_directory.mkdir(parents=True, exist_ok=True)
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
play_model_path = Path(play_model)
if not play_model_path.is_absolute():
    if play_model_path.parts and play_model_path.parts[0] == "models":
        play_model_path = model_directory / play_model_path
    elif (model_directory / play_model_path).exists():
        play_model_path = model_directory / play_model_path
    else:
        play_model_path = models_directory / play_model_path
metrics = TrainingMetrics(
    results_directory / "training_metrics.csv",
    results_directory / "training_metrics.svg",
)

# instantiating network parameters and training loop hyperparameters
num_actions = 2 
num_states = 5
rewards_per_episode = []
epsilon = 0.2
epsilon_decay = 0.995
epsilon_min = 0.0001
mini_batch_size = 32
network_sync_rate = 1000
epsilon_history = []
memory = ReplayMemory(maxlen = 100000)

#instantiating networks
policy_dqn = DQN(num_states, num_actions, hidden_dim=256, enable_dueling_dqn=True).to(device)
target_dqn = DQN(num_states, num_actions, hidden_dim=256, enable_dueling_dqn=True).to(device)

# play a model instead of training
if play_mode:

    # load the model weights
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
            # no need to train - use no_grad
            with torch.no_grad():
                action = policy_dqn(state_tensor.unsqueeze(dim=0)).argmax(dim=1).item()
            state, reward, terminated, _, _ = env.step(action, episode_reward)
            episode_reward += reward

        # calculate reward for interest sake
        print(f"Episode finished: reward={episode_reward:.2f}")

# load current best model to train more
continue_training = "--continue" in sys.argv
if continue_training:
    saved_models = list(models_directory.glob("best_model*.pt"))
    if not saved_models:
        raise FileNotFoundError(f"No best model checkpoint found in {models_directory}")
    checkpoint_path = max(saved_models, key=lambda path: path.stat().st_mtime)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    weights = checkpoint.get("policy_state_dict", checkpoint)
    policy_dqn.load_state_dict(weights)
    epsilon = epsilon_min
    print(f"Loaded model weights from {checkpoint_path}")
    print(f"Starting another {episodes} episodes; replay memory and optimizer start fresh.")
    output_model_path = models_directory / f"best_model_continued_{run_timestamp}.pt"
else:
    output_model_path = models_directory / f"best_model_{run_timestamp}.pt"
    start_episode = 0
target_dqn.load_state_dict(policy_dqn.state_dict()) #same as the target network

optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=0.0001)
loss_fn = nn.SmoothL1Loss()

step_count=0
best_reward = -float("inf")
metrics.reset()

# main episode training loop
for episode in range(start_episode, start_episode + episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)
    print(f"\n==========================\nEpisode: {episode}:\nEpisode Reward: {best_reward}")

    # internal game/train loop per episode
    stop_on_reward = 85
    terminated = False
    episode_reward = 0.0
    episode_steps = 0
    jump_actions = 0

    while(not terminated and episode_reward < stop_on_reward):

        # epsilon-greedy action selection
        if random.random() < epsilon:
            # select random action
            action = env.get_random_action()
            action = torch.tensor(action, dtype=torch.int64, device=device)
        else:
            # select best action
            with torch.no_grad():
                state_pytorch = state.unsqueeze(dim=0)
                action = policy_dqn(state_pytorch).squeeze().argmax()

        # execute the action and update metrics
        new_state, reward, terminated, truncated, info = env.step(action.item(), episode_reward)
        episode_reward += reward
        episode_steps += 1
        jump_actions += int(action.item() == 1)

        new_state = torch.tensor(new_state, dtype=torch.float, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)

        # adding step returns to memory 
        memory.append((state, action, new_state, reward, terminated))
        step_count+=1
        
        # learn from a sample batch from memroy every 4 enviroment step
        if len(memory) >= mini_batch_size and step_count % 4 == 0:
            mini_batch = memory.sample(mini_batch_size)
            optimize(mini_batch, policy_dqn, target_dqn, optimizer, loss_fn, device)

            # target network copied from policy when step > 1000
            if step_count >= network_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0

        state = new_state

    # store metrics 
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

    # check if found new best reward
    if episode_reward > best_reward:
            print(f"{datetime.now().strftime('%m-%d %H:%M:%S')}: New best reward {episode_reward:0.1f} at episode {episode}, saving model...")
            torch.save(policy_dqn.state_dict(), output_model_path)
            best_reward = episode_reward

    # reduce exploration rate
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    epsilon_history.append(epsilon)

print(f"Training metrics saved to {metrics.csv_path}")
for plot_path in metrics.plot():
    print(f"Training plot saved to {plot_path}")
