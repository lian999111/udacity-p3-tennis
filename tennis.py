# %%
from replay_buffer import ReplayBuffer
from maddpg import MADDPG
from noise import OUNoise

from collections import deque
from unityagents import UnityEnvironment
import numpy as np
import torch


# %%
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")

# %% Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# %% Examine the state and action spaces
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# %% Train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {}
config["state_size"] = 24
config["action_size"] = 2
config["global_state_size"] = 2 * 24
config["global_action_size"] = 2 * 2
config["actor_hidden_sizes"] = [256, 128]
config["critic_hidden_sizes"] = [256, 128]
config["actor_lr"] = 1e-4
config["critic_lr"] = 1e-4
config["weight_decay"] = 0
config["device"] = device
config["tau"] = 1e-2            # mix ratio of soft update
config["gamma"] = 0.99          # discount factor

buffer_size = int(1e5)
batch_size = 128

replay_buffer = ReplayBuffer(buffer_size, batch_size)
maddpg = MADDPG(config)

max_episodes = 3000
steps_per_update = 1

score_window = deque(maxlen=100)

# OU Noise
ou_scale = 1.0                    # initial scaling factor
ou_decay = 0.9995                 # decay of the scaling factor ou_scale
ou_mu = 0.0                       # asymptotic mean of the noise
ou_theta = 0.15                   # magnitude of the drift term
ou_sigma = 0.20                   # magnitude of the diffusion term
noise_process = OUNoise(2*2, ou_mu, ou_theta, ou_sigma)

for episode_count in range(max_episodes):                       # play game until max_episodes
    env_info = env.reset(train_mode=False)[brain_name]          # reset the environment    
    states = env_info.vector_observations                        # get the current state (for each agent)
    scores = np.zeros(num_agents)                               # initialize the score (for each agent)

    noise_process.reset()

    step_count = 0
    while True:
        step_count += 1
        # Make a list of local states
        list_of_local_states = [torch.from_numpy(states[i]).float().to(device) for i in range(states.shape[0])]

        # Determine action of each agent and convert to ndarray for env input
        noise = ou_scale * noise_process.get_noise().reshape((1, 4))
        actions = maddpg.act(list_of_local_states, local=True, train_mode=False, add_noise=False)
        actions = [action.detach().cpu().numpy()+noise[:, 2*i:2*i+2] for i, action in enumerate(actions)]
        actions = np.vstack(actions)
        actions = np.clip(actions, -1, 1)                       # all actions between -1 and 1

        env_info = env.step(actions)[brain_name]                # send all actions to tne environment
        next_states = env_info.vector_observations              # get next state (for each agent)
        rewards = env_info.rewards                              # get reward (for each agent)
        dones = env_info.local_done                             # see if episode finished

        replay_buffer.add(states.flatten(),
                          actions.flatten(),
                          np.asarray(rewards),
                          next_states.flatten(),
                          np.asarray(dones))
        
        if len(replay_buffer) > batch_size:
            maddpg.learn(replay_buffer.sample(device))

            if step_count % steps_per_update == 0:
                maddpg.soft_update()

        scores += env_info.rewards                              # update the score (for each agent)
        states = next_states                                    # roll over states to next time step

        if np.any(dones):                                       # exit loop if episode finished
            score_window.append(np.max(scores))
            break
    
    ou_scale *= ou_decay
    print('Episode: {}'.format(episode_count))
    print('Avg score of last 100 episodes: {}'.format(np.mean(score_window)))

# %%
env.close()
# %%
