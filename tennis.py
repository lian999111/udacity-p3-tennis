# %%
from maddpg import MADDPG
from noise import OUNoise

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
from unityagents import UnityEnvironment


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

# %% Create MADDPG agents
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = {}
config["device"] = device
config["state_size"] = 24
config["action_size"] = 2
config["global_state_size"] = 2 * 24
config["global_action_size"] = 2 * 2
config["actor_hidden_sizes"] = [256, 128]
config["critic_hidden_sizes"] = [256, 128]
config["actor_lr"] = 1e-4
config["critic_lr"] = 1e-4
config["weight_decay"] = 0
config["tau"] = 1e-2            # mix ratio of soft update
config["gamma"] = 0.99         # discount factor
config["steps_per_update"] = 1
# For replay memory
config["buffer_size"] = int(1e5)
config["batch_size"] = 128

maddpg = MADDPG(config)

# %% Define the training process
def train_maddpg(n_episodes=3000, goal=0.6):
    # OU noise parameters
    ou_scale = 1.0                    # initial scaling factor
    ou_decay = 0.9995                 # decay of the scaling factor ou_scale
    ou_mu = 0.0                       # asymptotic mean of the noise
    ou_theta = 0.15                   # magnitude of the drift term
    ou_sigma = 0.20                   # magnitude of the diffusion term

    noise_process = OUNoise(2*2, ou_mu, ou_theta, ou_sigma)

    episode_scores = []
    score_window = deque(maxlen=100)
    avg_scores_over_window = []

    for i_episode in range(n_episodes):                       # play game until max_episodes
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
            actions = maddpg.act(list_of_local_states, local=True, train_mode=False)
            actions = [action.detach().cpu().numpy()+noise[:, 2*i:2*i+2] for i, action in enumerate(actions)]
            actions = np.vstack(actions)
            actions = np.clip(actions, -1, 1)                       # all actions between -1 and 1

            env_info = env.step(actions)[brain_name]                # send all actions to tne environment
            next_states = env_info.vector_observations              # get next state (for each agent)
            rewards = env_info.rewards                              # get reward (for each agent)
            dones = env_info.local_done                             # see if episode finished

            maddpg.step(states, actions, rewards, next_states, dones)

            scores += env_info.rewards                              # update the score (for each agent)
            states = next_states                                    # roll over states to next time step

            if np.any(dones):                                       # exit loop if episode finished
                break
        
        ou_scale *= ou_decay

        episode_score = np.max(scores)
        episode_scores.append(episode_score)
        score_window.append(episode_score)
        avg_score_over_window = np.mean(score_window)
        avg_scores_over_window.append(avg_score_over_window)
        print('Episode: {} Score: {}'.format(i_episode, episode_score))
        print('Avg score of last 100 episodes: {}'.format(avg_score_over_window))

        if avg_score_over_window >= goal:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, avg_score_over_window))
            torch.save({"state_dict": maddpg.ddpg_agents[0].local_actor_net.state_dict()}, "result/actor0_checkpoint.pth")
            torch.save({"state_dict": maddpg.ddpg_agents[0].local_critic_net.state_dict()}, "result/critic0_checkpoint.pth")
            torch.save({"state_dict": maddpg.ddpg_agents[1].local_actor_net.state_dict()}, "result/actor1_checkpoint.pth")
            torch.save({"state_dict": maddpg.ddpg_agents[1].local_critic_net.state_dict()}, "result/critic1_checkpoint.pth")
            break

    return episode_scores, avg_scores_over_window

# %% Train MADDPG agents
scores, average_scores = train_maddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='score')
plt.plot(np.arange(len(average_scores)), average_scores, label='avg score')
plt.legend()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('result/score.jpg')

# %% Watch trained agents
maddpg = MADDPG(config)
maddpg.ddpg_agents[0].local_actor_net.load_state_dict(torch.load('result/actor0_checkpoint.pth')['state_dict'])
maddpg.ddpg_agents[0].local_critic_net.load_state_dict(torch.load('result/critic0_checkpoint.pth')['state_dict'])
maddpg.ddpg_agents[1].local_actor_net.load_state_dict(torch.load('result/actor1_checkpoint.pth')['state_dict'])
maddpg.ddpg_agents[1].local_critic_net.load_state_dict(torch.load('result/critic1_checkpoint.pth')['state_dict'])

for i in range(3):
    # Reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations 
    scores = np.zeros(num_agents)
    while True:
        list_of_local_states = [torch.from_numpy(states[i]).float().to(device) for i in range(states.shape[0])]
        actions = maddpg.act(list_of_local_states, local=True, train_mode=False)
        actions = [action.detach().cpu().numpy() for action in actions]
        actions = np.vstack(actions)
        env_info = env.step(actions)[brain_name]                  # send the action to the env
        next_states = env_info.vector_observations                                    # get the next state
        rewards = env_info.rewards                                                    # get the reward
        dones = env_info.local_done                                                   # see if episode has finished
        states = next_states
        scores += rewards
        if np.any(dones):
            break
    
    print('Average Score: {}'.format(np.max(scores)))

# %%
env.close()
# %%
