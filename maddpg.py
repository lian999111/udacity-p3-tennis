from ddpg_agent import DDPGAgent
from replay_buffer import ReplayBuffer

import numpy as np
import torch
import torch.nn.functional as F

class MADDPG:
    """Wrapper around DDPGAgents to form a multi-agent setting"""
    
    def __init__(self, config: dict):
        self.config = config

        self.ddpg_agents = [DDPGAgent(state_size=config["state_size"],
                                      action_size=config["action_size"],
                                      global_state_size=config["global_state_size"],
                                      global_action_size=config["global_action_size"],
                                      config=config,
                                      seed=1),
                            DDPGAgent(state_size=config["state_size"],
                                      action_size=config["action_size"],
                                      global_state_size=config["global_state_size"],
                                      global_action_size=config["global_action_size"],
                                      config=config,
                                      seed=1)]

        self.memory = ReplayBuffer(config["buffer_size"], config["batch_size"])
        self.t_step = 0

    def act(self, states_of_all_agents: list, local: bool=False, train_mode: bool=False):
        """Determine actions given current states of all agents
        Params
        ======
            states_of_all_agents: list of current local states for all agents
            local: whether to use local network or target network
            train_mode: run actor net in train mode or not
        """
        actions = [agent.act(state, local, train_mode) for agent, state in zip(self.ddpg_agents, states_of_all_agents)]
        return actions

    def step(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray, dones: np.ndarray):
        """One step of training process
        Params
        =====
            states: current states of agents
            actions: current actions of agents
            rewards: immediate rewards of agents
            next_states: next states of agents
            dones: flags of dones of agents
        """
        self.t_step += 1

        # All data are flattened into 1-D array
        self.memory.add(states.flatten(),
                        actions.flatten(),
                        np.asarray(rewards),
                        next_states.flatten(),
                        np.asarray(dones))

        if self.t_step % self.config["steps_per_update"] == 0 and len(self.memory) > self.memory.batch_size:
            self.learn(self.memory.sample())
            self.soft_update()
        
    def learn(self, experiences: tuple):
        """Train MADDPG agents using replays
        Params
        ======
            experiences: tuple containing numpy arrays (states, actions, rewards, next_states, dones)
        """

        state_size = self.config["state_size"]
        device = self.config["device"]
        gamma = self.config["gamma"]
        num_agents = len(self.ddpg_agents)

        # Note that these are batched
        full_states, full_actions, rewards, full_next_states, dones = map(lambda x: torch.from_numpy(x).float().to(device),experiences)

        ############### Train critic using TD-error ###############
        ### Q-target ###
        local_next_states_of_agents = [full_next_states[:, i*state_size: (i+1)*state_size]
                                       for i in range(num_agents)]
        with torch.no_grad():
            # Get next actions using the target actor network
            next_actions_of_agents = self.act(
                local_next_states_of_agents, local=False, train_mode=False)

            # Concatenate actions of agents to make full action
            full_next_actions = torch.cat(next_actions_of_agents, dim=1)

            # The seemingly unnecessary i:i+1 keeps the dimension after slicing
            q_targets = [rewards[:, i:i+1] + (1 - dones[:, i:i+1]) * gamma * agent.target_critic_net(full_next_states, full_next_actions)
                         for i, agent in enumerate(self.ddpg_agents)]

        ### Q-value ###
        q_values = [agent.local_critic_net(full_states, full_actions)
                    for agent in self.ddpg_agents]

        for i in range(num_agents):
            agent = self.ddpg_agents[i]
            critic_loss = F.smooth_l1_loss(q_values[i], q_targets[i].detach())

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

        ############### Train actor ###############
        local_states_of_agents = [full_states[:, i*state_size: (i+1)*state_size]
                                  for i in range(num_agents)]

        actions_of_agents = self.act(local_states_of_agents, local=True, train_mode=True)

        # Loop over agents and do updates to their actor nets
        for i in range(num_agents):
            agent = self.ddpg_agents[i]
            # Start populating critic input with actions 
            full_actions = [actions if j == i
                            else actions.detach()
                            for j, actions in enumerate(actions_of_agents)]
            full_actions = torch.cat(full_actions, dim=1)
            actor_loss = - agent.local_critic_net(full_states, full_actions).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

    def soft_update(self):
        """Perform soft update to each agent"""
        [agent.soft_update() for agent in self.ddpg_agents]