from ddpg_agent import DDPGAgent

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

    def act(self, states_of_all_agents: list, local: bool=False, train_mode: bool=False, add_noise: bool=False):
        """Determine actions given current states of all agents
        Params
        ======
            states_of_all_agents: list of current local states for all agents
            local: whether to use local network or target network
            train_mode: run actor net in train mode or not
        """
        actions = [agent.act(state, local, train_mode, add_noise) for agent, state in zip(self.ddpg_agents, states_of_all_agents)]
        return actions
        
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

        ############### Critic ###############
        ### Next Q-value ###
        local_next_states_of_agents = [full_next_states[:, i*state_size: (i+1)*state_size]
                                       for i in range(num_agents)]
        with torch.no_grad():
            # using target actor net for dualing purpose
            next_actions_of_agents = self.act(
                local_next_states_of_agents, local=False, train_mode=False, add_noise=False)

            # Concatenate actions of agents to make full action
            full_next_actions = torch.cat(next_actions_of_agents, dim=1)
            # Concatenate full states and full actions to get critic input
            next_critic_input = torch.cat(
                (full_next_states, full_next_actions), dim=1)

            # The seemingly unnecessary i:i+1 keeps the dimension after slicing
            q_targets = [rewards[:, i:i+1] + (1 - dones[:, i:i+1]) * gamma * agent.target_critic_net(next_critic_input)
                         for i, agent in enumerate(self.ddpg_agents)]
            # # Each agent has column (batch) of q values in this tensor
            # q_targets = torch.cat(q_targets, dim=1).detach()

        ### Q-value ###
        critic_input = torch.cat((full_states, full_actions), dim=1)
        q_values = [agent.local_critic_net(critic_input)
                    for agent in self.ddpg_agents]
        # q_values = torch.cat(q_values, dim=1)

        for i in range(num_agents):
            agent = self.ddpg_agents[i]
            agent.critic_optimizer.zero_grad()
            critic_loss = F.mse_loss(q_values[i], q_targets[i].detach())
            critic_loss.backward()
            # print('local critic grad: {}'.format(agent.local_critic_net.linear_layers[-1].weight.grad))
            # print('target critic grad: {}'.format(agent.target_critic_net.linear_layers[-1].weight.grad))
            agent.critic_optimizer.step()

        # [agent.critic_optimizer.zero_grad() for agent in self.ddpg_agents]
        # critic_loss = F.mse_loss(q_values, q_targets)
        # critic_loss.backward()
        # [agent.critic_optimizer.step() for agent in self.ddpg_agents]

        ############### Actor ###############
        local_states_of_agents = [full_states[:, i*state_size: (i+1)*state_size]
                                  for i in range(num_agents)]

        actions_of_agents = self.act(local_states_of_agents, local=True, train_mode=True, add_noise=False)
        # critic_input = torch.cat(actions_of_agents, dim=1)
        # # Add full states to critic input
        # critic_input = torch.cat((full_states, critic_input), dim=1)

        # Loop over agents and do updates to their actor nets
        for i in range(num_agents):
            agent = self.ddpg_agents[i]
            agent.actor_optimizer.zero_grad()
            # Start populating critic input with actions 
            actions_of_agents = [actions if j == i
                                 else actions.detach()
                                 for j, actions in enumerate(actions_of_agents)]
            critic_input = torch.cat(actions_of_agents, dim=1)
            # Add full states to critic input
            critic_input = torch.cat((full_states, critic_input), dim=1)
            actor_loss = -agent.local_critic_net(critic_input).mean()
            actor_loss.backward(retain_graph=False)
            # print('actor grad: {}'.format(agent.local_actor_net.linear_layers[-1].weight.grad))
            # print('actor grad: {}'.format(agent.local_actor_net.std.grad))
            # print(agent.local_actor_net.std)
            # for param in agent.local_actor_net.parameters():
            #     print(param)
            agent.actor_optimizer.step()

            # print('agent {} actor_loss: {}'.format(i, actor_loss.cpu().detach().item()))

            
        
        

        # critic_input = [agent.local_actor_net(local_states)
        #                 for agent, local_states in zip(self.ddpg_agents, local_states_of_agents)]
        # critic_input = torch.cat(critic_input, dim=1)
        # # critic_input = torch.cat(actions_of_agents, dim=1)
        # # Add full states to critic input
        # critic_input = torch.cat((full_states, critic_input), dim=1)
        

        # actor_loss = [-agent.local_critic_net(critic_input)
        #               for agent in self.ddpg_agents]
        # actor_loss = torch.cat(actor_loss, dim=1).mean()

        # [agent.actor_optimizer.zero_grad() for agent in self.ddpg_agents]
        # actor_loss.backward(retain_graph=False)
        # [agent.actor_optimizer.step() for agent in self.ddpg_agents]

    def soft_update(self):
        """Perform soft update to each agent"""
        [agent.soft_update() for agent in self.ddpg_agents]