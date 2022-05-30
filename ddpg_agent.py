from model import ActorNet, CriticNet

import torch
import torch.nn.functional as F

class DDPGAgent:
    """DDPG agent with Gaussian sampling on the action"""

    def __init__(self, state_size: int, action_size: int, global_state_size: int, global_action_size: int, config: dict, seed: int=0):
        """Initialize actor critic network
        Params
        ======
            state_size: dimension of each state
            action_size: dimension of each action
            global_state_size: dimension of global state
            global_state_size: dimension of global action
            config: dict of configuration
        """

        self.local_actor_net = ActorNet(
            state_size, action_size, config["actor_hidden_sizes"], seed=seed).to(config["device"])
        self.local_critic_net = CriticNet(global_state_size, global_action_size,
                                          config["critic_hidden_sizes"], seed=seed).to(config["device"])  # note the input size (Q network)
        self.target_actor_net = ActorNet(
            state_size, action_size, config["actor_hidden_sizes"], seed=seed).to(config["device"])
        self.target_critic_net = CriticNet(global_state_size, global_action_size,
                                           config["critic_hidden_sizes"], seed=seed).to(config["device"])  # note the input size (Q network)

        self.actor_optimizer = torch.optim.Adam(
            self.local_actor_net.parameters(), lr=config["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(
            self.local_critic_net.parameters(), lr=config["critic_lr"], weight_decay=config["weight_decay"])

        self.state_size = state_size
        self.action_size = action_size
        self.global_state_size = global_state_size
        self.global_action_size = global_action_size
        self.seed = seed

        self.config = config
    
    def act(self, state: torch.Tensor, local: bool=False, train_mode: bool=False):
        """Determine action given current state
        Params
        ======
            state: current local state
            local: whether to use local network or target network
            train_mode: run actor net in train mode or not
        """
        actor_net = self.local_actor_net if local else self.target_actor_net

        if not train_mode:
            actor_net.eval()

        with torch.set_grad_enabled(train_mode):
            action = actor_net(state)

        if not train_mode:
            actor_net.train()

        return action

    def soft_update(self):
        """Performs soft update from local to target networks"""
        tau = self.config["tau"]
        for local_param, target_param in zip(self.local_actor_net.parameters(), self.target_actor_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        for local_param, target_param in zip(self.local_critic_net.parameters(), self.target_critic_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        


