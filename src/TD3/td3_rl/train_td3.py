#!/usr/bin/env python3
import os
import threading
import time

import numpy as np
import rclpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from td3_rl.gazebo_env import GazeboEnv
from td3_rl.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        td3_rl_path = get_package_share_directory('td3_rl')
        
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter(log_dir=os.path.join(td3_rl_path, "runs"))
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -np.inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{filename}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{filename}_critic.pth"))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic.pth")))


class TD3Trainer(Node):
    def __init__(self):
        super().__init__('td3_trainer')
        self.declare_params()
        self.initialize_environment()
        self.train_loop()

    def declare_params(self):
        self.declare_parameter('seed', 0)
        self.declare_parameter('eval_freq', 500)
        self.declare_parameter('max_ep', 150)
        self.declare_parameter('eval_ep', 10)
        self.declare_parameter('max_timesteps', 5000)
        self.declare_parameter('expl_noise', 1.0)
        self.declare_parameter('expl_decay_steps', 500000)
        self.declare_parameter('expl_min', 0.1)
        self.declare_parameter('batch_size', 128)
        self.declare_parameter('discount', 0.99999)
        self.declare_parameter('tau', 0.005)
        self.declare_parameter('policy_noise', 0.2)
        self.declare_parameter('noise_clip', 0.5)
        self.declare_parameter('policy_freq', 2)
        self.declare_parameter('buffer_size', 1000000)
        self.declare_parameter('file_name', 'TD3_Turtlebot')
        self.declare_parameter('save_model', True)
        self.declare_parameter('load_model', False)
        self.declare_parameter('random_near_obstacle', True)
        self.declare_parameter('environment_dim', 20)
    
    def print_parameters(self):
        self.get_logger().info("Loaded Parameters:")
        for param in self._parameters.keys():
            self.get_logger().info(f"{param}: {self.get_parameter(param).value}")

    def initialize_environment(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = self.get_parameter('seed').value
        self.eval_freq = self.get_parameter('eval_freq').value
        self.max_ep = self.get_parameter('max_ep').value
        self.eval_ep = self.get_parameter('eval_ep').value
        self.max_timesteps = self.get_parameter('max_timesteps').value
        self.expl_noise = self.get_parameter('expl_noise').value
        self.expl_decay_steps = self.get_parameter('expl_decay_steps').value
        self.expl_min = self.get_parameter('expl_min').value
        self.batch_size = self.get_parameter('batch_size').value
        self.discount = self.get_parameter('discount').value
        self.tau = self.get_parameter('tau').value
        self.policy_noise = self.get_parameter('policy_noise').value
        self.noise_clip = self.get_parameter('noise_clip').value
        self.policy_freq = self.get_parameter('policy_freq').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.file_name = self.get_parameter('file_name').value
        self.save_model = self.get_parameter('save_model').value
        self.load_model = self.get_parameter('load_model').value
        self.random_near_obstacle = self.get_parameter('random_near_obstacle').value
        self.environment_dim = self.get_parameter('environment_dim').value
        
        self.print_parameters()

        td3_rl_path = get_package_share_directory('td3_rl')
        
        self.results_path = os.path.join(td3_rl_path, "results")
        self.models_path = os.path.join(td3_rl_path, "pytorch_models")
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

        self.env = GazeboEnv(self.environment_dim)
        # Create and spin GazeboEnv in a separate thread
        executor = MultiThreadedExecutor()
        executor.add_node(self.env)

        # Run the environment in a separate thread so that its subscriptions work
        env_thread = threading.Thread(target=executor.spin, daemon=True)
        env_thread.start()
        
        time.sleep(5)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.state_dim = self.environment_dim + 4
        self.action_dim = 2
        self.max_action = 1

        self.network = TD3(self.state_dim, self.action_dim, self.max_action)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.seed)
        
        if self.load_model:
            try:
                self.network.load(self.file_name, self.models_path)
            except:
                self.get_logger().warn("Could not load model, initializing training with random parameters")

    def train_loop(self):
        evaluations = []
        timestep, timesteps_since_eval, episode_num = 0, 0, 0
        done = True
        epoch = 1
        last_time = time.time()
        
        count_rand_actions = 0
        episode_timesteps = 0
        random_action = []
        last_time = time.time()

        while timestep < self.max_timesteps:
            if timestep % 100 == 0:
                current_time = time.time()
                self.get_logger().info(f"{timestep} timesteps. Last 100 timesteps finished in {(current_time - last_time):.2f} seconds")
                last_time = current_time
            
            if done:
                if timestep != 0:
                    self.network.train(
                        self.replay_buffer,
                        episode_timesteps,
                        self.batch_size,
                        self.discount,
                        self.tau,
                        self.policy_noise,
                        self.noise_clip,
                        self.policy_freq,
                    )
                if timesteps_since_eval >= self.eval_freq:
                    self.get_logger().info(f"Validating at timestep {timestep}")
                    timesteps_since_eval %= self.eval_freq
                    evaluations.append(
                        self.evaluate(network=self.network, epoch=epoch, eval_episodes=self.eval_ep)
                    )
                    if self.save_model:
                        self.network.save(self.file_name, directory=self.models_path)
                        np.save(os.path.join(self.results_path, f"{self.file_name}.npy"), evaluations)
                    epoch += 1
                
                state = self.env.reset()
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
            
            # add some exploration noise
            if self.expl_noise > self.expl_min:
                self.expl_noise = self.expl_noise - ((1 - self.expl_min) / self.expl_decay_steps)
            
            action = self.network.get_action(np.array(state))
            action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip(
                -self.max_action, self.max_action)
            
            # If the robot is facing an obstacle, randomly force it to take a consistent random action.
            # This is done to increase exploration in situations near obstacles.
            # Training can also be performed without it
            if self.random_near_obstacle:
                if (
                    np.random.uniform(0, 1) > 0.85
                    and min(state[4:-8]) < 0.6
                    and count_rand_actions < 1
                ):
                    count_rand_actions = np.random.randint(8, 15)
                    random_action = np.random.uniform(-1, 1, 2)

                if count_rand_actions > 0:
                    count_rand_actions -= 1
                    action = random_action
                    action[0] = -1
            
            # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, _ = self.env.step(a_in)
            done_bool = 0 if episode_timesteps + 1 == self.max_ep else int(done)
            done = 1 if episode_timesteps + 1 == self.max_ep else int(done)
            episode_reward += reward
            
            # Save the tuple in replay buffer
            self.replay_buffer.add(state, action, reward, done_bool, next_state)

            state = next_state
            episode_timesteps += 1
            timestep += 1
            timesteps_since_eval += 1

        # After the training is done, evaluate the network and save it
        evaluations.append(self.evaluate(network=self.network, epoch=epoch, eval_episodes=self.eval_ep))
        if self.save_model:
            self.network.save(self.file_name, directory=self.models_path)
            np.save(os.path.join(self.results_path, f"{self.file_name}.npy"), evaluations)

    def evaluate(self, network, epoch, eval_episodes=10):
        avg_reward = 0.0
        col = 0
        for _ in range(eval_episodes):
            count = 0
            state = self.env.reset()
            done = False
            while not done and count < self.max_ep:
                action = network.get_action(np.array(state))
                a_in = [(action[0] + 1) / 2, action[1]]
                state, reward, done, _ = self.env.step(a_in)
                avg_reward += reward
                count += 1
                if reward < -90:
                    col += 1
        avg_reward /= eval_episodes
        avg_col = col / eval_episodes
        self.get_logger().info("..............................................")
        self.get_logger().info(
            f"Average Reward over {eval_episodes} Evaluation Episodes, Epoch {epoch}: {avg_reward}, {avg_col}"
        )
        self.get_logger().info("..............................................")
        return avg_reward


def main(args=None):
    rclpy.init(args=args)
    node = TD3Trainer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
