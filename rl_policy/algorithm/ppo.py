"""The script is heavily borrowed from Stable Baselines3 (SB3) implementation of PPO.
https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
"""
from typing import Any, Optional, Union, Dict, List, Tuple
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.utils import update_learning_rate


class PPO:

    def __init__(
        self,
        policy: nn.Module,
        env: gym.Env,
        val_env: gym.Env,
        n_steps_collect_rollout: int = 2048,
        learning_rate_train: float = 3e-4,
        batch_size_train: int = 64,
        n_epochs_train: int = 10,
        discount_factor: float = 0.99,
        td_lambda: float = 1.0,
        clip_range: float = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        stats_window_size: int = 100,
        n_eval_episodes: int = 5,
        device: str = "cuda",
    ):
        """
        Proximal Policy Optimization algorithm (PPO) (clip version)

        Args:
            policy: a nn.Module class that defines the actor and critic network
            env: a gym.Env instance indicating the environment.  Use vectorized
                environment wrapper to make multiple copies of the environment
                running in parallel
                (see stable_baselines3.common.env_util.make_vec_env)
            n_steps: an integer indicating the number of steps to collect
                rollout in each environment.  Notably, the total number of
                samples collected will be n_steps * n_envs where n_envs is the
                number of environment copies running in parallel.
            learning_rate_train: The learning rate for training the actor and
                critic network
            batch_size_train: an integer indicating the batch size for training
                the actor and critic network
            n_epochs_train: an integer indicating the number of epochs for
                training the actor and critic network
            discount_factor: a float indicating the discount factor for
                computing the return
            td_lambda: an integer indicating the lambda parameter for computing
                the value function target
            clip_range: a float indicating the clipping range for the policy
                loss (Eqn 7 in the PPO paper)
            normalize_advantage: Whether to normalize or not the advantage
            ent_coef: a float indicating the coefficient for the entropy loss
                (the 3rd term in Eqn 9 in the PPO paper)
            vf_coef: a float indicating the coefficient for the value function
                loss (the 2nd term in Eqn 9 in the PPO paper)
            max_grad_norm: a float indicating the maximum value for the gradient
                clipping
            stats_window_size: an integer indicating the window size for the
                rollout logging, specifying the number of episodes to average
                the reported success rate, mean episode length, and mean reward
                over
        """
        self.policy = policy
        self.env = env
        self.val_env = val_env
        self.n_steps_collect_rollout = n_steps_collect_rollout
        self.learning_rate_train = learning_rate_train
        self.batch_size_train = batch_size_train
        self.n_epochs_train = n_epochs_train
        self.discount_factor = discount_factor
        self.td_lambda = td_lambda
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.stats_window_size = stats_window_size
        self.device = device
        self.n_eval_episodes = n_eval_episodes

        self.rollout_buffer = RolloutBuffer(
            self.n_steps_collect_rollout,
            env.observation_space,  # type: ignore[arg-type]
            env.action_space,
            device=self.device,
            gamma=self.discount_factor,
            gae_lambda=self.td_lambda,
            n_envs=env.num_envs,
        )

        self._last_obs = self.env.reset()  # type: ignore[assignment]
        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate_train
        )

        self.num_timesteps = 0
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = deque(maxlen=self._stats_window_size)
        self.ep_success_buffer = deque(maxlen=self._stats_window_size)

    def collect_rollouts(
        self,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        Args:
            env: The training environment
            rollout_buffer: Buffer to fill with rollouts
            n_rollout_steps: Number of experiences to collect per environment
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.eval()

        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device).float()
                values = self.policy.predict_value(obs_tensor)
                actions, log_probs = self.policy.predict_action(
                    obs_tensor, deterministic=False
                )
            actions = actions.cpu().numpy()

            # Otherwise, clip the actions to avoid out of bound error
            # as we are sampling from an unbounded Gaussian distribution
            clipped_actions = np.clip(
                actions, env.action_space.low, env.action_space.high
            )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #33
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = obs_as_tensor(
                        infos[idx]["terminal_observation"], self.device
                    ).float()
                    with torch.no_grad():
                        terminal_value = self.policy.predict_value(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_value(
                obs_as_tensor(new_obs, self.device).float()
            )  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )
    
    def learn(self, total_timesteps: int):
        """Alternate rollout collection and training steps until
        `total_timesteps` are reached.

        Args:
            total_timesteps (int): the total number of interaction steps to
                collect rollouts from the environment
        """
        iteration = 0

        # Test
        self.evaluate_policy(False)

        while self.num_timesteps < total_timesteps:
            self.collect_rollouts(
                self.env,
                self.rollout_buffer,
                n_rollout_steps=self.n_steps_collect_rollout
            )
            iteration += 1

            # Display training infos
            assert self.ep_info_buffer is not None
            self._print_logs(iteration)

            # Train
            self.train()

            # Test
            self.evaluate_policy(False)

        return self

    def _print_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None

        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            print("rollout/ep_rew_mean",
                  safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            print("rollout/ep_rew_std",
                  np.std([ep_info["r"] for ep_info in self.ep_info_buffer]).item())
            print("rollout/ep_len_mean",
                  safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

    def _update_info_buffer(self, infos: List[Dict[str, Any]],
                            dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def _update_learning_rate(
            self,
            optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]
        ) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.learning_rate_train)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.train()

        # Compute current clip range
        clip_range = self.clip_range

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs_train):
            val_losses, policy_losses, entropy_losses = [], [], []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size_train):
                actions = torch.tensor(rollout_data.actions,
                                       dtype=torch.float,
                                       device=self.device).detach()
                obs = torch.tensor(rollout_data.observations,
                                   dtype=torch.float,
                                   device=self.device).detach()

                values = self.policy.predict_value(obs)
                log_prob, entropy = self.policy.evaluate_action(obs, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # TODO: Compute the ratio (pi_theta / pi_theta__old) using the log probability
                # ratio between old and new policy, should be one at the first iteration
                # ratio = ...

                # TODO: Compute the clipped loss
                # clipped surrogate loss
                # policy_loss = ...

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                # TODO: Compute the value loss bwteen the predicted value and returns (rollout_data.returns)
                # No clipping
                values_pred = values
                # Value loss using the TD(gae_lambda) target
                # value_loss = ...
                value_losses.append(value_loss.item())

                # TODO: Compute entropy loss (favor exploration)
                # entropy_loss = ...
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                               self.max_grad_norm)
                self.optimizer.step()

                val_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropy_losses.append(entropy_loss.item())
            print(f"Train, epoch: {epoch}, policy loss: {np.mean(pg_losses)}, "
                  f"value loss: {np.mean(value_losses)}, entropy loss: {np.mean(entropy_losses)}")

    def evaluate_policy(
        self, return_episode_rewards: bool = False,
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        """
        Runs policy for ``n_eval_episodes`` episodes and returns average reward.
        If a vector env is passed in, this divides the episodes to evaluate onto the
        different elements of the vector env. This static division of work is done to
        remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
        details and discussion.

        Args:
            model: a nn.Module indicates the policy network
            env: The gym environment or ``VecEnv`` environment.
            n_eval_episodes: an integer indicates the number of episode to
                evaluate the agent
            deterministic: If True, use deterministic policy
            return_episode_rewards: If True, a list of rewards and episode lengths
                per episode will be returned instead of the mean.
        """
        env = self.val_env
        if not isinstance(self.val_env, VecEnv):
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        n_envs = env.num_envs
        episode_rewards = []
        episode_lengths = []

        episode_counts = np.zeros(n_envs, dtype="int")
        # Divides episodes among different sub environments in the vector as evenly as possible
        episode_count_targets = np.array(
            [(self.n_eval_episodes + i) // n_envs for i in range(n_envs)],
            dtype="int"
        )

        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")
        observations = env.reset()
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        while (episode_counts < episode_count_targets).any():
            actions, _, _ = self.policy.forward(
                obs_as_tensor(observations, self.device).float(),
                deterministic=True,
            )
            actions = actions.data.cpu().numpy()
            new_observations, rewards, dones, _ = env.step(actions)
            current_rewards += rewards
            current_lengths += 1
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    # unpack values so that the callback can access the local variables
                    done = dones[i]
                    episode_starts[i] = done

                    if dones[i]:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                        current_rewards[i] = 0
                        current_lengths[i] = 0

            observations = new_observations

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        if return_episode_rewards:
            return episode_rewards, episode_lengths
        
        print(f"Test, mean reward: {mean_reward}, std reward: {std_reward}")
        return mean_reward, std_reward
