if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib
import copy
import random

import hydra
import torch
from omegaconf import OmegaConf
import numpy as np
import gym
# from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import wandb.sdk.data_types.video as wv

from rl_policy.workspace.base_workspace import BaseWorkspace
from rl_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from rl_policy.policy.lowdim_policy import LowDimPolicy
from rl_policy.algorithm.ppo import PPO


OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainRLLowDimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        def env_fn() -> gym.Env:
            rs = np.random.RandomState(seed=cfg.task.env_meta.seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
            env = hydra.utils.instantiate(cfg.task.env, reset_to_state=state)
            # env = VideoRecordingWrapper(
            #     hydra.utils.instantiate(cfg.task.env, reset_to_state=state),
            #     video_recoder=VideoRecorder.create_h264(
            #         fps=10,
            #         codec='h264',
            #         input_pix_fmt='rgb24',
            #         crf=22,
            #         thread_type='FRAME',
            #         thread_count=1
            #     ),
            #     file_path=os.path.join("./outputs/video",  "train_" + wv.util.generate_id() + ".mp4"),
            #     steps_per_render=1
            # )
            return env

        def eval_env_fn() -> gym.Env:
            rs = np.random.RandomState(seed=cfg.task.env_meta.seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
            env = VideoRecordingWrapper(
                hydra.utils.instantiate(cfg.task.env, reset_to_state=state),
                video_recoder=VideoRecorder.create_h264(
                    fps=10,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=22,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=os.path.join("./outputs/video",  "val_" + wv.util.generate_id() + ".mp4"),
                steps_per_render=1
            )
            return env


        env = make_vec_env(env_fn, n_envs=cfg.task.env_meta.n_train)
        val_env = make_vec_env(eval_env_fn, n_envs=cfg.task.env_meta.n_val)
        env.reset()
        val_env.reset()
        policy = LowDimPolicy(obs_dim=env.observation_space.shape[0],
                              action_space=env.action_space,
                              log_std_init=cfg.policy.log_std_init).cuda()
        ppo = PPO(policy=policy,
                  env=env,
                  val_env=val_env,
                  n_steps_collect_rollout=cfg.training.n_steps_collect_rollout,
                  learning_rate_train=cfg.training.lr,
                  batch_size_train=cfg.training.batch_size,
                  n_epochs_train=cfg.training.n_epochs_per_rollout,
                  discount_factor=cfg.training.discount_factor,
                  td_lambda=cfg.training.td_lambda,
                  clip_range=cfg.training.clip_range,
                  normalize_advantage=cfg.training.normalize_advantage,
                  ent_coef=cfg.training.ent_coef,
                  vf_coef=cfg.training.vf_coef,
                  max_grad_norm=cfg.training.max_grad_norm,
                  stats_window_size=cfg.training.stats_window_size,
                  n_eval_episodes=cfg.testing.n_eval_episodes,
                  device=cfg.training.device)
        ppo.learn(12_000_000)

        # policy_kwargs = dict(
        #     log_std_init=3,
        # )
        # model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
        #             tensorboard_log="/tmp/outputs",
        #             n_steps=300, n_epochs=20, ent_coef=0.0)
        # eval_callback = EvalCallback(eval_env=val_env, best_model_save_path="/tmp/outputs",
        #                              log_path="/tmp/outputs", eval_freq=6000)
        # model.learn(total_timesteps=12_000_000, callback=eval_callback)



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainRLLowDimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

