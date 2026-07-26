from __future__ import annotations  

import argparse  
import os  
import sys  

if __package__ is None or __package__ == "":  
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  

from settings import PPOConfig  
from training import add_common_args, base_train_config_from_args, train_ppo  


def parse_args():
    parser = argparse.ArgumentParser(description="PPO training for half-hull latent-variable optimization")  
    add_common_args(parser)  
    parser.add_argument("--rollout_steps", type=int, default=2048)  
    parser.add_argument("--ppo_epochs", type=int, default=10)  
    parser.add_argument("--minibatch_size", type=int, default=256)  
    parser.add_argument("--clip_ratio", type=float, default=0.20)  
    parser.add_argument("--gae_lambda", type=float, default=0.95)  
    parser.add_argument("--value_coef", type=float, default=0.50)  
    parser.add_argument("--entropy_coef", type=float, default=0.00)  
    parser.add_argument("--max_grad_norm", type=float, default=0.50)  
    parser.add_argument("--actor_lr", type=float, default=3e-4)  
    parser.add_argument("--critic_lr", type=float, default=3e-4)  
    parser.add_argument("--gamma", type=float, default=0.99)  
    return parser.parse_args()  


def main():
    args = parse_args()  
    cfg = base_train_config_from_args(args, PPOConfig)  
    cfg.rollout_steps = args.rollout_steps  
    cfg.ppo_epochs = args.ppo_epochs  
    cfg.minibatch_size = args.minibatch_size  
    cfg.clip_ratio = args.clip_ratio  
    cfg.gae_lambda = args.gae_lambda  
    cfg.value_coef = args.value_coef  
    cfg.entropy_coef = args.entropy_coef  
    cfg.max_grad_norm = args.max_grad_norm  
    cfg.actor_lr = args.actor_lr  
    cfg.critic_lr = args.critic_lr  
    cfg.gamma = args.gamma  
    if args.output_dir == "outputs/run":  
        args.output_dir = "outputs/ppo"  
    train_ppo(args, cfg)  


if __name__ == "__main__":  
    main()  
