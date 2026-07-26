from __future__ import annotations  

import argparse  
import os  
import sys  

if __package__ is None or __package__ == "":  
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  

from settings import TD3Config  
from training import add_common_args, base_train_config_from_args, train_offpolicy  


def parse_args():
    parser = argparse.ArgumentParser(description="TD3 training for half-hull latent-variable optimization")  
    add_common_args(parser)  
    parser.add_argument("--exploration_noise", type=float, default=0.10)  
    parser.add_argument("--policy_noise", type=float, default=0.20)  
    parser.add_argument("--noise_clip", type=float, default=0.50)  
    parser.add_argument("--policy_delay", type=int, default=2)  
    parser.add_argument("--actor_lr", type=float, default=3e-4)  
    parser.add_argument("--critic_lr", type=float, default=3e-4)  
    parser.add_argument("--gamma", type=float, default=0.99)  
    parser.add_argument("--tau", type=float, default=0.005)  
    return parser.parse_args()  


def main():
    args = parse_args()  
    cfg = base_train_config_from_args(args, TD3Config)  
    cfg.exploration_noise = args.exploration_noise  
    cfg.policy_noise = args.policy_noise  
    cfg.noise_clip = args.noise_clip  
    cfg.policy_delay = args.policy_delay  
    cfg.actor_lr = args.actor_lr  
    cfg.critic_lr = args.critic_lr  
    cfg.gamma = args.gamma  
    cfg.tau = args.tau  
    if args.output_dir == "outputs/run":  
        args.output_dir = "outputs/td3"  
    train_offpolicy("TD3", args, cfg)  


if __name__ == "__main__":  
    main()  
