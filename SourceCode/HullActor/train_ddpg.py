from __future__ import annotations  

import argparse  
import os  
import sys  

if __package__ is None or __package__ == "":  
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  

from settings import DDPGConfig  
from training import add_common_args, base_train_config_from_args, train_offpolicy  


def parse_args():
    parser = argparse.ArgumentParser(description="DDPG training for half-hull latent-variable optimization")  
    add_common_args(parser)  
    parser.add_argument("--exploration_noise", type=float, default=0.10)  
    parser.add_argument("--actor_lr", type=float, default=3e-4)  
    parser.add_argument("--critic_lr", type=float, default=3e-4)  
    parser.add_argument("--gamma", type=float, default=0.99)  
    parser.add_argument("--tau", type=float, default=0.005)  
    return parser.parse_args()  


def main():
    """DDPG 训练入口。"""
    args = parse_args()  
    cfg = base_train_config_from_args(args, DDPGConfig)  
    cfg.exploration_noise = args.exploration_noise  
    cfg.actor_lr = args.actor_lr  
    cfg.critic_lr = args.critic_lr  
    cfg.gamma = args.gamma  
    cfg.tau = args.tau  
    if args.output_dir == "outputs/run":  
        args.output_dir = "outputs/ddpg"  
    train_offpolicy("DDPG", args, cfg)  


if __name__ == "__main__":  
    main()  
