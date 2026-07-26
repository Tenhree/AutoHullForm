from __future__ import annotations  

import argparse  
import os  
import sys  

if __package__ is None or __package__ == "":  
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  

from settings import SACConfig  
from training import add_common_args, base_train_config_from_args, train_offpolicy  


def parse_args():
    parser = argparse.ArgumentParser(description="SAC training for half-hull latent-variable optimization")  
    add_common_args(parser)  
    parser.add_argument("--actor_lr", type=float, default=3e-4)  
    parser.add_argument("--critic_lr", type=float, default=3e-4)  
    parser.add_argument("--alpha_lr", type=float, default=3e-4)  
    parser.add_argument("--init_alpha", type=float, default=0.20)  
    parser.add_argument("--target_entropy", type=float, default=-3.0)  
    parser.add_argument("--fixed_alpha", action="store_true", help="关闭自动温度调节")  
    parser.add_argument("--gamma", type=float, default=0.99)  
    parser.add_argument("--tau", type=float, default=0.005)  
    return parser.parse_args()  


def main():
    args = parse_args()  
    cfg = base_train_config_from_args(args, SACConfig)  
    cfg.actor_lr = args.actor_lr  
    cfg.critic_lr = args.critic_lr  
    cfg.alpha_lr = args.alpha_lr  
    cfg.init_alpha = args.init_alpha  
    cfg.target_entropy = args.target_entropy  
    cfg.automatic_entropy_tuning = not args.fixed_alpha  
    cfg.gamma = args.gamma  
    cfg.tau = args.tau  
    if args.output_dir == "outputs/run":  
        args.output_dir = "outputs/sac"  
    train_offpolicy("SAC", args, cfg)  


if __name__ == "__main__":  
    main()  
