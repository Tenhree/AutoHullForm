from __future__ import annotations


import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from agents import DDPGAgent, SACAgent, TD3Agent
from cases import load_cases_csv
from env import HullOptimizationEnv
from hull_interfaces import make_generator, make_resistance_predictor
from ppo_agent import PPOAgent
from replay_buffer import ReplayBuffer
from settings import DDPGConfig, EnvConfig, PPOConfig, SACConfig, TD3Config
from utils import CSVLogger, ensure_dir, save_json, set_seed, resolve_device


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:

    parser.add_argument("--code1_dir", type=str, default="./Generator/", )#Modify to the corresponding path.

    parser.add_argument("--code2_dir", type=str, default="./Predictor/")#Modify to the corresponding path.

    parser.add_argument("--code2_train_file", type=str, default="./Predictor/Hull2Hydro.py")#Modify to the corresponding path.

    parser.add_argument("--generator_ckpt", type=str, default="./Generator/Model/stage2_final.pth")#Modify to the corresponding path.

    parser.add_argument("--resistance_ckpt", type=str, default="./Predictor/Model/prediction_best.pth")#Modify to the corresponding path.

    parser.add_argument("--output_dir", type=str, default="outputs/run",)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--amp", action="store_true",)

    parser.add_argument("--mock", action="store_true",)

    parser.add_argument("--seed", type=int, default=970709)

    parser.add_argument("--total_steps", type=int, default=5000000)

    parser.add_argument("--start_steps", type=int, default=250000)

    parser.add_argument("--update_after", type=int, default=1000)

    parser.add_argument("--update_every", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--replay_size", type=int, default=200000)

    parser.add_argument("--episode_len", type=int, default=10)

    parser.add_argument("--action_scale", type=float, default=0.10)

    parser.add_argument("--eps_volume", type=float, default=0.02)

    parser.add_argument("--lambda_volume", type=float, default=50.0)

    parser.add_argument("--lambda_geometry", type=float, default=1.0)

    parser.add_argument("--lambda_resistance", type=float, default=50.0)

    parser.add_argument("--geometry_penalty_threshold", type=float, default=0.10)

    parser.add_argument("--case_csv", type=str, default="",)

    parser.add_argument("--eval_case_csv", type=str, default="", )

    parser.add_argument("--eval_interval", type=int, default=5000)

    parser.add_argument("--eval_episodes", type=int, default=8)

    parser.add_argument("--log_steps", action="store_true", )

    parser.add_argument("--save_best_pointclouds", action="store_true")

    parser.add_argument("--torch_num_threads", type=int, default=1, )

    parser.add_argument("--save_interval", type=int, default=100000)

    return parser


def env_config_from_args(args) -> EnvConfig:



    return EnvConfig(
        action_scale=float(args.action_scale),
        episode_len=int(args.episode_len),
        eps_volume=float(args.eps_volume),
        lambda_volume=float(args.lambda_volume),
        lambda_geometry=float(args.lambda_geometry),
        lambda_resistance=float(args.lambda_resistance),
        geometry_penalty_threshold=float(args.geometry_penalty_threshold),
    )


def base_train_config_from_args(args, cls):



    cfg = cls()
    
    cfg.seed = int(args.seed)
    
    cfg.total_steps = int(args.total_steps)
    
    cfg.start_steps = int(args.start_steps)
    
    cfg.update_after = int(args.update_after)
    
    cfg.update_every = int(args.update_every)
    
    cfg.batch_size = int(args.batch_size)
    
    cfg.eval_interval = int(args.eval_interval)
    
    cfg.eval_episodes = int(args.eval_episodes)
    
    cfg.log_steps = bool(args.log_steps)
    
    cfg.save_best_pointclouds = bool(args.save_best_pointclouds)
    
    return cfg


def make_envs(args, env_cfg: EnvConfig):


    

    
    generator = make_generator(args.code1_dir, args.generator_ckpt, device=args.device, mock=args.mock)
    
    predictor = make_resistance_predictor(
        args.code2_dir,
        args.resistance_ckpt,
        device=args.device,
        mock=args.mock,
        code2_train_file=args.code2_train_file or None,
        amp=args.amp,
    )
    
    train_cases = load_cases_csv(args.case_csv) if args.case_csv else None
    
    eval_cases = load_cases_csv(args.eval_case_csv) if args.eval_case_csv else None
    
    train_env = HullOptimizationEnv(generator, predictor, cfg=env_cfg, seed=args.seed, cases=train_cases)
    
    eval_env = HullOptimizationEnv(generator, predictor, cfg=env_cfg, seed=args.seed + 999, cases=eval_cases)
    
    return train_env, eval_env, eval_cases


def step_log_row(algo: str, global_step: int, episode: int, reward: float, info: Dict) -> Dict:


    
    return {
        "algo": algo,
        "global_step": global_step,
        "episode": episode,
        "t": info.get("t", ""),
        "reward": float(reward),
        "reward_resistance": info.get("reward_resistance", ""),
        "penalty_volume": info.get("penalty_volume", ""),
        "penalty_geometry": info.get("penalty_geometry", ""),
        "L": info.get("L", ""),
        "B": info.get("B", ""),
        "D": info.get("D", ""),
        "T": info.get("T", ""),
        "CB": info.get("CB", ""),
        "V": info.get("V", ""),
        "z0_1": info.get("z0_1", ""),
        "z0_2": info.get("z0_2", ""),
        "z0_3": info.get("z0_3", ""),
        "z1": info.get("z1", ""),
        "z2": info.get("z2", ""),
        "z3": info.get("z3", ""),
        "action_1": info.get("action_1", ""),
        "action_2": info.get("action_2", ""),
        "action_3": info.get("action_3", ""),
        "cb_estimated": info.get("cb_estimated", ""),
        "resistance": info.get("resistance", ""),
        "R": info.get("R", ""),
        "initial_resistance": info.get("initial_resistance", ""),
        "resistance_ratio": info.get("resistance_ratio", ""),
        "resistance_reduction": info.get("resistance_reduction", ""),
        "volume_actual": info.get("volume_actual", ""),
        "volume_target": info.get("volume_target", ""),
        "volume_error": info.get("volume_error", ""),
        "abs_volume_error": info.get("abs_volume_error", ""),
        "geometry_penalty": info.get("geometry_penalty", ""),
        "constraint_ok": info.get("constraint_ok", ""),
        "is_half_hull": info.get("is_half_hull", ""),
        "best_resistance": info.get("best_resistance", ""),
        "best_resistance_reduction": info.get("best_resistance_reduction", ""),
        "best_volume_error": info.get("best_volume_error", ""),
        "best_abs_volume_error": info.get("best_abs_volume_error", ""),
        "best_constraint_ok": info.get("best_constraint_ok", ""),
        "error": info.get("error", ""),
    }


def episode_log_row(algo: str, global_step: int, episode: int, summary: Dict, wall_time: float, point_cloud_path: str = "") -> Dict:


    
    return {
        "algo": algo,
        "global_step": global_step,
        "episode": episode,
        "episode_len": summary.get("episode_len", ""),
        "episode_return": summary.get("episode_return", ""),
        "reward_mean": summary.get("reward_mean", ""),
        "reward_std": summary.get("reward_std", ""),
        "initial_resistance": summary.get("initial_resistance", ""),
        "initial_cb_estimated": summary.get("initial_cb_estimated", ""),
        "final_resistance": summary.get("final_resistance", ""),
        "final_cb_estimated": summary.get("final_cb_estimated", ""),
        "best_resistance": summary.get("best_resistance", ""),
        "best_cb_estimated": summary.get("best_cb_estimated", ""),
        "final_resistance_reduction": summary.get("final_resistance_reduction", ""),
        "best_resistance_reduction": summary.get("best_resistance_reduction", ""),
        "final_volume_error": summary.get("final_volume_error", ""),
        "final_abs_volume_error": summary.get("final_abs_volume_error", ""),
        "best_volume_error": summary.get("best_volume_error", ""),
        "best_abs_volume_error": summary.get("best_abs_volume_error", ""),
        "num_all_candidates": summary.get("num_all_candidates", ""),
        "num_valid_candidates": summary.get("num_valid_candidates", ""),
        "best_constraint_ok": summary.get("best_constraint_ok", ""),
        "best_z1": summary.get("best_z1", ""),
        "best_z2": summary.get("best_z2", ""),
        "best_z3": summary.get("best_z3", ""),
        "is_half_hull": summary.get("is_half_hull", ""),
        "point_cloud_path": point_cloud_path,
        "wall_time_sec": wall_time,
    }


def candidate_log_row(algo: str, global_step: int, episode: int, info: Dict) -> Dict:


    
    return {
        "algo": algo,
        "global_step": global_step,
        "episode": episode,
        "L": info.get("L", ""),
        "B": info.get("B", ""),
        "D": info.get("D", ""),
        "T": info.get("T", ""),
        "CB": info.get("CB", ""),
        "V": info.get("V", ""),
        "z0_1": info.get("z0_1", ""),
        "z0_2": info.get("z0_2", ""),
        "z0_3": info.get("z0_3", ""),
        "z1": info.get("z1", ""),
        "z2": info.get("z2", ""),
        "z3": info.get("z3", ""),
        "cb_estimated": info.get("cb_estimated", ""),
        "resistance": info.get("resistance", ""),
        "R": info.get("R", ""),
        "initial_resistance": info.get("initial_resistance", ""),
        "resistance_reduction": info.get("resistance_reduction", ""),
        "volume_actual": info.get("volume_actual", ""),
        "volume_target": info.get("volume_target", ""),
        "volume_error": info.get("volume_error", ""),
        "abs_volume_error": info.get("abs_volume_error", ""),
        "geometry_penalty": info.get("geometry_penalty", ""),
        "constraint_ok": info.get("constraint_ok", ""),
        "is_half_hull": info.get("is_half_hull", ""),
    }


def all_design_log_row(algo: str, global_step: int, episode: int, step_kind: str, env: HullOptimizationEnv, reward="", action=None, error: str = "") -> Dict:


    
    if env.case is None or env.current is None:
        raise RuntimeError("环境尚未 reset，不能记录 all_design_log")
    
    rec = env.current
    
    z_episode0 = np.asarray(env.case.z0, dtype=np.float32).reshape(3)
    
    z_now = np.asarray(rec.z, dtype=np.float32).reshape(3)
    
    if action is None:
        a1 = a2 = a3 = ""
    
    else:
        action = np.asarray(action, dtype=np.float32).reshape(3)
        a1, a2, a3 = float(action[0]), float(action[1]), float(action[2])
    
    return {
        "algo": algo,
        "global_step": int(global_step),
        "episode": int(episode),
        "episode_step": int(env.t),
        "step_kind": step_kind,
        "L": float(env.case.L),
        "B": float(env.case.B),
        "D": float(env.case.D),
        "T": float(env.case.T),
        "CB": float(env.case.CB),
        "V": float(env.case.V),
        "z0": [float(z_now[0]), float(z_now[1]), float(z_now[2])],
        "z0_1": float(z_now[0]),
        "z0_2": float(z_now[1]),
        "z0_3": float(z_now[2]),
        "episode_initial_z0": [float(z_episode0[0]), float(z_episode0[1]), float(z_episode0[2])],
        "episode_initial_z0_1": float(z_episode0[0]),
        "episode_initial_z0_2": float(z_episode0[1]),
        "episode_initial_z0_3": float(z_episode0[2]),
        "R": float(rec.resistance),
        "resistance": float(rec.resistance),
        "initial_resistance": float(env.R0),
        "resistance_ratio": float(rec.resistance / max(float(env.R0), 1e-8)),
        "resistance_reduction": float((float(env.R0) - rec.resistance) / max(float(env.R0), 1e-8)),
        "cb_estimated": float(rec.cb_estimated),
        "volume_actual": float(rec.volume_actual),
        "volume_target": float(rec.volume_target),
        "volume_error": float(rec.volume_error),
        "abs_volume_error": float(abs(rec.volume_error)),
        "geometry_penalty": float(rec.geometry_penalty),
        "constraint_ok": bool(rec.constraint_ok),
        "is_half_hull": bool(rec.is_half_hull),
        "reward": reward,
        "action_1": a1,
        "action_2": a2,
        "action_3": a3,
        "error": error,
    }


def record_all_design_scheme(logger: CSVLogger, algo: str, global_step: int, episode: int, step_kind: str, env: HullOptimizationEnv, reward="", action=None, error: str = "") -> None:


    
    row = all_design_log_row(algo, global_step, episode, step_kind, env, reward=reward, action=action, error=error)
    
    logger.write(row)


def update_log_row(algo: str, global_step: int, update_info: Dict) -> Dict:

    
    return {
        "algo": algo,
        "global_step": global_step,
        "critic_loss": update_info.get("critic_loss", ""),
        "actor_loss": update_info.get("actor_loss", ""),
        "q1_loss": update_info.get("q1_loss", ""),
        "q2_loss": update_info.get("q2_loss", ""),
        "policy_loss": update_info.get("policy_loss", ""),
        "value_loss": update_info.get("value_loss", ""),
        "alpha": update_info.get("alpha", ""),
        "alpha_loss": update_info.get("alpha_loss", ""),
        "approx_kl": update_info.get("approx_kl", ""),
        "entropy_proxy": update_info.get("entropy_proxy", ""),
    }


def evaluate_live_agent(agent, env: HullOptimizationEnv, algo: str, global_step: int, num_episodes: int, eval_cases=None) -> List[Dict]:


    
    rows = []
    
    if eval_cases:
        cases = eval_cases[:]
    
    else:
        cases = [None] * int(num_episodes)
    
    for i, case in enumerate(cases[: max(1, int(num_episodes))] if not eval_cases else cases):
        
        state = env.reset(case=case)
        
        done = False
        
        while not done:
            
            if isinstance(agent, PPOAgent):
                action, _, _ = agent.select_action(state, explore=False)
            
            else:
                action = agent.select_action(state, explore=False)
            
            state, reward, done, info = env.step(action)
        
        row = env.episode_summary()
        
        row.update({"algo": algo, "global_step": global_step, "eval_episode": i})
        
        rows.append(row)
    
    return rows


def maybe_save_best_pointcloud(env: HullOptimizationEnv, output_dir: Path, algo: str, episode: int) -> str:


    
    rec = env.best_record(require_valid=True)
    
    pc_dir = ensure_dir(output_dir / "best_pointclouds")
    
    path = pc_dir / f"{algo}_episode_{episode:06d}_best_half_hull_points.csv"
    
    np.savetxt(path, rec.points, delimiter=",", fmt="%.8f")
    
    return str(path)


def train_offpolicy(algo: str, args, cfg) -> None:

    
    algo = algo.upper()
    
    set_seed(args.seed)
    
    if getattr(args, "torch_num_threads", 0) and int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    
    device = resolve_device(args.device)
    
    out_dir = ensure_dir(args.output_dir)
    
    env_cfg = env_config_from_args(args)
    
    train_env, eval_env, eval_cases = make_envs(args, env_cfg)

    
    if algo == "DDPG":
        agent = DDPGAgent(env_cfg.state_dim, env_cfg.action_dim, cfg, device)
    elif algo == "TD3":
        agent = TD3Agent(env_cfg.state_dim, env_cfg.action_dim, cfg, device)
    elif algo == "SAC":
        agent = SACAgent(env_cfg.state_dim, env_cfg.action_dim, cfg, device)
    else:
        raise ValueError(f" off-policy ：{algo}")

    
    replay = ReplayBuffer(env_cfg.state_dim, env_cfg.action_dim, size=args.replay_size, device=device)
    
    save_json(out_dir / "run_config.json", {"algo": algo, "env_config": env_cfg.to_dict(), "train_config": cfg.to_dict(), "args": vars(args)})
    
    episode_logger = CSVLogger(out_dir / "episode_log.csv")
    
    step_logger = CSVLogger(out_dir / "step_log.csv") if cfg.log_steps else None
    
    candidate_logger = CSVLogger(out_dir / "candidate_log.csv")
    
    all_design_logger = CSVLogger(out_dir / "all_design_log.csv")
    
    update_logger = CSVLogger(out_dir / "update_log.csv")
    
    eval_logger = CSVLogger(out_dir / "eval_log.csv")

    
    state = train_env.reset()
    
    episode = 0
    
    best_metric = -np.inf
    
    start_time = time.time()
    
    record_all_design_scheme(all_design_logger, algo, 0, episode, "reset", train_env)

    
    for global_step in range(1, cfg.total_steps + 1):
        print("Current serp"+str(global_step))
        
        if global_step <= cfg.start_steps:
            action = np.random.uniform(-1.0, 1.0, size=env_cfg.action_dim).astype(np.float32)
        
        else:
            action = agent.select_action(state, explore=True)

        
        next_state, reward, done, info = train_env.step(action)
        
        replay.add(state, action, reward, next_state, done)
        
        record_all_design_scheme(all_design_logger, algo, global_step, episode, "step", train_env, reward=float(reward), action=action, error=info.get("error", ""))
        
        if step_logger is not None:
            step_logger.write(step_log_row(algo, global_step, episode, reward, info))
        
        if info.get("constraint_ok", False):
            candidate_logger.write(candidate_log_row(algo, global_step, episode, info))

        
        state = next_state

        
        if replay.size >= cfg.batch_size and global_step >= cfg.update_after and global_step % cfg.update_every == 0:
            update_info = agent.update(replay, cfg.batch_size, global_step)
            update_logger.write(update_log_row(algo, global_step, update_info))

        
        if done:
            
            summary = train_env.episode_summary()
            
            pc_path = ""
            
            if cfg.save_best_pointclouds:
                pc_path = maybe_save_best_pointcloud(train_env, out_dir, algo, episode)
            
            episode_logger.write(episode_log_row(algo, global_step, episode, summary, time.time() - start_time, pc_path))
            
            metric = float(summary.get("best_resistance_reduction", -np.inf))
            
            if metric > best_metric:
                best_metric = metric
                agent.save(str(out_dir / "model_best.pt"), extra={"global_step": global_step, "best_metric": best_metric})
            
            episode += 1
            
            if global_step < cfg.total_steps:
                state = train_env.reset()
                record_all_design_scheme(all_design_logger, algo, global_step, episode, "reset", train_env)

        
        if args.save_interval > 0 and global_step % args.save_interval == 0:
            agent.save(str(out_dir / f"model_step_{global_step}.pt"), extra={"global_step": global_step, "best_metric": best_metric})

        
        if cfg.eval_interval > 0 and global_step % cfg.eval_interval == 0:
            rows = evaluate_live_agent(agent, eval_env, algo, global_step, cfg.eval_episodes, eval_cases=eval_cases)
            for row in rows:
                eval_logger.write(row)

    
    agent.save(str(out_dir / "model_final.pt"), extra={"global_step": cfg.total_steps, "best_metric": best_metric})



def train_ppo(args, cfg: PPOConfig) -> None:

    
    algo = "PPO"
    
    set_seed(args.seed)
    
    if getattr(args, "torch_num_threads", 0) and int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
    
    device = resolve_device(args.device)
    
    out_dir = ensure_dir(args.output_dir)
    
    env_cfg = env_config_from_args(args)
    
    train_env, eval_env, eval_cases = make_envs(args, env_cfg)
    
    agent = PPOAgent(env_cfg.state_dim, env_cfg.action_dim, cfg, device)

    
    save_json(out_dir / "run_config.json", {"algo": algo, "env_config": env_cfg.to_dict(), "train_config": cfg.to_dict(), "args": vars(args)})
    
    episode_logger = CSVLogger(out_dir / "episode_log.csv")
    
    step_logger = CSVLogger(out_dir / "step_log.csv") if cfg.log_steps else None
    
    candidate_logger = CSVLogger(out_dir / "candidate_log.csv")
    
    all_design_logger = CSVLogger(out_dir / "all_design_log.csv")
    
    update_logger = CSVLogger(out_dir / "update_log.csv")
    
    eval_logger = CSVLogger(out_dir / "eval_log.csv")

    
    state = train_env.reset()
    
    episode = 0
    
    global_step = 0
    
    best_metric = -np.inf
    
    start_time = time.time()
    
    record_all_design_scheme(all_design_logger, algo, global_step, episode, "reset", train_env)

    
    while global_step < cfg.total_steps:
        
        states, actions, rewards, dones, logps, values = [], [], [], [], [], []
        print("Current serp"+str(global_step))
        
        for _ in range(cfg.rollout_steps):
            
            if global_step >= cfg.total_steps:
                break
            
            action, logp, value = agent.select_action(state, explore=True)
            
            next_state, reward, done, info = train_env.step(action)
            
            global_step += 1
            
            states.append(state)
            
            actions.append(action)
            
            rewards.append(float(reward))
            
            dones.append(float(done))
            
            logps.append(float(logp))
            
            values.append(float(value))
            
            record_all_design_scheme(all_design_logger, algo, global_step, episode, "step", train_env, reward=float(reward), action=action, error=info.get("error", ""))
            
            if step_logger is not None:
                step_logger.write(step_log_row(algo, global_step, episode, reward, info))
            
            if info.get("constraint_ok", False):
                candidate_logger.write(candidate_log_row(algo, global_step, episode, info))

            
            state = next_state
            
            if done:
                
                summary = train_env.episode_summary()
                
                pc_path = ""
                
                if cfg.save_best_pointclouds:
                    pc_path = maybe_save_best_pointcloud(train_env, out_dir, algo, episode)
                
                episode_logger.write(episode_log_row(algo, global_step, episode, summary, time.time() - start_time, pc_path))
                
                metric = float(summary.get("best_resistance_reduction", -np.inf))
                if metric > best_metric:
                    best_metric = metric
                    agent.save(str(out_dir / "model_best.pt"), extra={"global_step": global_step, "best_metric": best_metric})
                
                episode += 1
                
                if global_step < cfg.total_steps:
                    state = train_env.reset()
                    record_all_design_scheme(all_design_logger, algo, global_step, episode, "reset", train_env)

            
            if args.save_interval > 0 and global_step % args.save_interval == 0:
                agent.save(str(out_dir / f"model_step_{global_step}.pt"), extra={"global_step": global_step, "best_metric": best_metric})

            
            if cfg.eval_interval > 0 and global_step % cfg.eval_interval == 0:
                rows = evaluate_live_agent(agent, eval_env, algo, global_step, cfg.eval_episodes, eval_cases=eval_cases)
                for row in rows:
                    eval_logger.write(row)

        
        if not states:
            break
        
        next_value = agent.value_of(state)
        
        adv, ret = agent.compute_gae(rewards, dones, values, next_value)
        
        update_info = agent.update(np.asarray(states), np.asarray(actions), np.asarray(logps), adv, ret)
        
        update_logger.write(update_log_row(algo, global_step, update_info))

    
    agent.save(str(out_dir / "model_final.pt"), extra={"global_step": global_step, "best_metric": best_metric})

