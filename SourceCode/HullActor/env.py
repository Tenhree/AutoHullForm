from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from cases import HullCase, sample_case
from geometry import weighted_geometry_penalty
from settings import EnvConfig

@dataclass
class EvaluationRecord:


    z: np.ndarray
    points: np.ndarray
    surface: np.ndarray
    cb_estimated: float
    resistance: float
    volume_actual: float
    volume_target: float
    volume_error: float
    geometry_penalty: float
    geometry_terms: Dict[str, float]
    constraint_ok: bool
    is_half_hull: bool = True

class HullOptimizationEnv:
    def __init__(self, generator, resistance_predictor, cfg: EnvConfig, seed: int = 42, cases: Optional[List[HullCase]] = None):
        self.generator = generator
        self.resistance_predictor = resistance_predictor
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.cases = cases or []
        self._case_cursor = 0
        self.case: Optional[HullCase] = None
        self.z: Optional[np.ndarray] = None
        self.t = 0
        self.R0 = np.nan
        self.current: Optional[EvaluationRecord] = None
        self.initial: Optional[EvaluationRecord] = None
        self.valid_candidates: List[EvaluationRecord] = []
        self.all_candidates: List[EvaluationRecord] = []
        self.best_valid: Optional[EvaluationRecord] = None
        self.best_any: Optional[EvaluationRecord] = None
        self.episode_rewards: List[float] = []

    def _next_case(self, case: Optional[HullCase] = None) -> HullCase:

        if case is not None:
            return case
        if self.cases:
            out = self.cases[self._case_cursor % len(self.cases)]
            self._case_cursor += 1
            return out
        return sample_case(self.rng, self.cfg)

    def reset(self, case: Optional[HullCase] = None) -> np.ndarray:

        self.case = self._next_case(case)
        self.z = np.clip(np.asarray(self.case.z0, dtype=np.float32).reshape(3), self.cfg.z_min, self.cfg.z_max)
        self.t = 0
        self.valid_candidates = []
        self.all_candidates = []
        self.best_valid = None
        self.best_any = None
        self.episode_rewards = []
        self.initial = self._evaluate(self.z)
        self.current = self.initial
        self.R0 = max(float(self.initial.resistance), 1e-8)
        self._record_candidate(self.initial)
        return self._state(self.current)

    def _evaluate(self, z: np.ndarray) -> EvaluationRecord:

        assert self.case is not None
        gen = self.generator.generate(L=self.case.L, B=self.case.B, D=self.case.D, T=self.case.T, CB=self.case.CB, z=z)
        geom_penalty, geom_terms = weighted_geometry_penalty(gen.surface)
        draft_ratio = self.case.T / max(self.case.D, 1e-8)
        resistance = self.resistance_predictor.predict(gen.points, draft_ratio=draft_ratio, speed=self.case.V)
        finite_ok = np.isfinite(resistance) and np.isfinite(gen.volume_error) and np.isfinite(geom_penalty) and np.isfinite(gen.cb_estimated)
        constraint_ok = bool(finite_ok and abs(float(gen.volume_error)) <= self.cfg.eps_volume and float(geom_penalty) <= self.cfg.geometry_penalty_threshold)
        return EvaluationRecord(
            z=np.asarray(z, dtype=np.float32).copy(),
            points=np.asarray(gen.points, dtype=np.float32),
            surface=np.asarray(gen.surface, dtype=np.float32),
            cb_estimated=float(gen.cb_estimated),
            resistance=float(resistance),
            volume_actual=float(gen.volume_actual),
            volume_target=float(gen.volume_target),
            volume_error=float(gen.volume_error),
            geometry_penalty=float(geom_penalty),
            geometry_terms=geom_terms,
            constraint_ok=constraint_ok,
            is_half_hull=bool(getattr(gen, "is_half_hull", True)),
        )

    def _record_candidate(self, rec: EvaluationRecord) -> None:
        self.all_candidates.append(rec)
        if self.best_any is None or rec.resistance < self.best_any.resistance:
            self.best_any = rec
        if rec.constraint_ok:
            self.valid_candidates.append(rec)
            if self.best_valid is None or rec.resistance < self.best_valid.resistance:
                self.best_valid = rec

    def _state(self, rec: EvaluationRecord) -> np.ndarray:

        assert self.case is not None
        R_ratio = float(rec.resistance) / max(float(self.R0), 1e-8)
        s = np.asarray([rec.z[0], rec.z[1], rec.z[2], self.case.B / max(self.case.L, 1e-8), self.case.D / max(self.case.L, 1e-8), self.case.T / max(self.case.D, 1e-8), self.case.CB, self.case.V, R_ratio, rec.volume_error], dtype=np.float32)
        if s.shape[0] != self.cfg.state_dim:
            raise RuntimeError(f"dim error：{s.shape[0]} vs {self.cfg.state_dim}")
        return s

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:

        if self.current is None or self.z is None:
            raise RuntimeError("reset() next step()")
        action = np.asarray(action, dtype=np.float32).reshape(3)
        action = np.clip(action, -1.0, 1.0)
        prev = self.current
        z_next = np.clip(self.z + self.cfg.action_scale * action, self.cfg.z_min, self.cfg.z_max).astype(np.float32)
        try:
            rec = self._evaluate(z_next)
            resistance_reward = self.cfg.lambda_resistance*(prev.resistance - rec.resistance) / max(self.R0, 1e-8)
            volume_violation = max(0.0, abs(rec.volume_error) - self.cfg.eps_volume)
            volume_penalty = self.cfg.lambda_volume * (volume_violation**2)
            geometry_penalty = self.cfg.lambda_geometry * rec.geometry_penalty
            reward = float(resistance_reward - volume_penalty - geometry_penalty)
            if not np.isfinite(reward):
                reward = self.cfg.invalid_reward
        except Exception as exc:
            rec = prev
            reward = float(self.cfg.invalid_reward)
            resistance_reward = 0.0
            volume_penalty = 0.0
            geometry_penalty = 0.0
            self.t += 1
            self.episode_rewards.append(reward)
            done = self.t >= self.cfg.episode_len
            info = self._info(rec)
            info.update({"error": repr(exc), "reward_resistance": float(resistance_reward), "penalty_volume": float(volume_penalty), "penalty_geometry": float(geometry_penalty), "action_1": float(action[0]), "action_2": float(action[1]), "action_3": float(action[2])})
            if done:
                info.update(self.episode_summary())
            return self._state(rec), reward, done, info
        self.z = z_next
        self.current = rec
        self.t += 1
        self._record_candidate(rec)
        self.episode_rewards.append(reward)
        done = self.t >= self.cfg.episode_len
        info = self._info(rec)
        info.update({"reward_resistance": float(resistance_reward), "penalty_volume": float(volume_penalty), "penalty_geometry": float(geometry_penalty), "action_1": float(action[0]), "action_2": float(action[1]), "action_3": float(action[2])})
        if done:
            info.update(self.episode_summary())
        return self._state(rec), reward, done, info

    def _info(self, rec: EvaluationRecord) -> Dict:

        assert self.case is not None
        best = self.best_valid or self.best_any or rec
        z0 = np.asarray(self.case.z0, dtype=np.float32).reshape(3)
        return {
            "t": int(self.t), "L": float(self.case.L), "B": float(self.case.B), "D": float(self.case.D), "T": float(self.case.T), "CB": float(self.case.CB), "V": float(self.case.V),
            "z0_1": float(z0[0]), "z0_2": float(z0[1]), "z0_3": float(z0[2]),
            "z1": float(rec.z[0]), "z2": float(rec.z[1]), "z3": float(rec.z[2]),
            "cb_estimated": float(rec.cb_estimated), "R": float(rec.resistance), "resistance": float(rec.resistance), "initial_resistance": float(self.R0),
            "resistance_ratio": float(rec.resistance / max(self.R0, 1e-8)), "resistance_reduction": float((self.R0 - rec.resistance) / max(self.R0, 1e-8)),
            "volume_actual": float(rec.volume_actual), "volume_target": float(rec.volume_target), "volume_error": float(rec.volume_error), "abs_volume_error": float(abs(rec.volume_error)),
            "geometry_penalty": float(rec.geometry_penalty), "constraint_ok": bool(rec.constraint_ok), "is_half_hull": bool(rec.is_half_hull),
            "best_resistance": float(best.resistance), "best_resistance_reduction": float((self.R0 - best.resistance) / max(self.R0, 1e-8)),
            "best_volume_error": float(best.volume_error), "best_abs_volume_error": float(abs(best.volume_error)), "best_cb_estimated": float(best.cb_estimated), "best_constraint_ok": bool(best.constraint_ok),
        }

    def current_info(self) -> Dict:

        if self.current is None:
            raise RuntimeError(" reset()")
        return self._info(self.current)

    def initial_info(self) -> Dict:
        if self.initial is None:
            raise RuntimeError(" reset()")
        return self._info(self.initial)

    def episode_summary(self) -> Dict:

        if self.current is None or self.initial is None:
            return {}
        best = self.best_valid or self.best_any or self.current
        rewards = np.asarray(self.episode_rewards, dtype=np.float64)
        return {
            "episode_len": int(self.t), "episode_return": float(np.sum(rewards)) if rewards.size else 0.0, "reward_mean": float(np.mean(rewards)) if rewards.size else 0.0, "reward_std": float(np.std(rewards)) if rewards.size else 0.0,
            "initial_resistance": float(self.R0), "initial_cb_estimated": float(self.initial.cb_estimated), "final_resistance": float(self.current.resistance), "final_cb_estimated": float(self.current.cb_estimated),
            "best_resistance": float(best.resistance), "best_cb_estimated": float(best.cb_estimated), "final_resistance_reduction": float((self.R0 - self.current.resistance) / max(self.R0, 1e-8)), "best_resistance_reduction": float((self.R0 - best.resistance) / max(self.R0, 1e-8)),
            "final_volume_error": float(self.current.volume_error), "final_abs_volume_error": float(abs(self.current.volume_error)), "best_volume_error": float(best.volume_error), "best_abs_volume_error": float(abs(best.volume_error)),
            "num_all_candidates": int(len(self.all_candidates)), "num_valid_candidates": int(len(self.valid_candidates)), "best_constraint_ok": bool(best.constraint_ok),
            "best_z1": float(best.z[0]), "best_z2": float(best.z[1]), "best_z3": float(best.z[2]), "is_half_hull": bool(best.is_half_hull),
        }

    def best_record(self, require_valid: bool = True) -> EvaluationRecord:

        if require_valid and self.best_valid is not None:
            return self.best_valid
        if self.best_any is not None:
            return self.best_any
        if self.current is not None:
            return self.current
        raise RuntimeError("no hull")
