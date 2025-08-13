# stable_pool_game.py (Capless Mode)
"""
Stable Pool Economy — Single-Class Core (Capless Mode)

What changed (per your request to remove caps):
- ❌ Removed pool clamping: pool may go negative or grow unbounded.
- ❌ Removed heat clamping: heat = bets − rewards (unclamped).
- ❌ Removed prize rounding to nearest 10.
- ❌ Removed pool-affordability cap on win payouts (pool can pay even if it goes negative).
- ❌ Removed top-up cap on consolation (was 10% of bet).
- ❌ Removed controller clamps (p_scale/cons_scale/adj no longer clipped).

What remains for correctness (mathematical safety only):
- ✅ Win probability is still bounded to [0, 1].
- ✅ Win prizes still respect your design rule: **50%–500% of bet** (not a cap we remove).

Run:
  streamlit run stable_pool_game.py        # UI (if Streamlit installed)
  python stable_pool_game.py --cli         # CLI sim
  python stable_pool_game.py --test        # unit tests (light)
"""

from __future__ import annotations

import argparse
import random
import sys
import unittest
from math import exp
from typing import Dict, Tuple, Optional

# Optional Streamlit import
try:
    import streamlit as st  # type: ignore
    ST_AVAILABLE = True
except Exception:
    st = None  # type: ignore
    ST_AVAILABLE = False

# =======================
# Config / constants
# =======================
# NOTE: No pool/heat clamps in capless mode.
E_SCALE = 300.0
P_SCALE = 50_000.0  # scale for pool activation; loosely set, not a cap

# RTP & probability control
RTP_TARGET = 0.98
RTP_SENSITIVITY = 0.25
BASE_P = 0.48

# Win prize range: 50%..500% of bet (kept by design)
PRIZE_MIN_MULT = 0.50
PRIZE_MAX_MULT = 5.00

# Prize skew — higher => more small wins
PRIZE_SKEW_BASE = 3.0

# Fees
HOUSE_FEE = 0.06
WIN_FEE_THRESHOLD_MULT = 1.2

# Bets menu
BET_CHOICES = [100, 500, 1000, 2000, 5000, 10_000, 20_000]

# Cosmetic: near-max counts as “jackpot” in labeling
JACKPOT_NEAR_MAX_THRESHOLD = 0.95


# =======================
# Core class
# =======================
class StablePoolGame:
    """
    Capless core for the stable pool economy game.

    Display Energy = Rewards − Bets  (unclamped, user-facing)
    Engine Heat    = Bets − Rewards  (unclamped, stability control)
    No loss-streak counters.
    """

    def __init__(self, starting_pool: float = 50_000.0):
        self.starting_pool: float = float(starting_pool)
        self.pool: float = float(starting_pool)
        self.players: Dict[str, Dict[str, float]] = {}
        # aggregates
        self.total_rounds: int = 0
        self.total_house_fees: float = 0.0
        self.total_pool_contributions: float = 0.0

    # ------------ player/state ------------
    def add_player(self, name: str) -> None:
        if name not in self.players:
            self.players[name] = {
                "energy": 0.0,        # Rewards - Bets (display)
                "heat": 0.0,          # Bets - Rewards (unclamped)
                "total_rewards": 0.0,
                "total_bets": 0.0,
                "wins": 0.0,
                "total_rounds": 0.0,
            }

    def ensure_player(self, name: str) -> None:
        if name not in self.players:
            self.add_player(name)

    # ------------ helpers ------------
    @staticmethod
    def _exp01_pos(x: float, scale: float) -> float:
        """Exponential activation on positive part only (0..1)."""
        return 1.0 - exp(-max(0.0, x) / max(1.0, scale))

    def heat_activation(self, heat: float) -> float:
        # Heat is bets − rewards. Only positive heat (player behind) increases activation.
        return self._exp01_pos(heat, E_SCALE)

    def pool_activation(self, pool_balance: float) -> float:
        # Use pool above 0 as a simple health proxy; negative pool gives ~0 activation.
        return self._exp01_pos(pool_balance, P_SCALE)

    # ------------ controller ------------
    def rtp_controller(self) -> Dict[str, float]:
        total_bets = sum(p["total_bets"] for p in self.players.values())
        total_rewards = sum(p["total_rewards"] for p in self.players.values())
        rtp = (total_rewards / total_bets) if total_bets > 0 else RTP_TARGET

        delta = rtp - RTP_TARGET
        adj = delta * RTP_SENSITIVITY  # no clamp

        p_scale = 1.0 - adj            # may be >1 or <1 (prob still bounded to [0,1] later)
        cons_scale = 1.0 - 0.7 * adj   # may be >1 or <1
        skew = PRIZE_SKEW_BASE + (1.0 - self.pool_activation(self.pool)) * 1.5 + max(0.0, delta) * 5.0
        return {"rtp": rtp, "p_scale": p_scale, "cons_scale": cons_scale, "skew": skew}

    # ------------ probabilities & payouts ------------
    def win_probability(self, heat: float, p_scale: float) -> float:
        heat_boost = self.heat_activation(heat) * 0.08
        pool_adjustment = (self.pool_activation(self.pool) - 0.5) * 0.06
        p = (BASE_P + heat_boost + pool_adjustment) * p_scale
        # Mathematical safety only
        if p < 0.0:
            p = 0.0
        elif p > 1.0:
            p = 1.0
        return p

    def prize_bounds(self, bet: float) -> Tuple[float, float]:
        # No affordability or rounding caps — just the 50%..500% design rule.
        return bet * PRIZE_MIN_MULT, bet * PRIZE_MAX_MULT

    def sample_win_prize(self, bet: float, skew: float, rng: Optional[random.Random] = None) -> Tuple[float, bool]:
        r = rng if rng is not None else random
        low, high = self.prize_bounds(bet)
        u = (r.random()) ** max(1.0, skew)  # skew to smaller wins
        raw = low + (high - low) * u
        prize = raw
        is_jp = prize >= JACKPOT_NEAR_MAX_THRESHOLD * (bet * PRIZE_MAX_MULT)
        return prize, is_jp

    def consolation(self, bet: float, heat: float, cons_scale: float, rng: Optional[random.Random] = None) -> float:
        # Heat-based consolation (no streaks, no caps)
        r = rng if rng is not None else random
        heat_fac = self.heat_activation(heat)  # 0..1
        base_pct = 0.05 + 0.30 * heat_fac      # 5%..35% base
        pool_mult = 0.5 + 0.5 * self.pool_activation(self.pool)  # 50%..100%
        max_cons = bet * base_pct * pool_mult * cons_scale
        raw = (r.random() ** 1.5) * max_cons
        return raw

    # ------------ one round ------------
    def play_round(self, player: str, bet: float, rng: Optional[random.Random] = None) -> Dict[str, float | bool]:
        self.ensure_player(player)
        r = rng if rng is not None else random

        # Pool takes the bet (capless: pool may go negative later)
        self.pool += bet

        ctrl = self.rtp_controller()
        heat = self.players[player]["heat"]
        p_win = self.win_probability(heat, ctrl["p_scale"])
        won = r.random() < p_win

        prize = 0.0
        consolation_amt = 0.0
        fee_retained = 0.0
        contribution_paid = 0.0
        is_jackpot = False

        if won:
            proposed, is_jackpot = self.sample_win_prize(bet, ctrl["skew"], r)
            gross = proposed  # no affordability cap
            profit_threshold = bet * WIN_FEE_THRESHOLD_MULT
            if gross > profit_threshold:
                profit = max(0.0, gross - bet)
                fee_retained = profit * HOUSE_FEE
                prize = gross - fee_retained
            else:
                prize = gross
            payout = prize
            self.pool -= payout
            self.players[player]["total_rewards"] += payout
            self.players[player]["wins"] += 1
        else:
            # Heat-based consolation and top-up (no cap on top-up)
            base_cons = self.consolation(bet, heat, ctrl["cons_scale"], r)
            heat_fac = self.heat_activation(heat)
            target_rate = 0.30 + 0.20 * heat_fac  # 30%..50%
            target_amt = bet * target_rate
            if base_cons < target_amt:
                contribution_paid = (target_amt - base_cons) * ctrl["cons_scale"]
            payout = base_cons + contribution_paid
            consolation_amt = base_cons
            self.pool -= payout
            self.players[player]["total_rewards"] += payout

        # aggregates
        self.total_rounds += 1
        self.total_house_fees += fee_retained
        self.total_pool_contributions += contribution_paid
        self.players[player]["total_bets"] += bet
        self.players[player]["total_rounds"] += 1

        # recompute Energy & Heat from totals (no clamps)
        tb = self.players[player]["total_bets"]
        tr = self.players[player]["total_rewards"]
        self.players[player]["energy"] = tr - tb
        self.players[player]["heat"] = tb - tr

        player_received = payout
        mult = (player_received / bet) if bet else 0.0

        return {
            "won": won,
            "p_win": p_win,
            "prize": prize,
            "consolation": consolation_amt,
            "contribution_paid": contribution_paid,
            "fee_retained": fee_retained,
            "is_jackpot": is_jackpot,
            "player_received": player_received,
            "multiplier": mult,
        }

    # ------------ stats & ledger ------------
    def totals(self) -> Tuple[float, float]:
        total_bets = sum(p["total_bets"] for p in self.players.values())
        total_rewards = sum(p["total_rewards"] for p in self.players.values())
        return total_bets, total_rewards

    def rtp(self) -> float:
        tb, tr = self.totals()
        return (tr / tb) if tb > 0 else 0.0

    def pool_health(self) -> float:
        # In capless mode, show a soft-normalized health based on positive pool only.
        return self._exp01_pos(self.pool, P_SCALE)

    def ledger_snapshot(self) -> Dict[str, float]:
        tb, tr = self.totals()
        expected_pool = self.starting_pool + tb - tr
        delta = self.pool - expected_pool
        return {
            "starting_pool": self.starting_pool,
            "pool": self.pool,
            "total_bets": tb,
            "total_rewards": tr,
            "expected_pool": expected_pool,
            "ledger_delta": delta,  # should remain ~0 if accounting is correct
        }


# =======================
# Streamlit UI
# =======================

def build_streamlit_app():
    if not ST_AVAILABLE:
        raise RuntimeError("Streamlit is not installed in this environment.")

    # Persist game state across reruns
    if "game" not in st.session_state:
        st.session_state.game = StablePoolGame()
        for name in ("UserA", "UserB", "UserC"):
            st.session_state.game.add_player(name)

    game: StablePoolGame = st.session_state.game

    st.title("🎲 Stable Pool Economy — Capless (Energy = Rewards − Bets)")

    ph = game.pool_health()
    status = "🟢 Excellent" if ph > 0.7 else "🟡 Good" if ph > 0.4 else "🟠 Low" if ph > 0.2 else "🔴 Critical"
    st.markdown(f"**💰 Pool:** {game.pool:,.0f} {status} (Health: {ph:.1%})")

    st.caption(
        f"Win prize range: {int(PRIZE_MIN_MULT*100)}%–{int(PRIZE_MAX_MULT*100)}% of bet · "
        f"RTP target: {RTP_TARGET:.1%} · Fee: {HOUSE_FEE:.0%} on profit above {WIN_FEE_THRESHOLD_MULT:.1f}×"
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        user_choice = st.selectbox("Player", list(game.players.keys()))
    with col2:
        default_idx = BET_CHOICES.index(1000) if 1000 in BET_CHOICES else 0
        bet_choice = st.selectbox("Bet", BET_CHOICES, index=default_idx)
    with col3:
        if st.button("🔄 Reset Game"):
            st.session_state.game = StablePoolGame()
            for name in ("UserA", "UserB", "UserC"):
                st.session_state.game.add_player(name)
            st.rerun()

    ctrl = game.rtp_controller()
    heat = game.players[user_choice]["heat"]
    p_win = game.win_probability(heat, ctrl["p_scale"])
    lo, hi = game.prize_bounds(bet_choice)

    st.markdown(
        f"**Win Probability:** {p_win:.1%} — Prize bounds: **{lo:,.0f}** to **{hi:,.0f}**  "
        f"**Energy (R−B):** {game.players[user_choice]['energy']:,.0f} • **Heat (B−R):** {heat:,.0f}"
    )
    st.info(
        f"""Actual RTP: {game.rtp()*100:.2f}% | Target: {RTP_TARGET*100:.2f}%
Control — p×: {ctrl['p_scale']:.3f}, cons×: {ctrl['cons_scale']:.3f}, skew: {ctrl['skew']:.2f}"""
    )

    if st.button("🎲 Play Round"):
        res = game.play_round(user_choice, bet_choice)
        outcome = "🎰 JACKPOT WIN!" if (res["won"] and res["is_jackpot"]) else ("WIN 🎉" if res["won"] else "LOSS ❌")
        st.success(
            f"{outcome}  |  Received: {res['player_received']:.0f} ({res['multiplier']:.2f}×)  |  pWin: {res['p_win']:.1%}"
        )

    st.markdown("**👥 Player Performance:**")
    for name, p in game.players.items():
        net = p["energy"]
        win_rate = (p["wins"] / p["total_rounds"]) if p["total_rounds"] else 0.0
        net_color = "green" if net >= 0 else "red"
        st.markdown(
            f"**{name}:** Bets: {p['total_bets']:.0f} | Rewards: {p['total_rewards']:.0f} | "
            f"<span style='color:{net_color}'>Net: {net:+.0f}</span> | Win Rate: {win_rate:.1%} "
            f"({int(p['wins'])}/{int(p['total_rounds'])}) | Energy (R−B): {p['energy']:.0f} | Heat: {p['heat']:.0f}",
            unsafe_allow_html=True,
        )

    led = game.ledger_snapshot()
    with st.expander("🧾 Ledger snapshot (debug)"):
        st.json({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in led.items()})


# =======================
# CLI Simulator (no Streamlit)
# =======================

def cli_sim(rounds: int, seed: int | None):
    rng = random.Random(seed) if seed is not None else random
    game = StablePoolGame()
    for name in ("UserA", "UserB", "UserC"):
        game.add_player(name)

    for i in range(1, rounds + 1):
        user = rng.choice(list(game.players.keys()))
        bet = rng.choice(BET_CHOICES)
        pool_before = game.pool
        res = game.play_round(user, bet, rng)
        pool_delta = game.pool - pool_before
        print(
            f"#{i:03d} {user} bet={bet:>5} pWin={res['p_win']:.2%} outcome={'WIN' if res['won'] else 'LOSS'} "
            f"received={res['player_received']:.0f} ({res['multiplier']:.2f}x) poolΔ={pool_delta:+.0f}"
        )

    tb, tr = game.totals()
    print("—" * 60)
    print(f"Rounds={rounds}  Total Bets={tb:.0f}  Total Rewards={tr:.0f}  RTP={game.rtp():.2%}")
    print(f"Final Pool={game.pool:.0f}  P&L vs start={game.pool-game.starting_pool:+.0f}")


# =======================
# Tests (light)
# =======================
class TestPrizeBounds(unittest.TestCase):
    def test_bounds(self):
        game = StablePoolGame(starting_pool=0)
        bet = 1000
        lo, hi = game.prize_bounds(bet)
        self.assertEqual(lo, bet * PRIZE_MIN_MULT)
        self.assertEqual(hi, bet * PRIZE_MAX_MULT)


# =======================
# Entrypoint
# =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run CLI simulator instead of Streamlit")
    parser.add_argument("--rounds", type=int, default=30, help="Rounds for CLI sim")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for CLI sim")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    args = parser.parse_args()

    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    if args.cli or not ST_AVAILABLE:
        cli_sim(args.rounds, args.seed)
    else:
        build_streamlit_app()


if __name__ == "__main__":
    main()
