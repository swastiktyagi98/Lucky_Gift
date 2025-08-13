# streamlit_app.py
"""
Stable Pool Economy — No Hard Caps Edition (Streamlit)

What changed (fair play):
- No hard clamp on Heat (can be any value).
- No hard clamp on Pool (can go negative; house owes).
- No fixed P(win) min/max — use a smooth logistic mapping to keep 0<p<1 without hard cutoffs.
- Prize affordability no longer clipped by pool balance (house can pay from debt).

What stays:
- Streak-free.
- Energy (shown) = Rewards − Bets.
- Heat (engine) = Bets − Rewards (unbounded; drives fairness).
- Win prize ∈ [0%, 500%] of bet. Zero is possible.
- Consolation is not guaranteed.
- Mid-bet guardrail (₹1k–₹2k).
- Ahead dampers & Behind recovery.
- Persistent UI via st.session_state.
"""

from __future__ import annotations

import math
import random
from math import exp, log
from typing import Dict, Tuple, Optional

import streamlit as st

# =======================
# Config / constants
# =======================
# Soft activation scales (tune only if you really need)
E_SCALE = 300.0
P_SCALE = 50000 # controls how “fast” pool health saturates (for UI + controller only)

# RTP & controller
RTP_TARGET = 1.00          # aim no-profit/no-loss overall
RTP_SENSITIVITY = 0.80     # controller strength

# Base win probability (pre-logistic)
BASE_P = 0.48

# Win prize range: 0%..500% of bet
PRIZE_MIN_MULT = 0.00
PRIZE_MAX_MULT = 10.00

# Prize skew — higher => more small wins; <1 favors big wins
PRIZE_SKEW_BASE = 3.0

# Fees
HOUSE_FEE = 0.06
WIN_FEE_THRESHOLD_MULT = 1.2

# Bets menu
BET_CHOICES = [100, 500, 1000, 2000, 5000, 10_000, 20_000]

# Cosmetic: near-max counts as “jackpot” in labeling
JACKPOT_NEAR_MAX_THRESHOLD = 0.95

# --- Bet-band guardrail (extra house edge around 1k–2k) ---
BET_GUARD_CENTER = 1500.0  # INR mid-band center
BET_GUARD_WIDTH  = 700.0   # smooth width
BET_EDGE_AMPLITUDE = 0.06  # up to -6% p-scale at peak (reduces win prob)
BET_SKEW_BOOST   = 2.0     # up to +2 skew at peak (smaller wins)
BET_CONS_REDUCTION = 0.25  # up to -25% consolation scale at peak
BET_GRANT_REDUCTION = 0.35 # up to -35% consolation grant probability

# --- Ahead (player profit) dampers ---
AHEAD_P_PENALTY = 0.18         # up to -18% p(win) multiplier when far ahead
AHEAD_PRIZE_CUT = 0.40         # cut prize upper bound by up to 40%
AHEAD_SKEW_BOOST = 1.5         # add up to +1.5 skew for small wins
AHEAD_FEE_BOOST  = 0.75        # fee multiplier add-on (1 + AHEAD_FEE_BOOST*ahead)
AHEAD_CONS_GRANT_REDUCTION = 0.60  # reduce consolation grant probability
AHEAD_CONS_SIZE_REDUCTION  = 0.50  # reduce consolation size

# --- Behind (player loss) recovery ---
BEHIND_MIN_FLOOR = 0.05       # raise lower bound of win prize at extreme heat (soft)
BEHIND_SKEW_REDUCTION = 1.2   # reduce exponent up to -1.2 (bigger wins)
MAX_CONS_BASE = 0.50          # max base consolation fraction at extreme heat (soft)

# =======================
# Utility
# =======================
def logistic01(x: float, center: float = 0.5, slope: float = 6.0) -> float:
    """
    Smoothly squash any real x to (0,1) around 'center'.
    If you pass x already ~probability-ish, set center=0.5 and slope around 4..8.
    """
    return 1.0 / (1.0 + math.exp(-slope * (x - center)))

def softplus(x: float) -> float:
    return math.log1p(math.exp(x))

# =======================
# Core class (no hard caps)
# =======================
class StablePoolGame:
    """
    Streak-free, no hard caps.

    Display Energy = Rewards − Bets  (unbounded)
    Engine Heat    = Bets − Rewards  (unbounded; used by activations)
    Consolation is not guaranteed.

    Guardrails:
      • Mid-bet band (₹1k–₹2k) small house-edge bump (probability, skew, consolation).
      • Ahead dampers when player is far ahead (lower p, capped top prizes, higher fees).
      • Behind recovery when player is far behind (higher p-scale, bigger wins, more consolation).

    Pool can go negative (house debt). Prizes are NOT cut by pool balance.
    """

    def __init__(self, starting_pool: float = 50_000.0):
        self.starting_pool: float = float(starting_pool)
        self.pool: float = float(starting_pool)
        self.players: Dict[str, Dict[str, float]] = {}
        # aggregates
        self.total_rounds: int = 0
        self.total_house_fees: float = 0.0
        self.total_pool_contributions: float = 0.0  # reserved; no top-up now

    # ------------ player/state ------------
    def add_player(self, name: str) -> None:
        if name not in self.players:
            self.players[name] = {
                "energy": 0.0,        # Rewards - Bets (display)
                "heat": 0.0,          # Bets - Rewards (unbounded)
                "total_rewards": 0.0,
                "total_bets": 0.0,
                "wins": 0.0,
                "total_rounds": 0.0,
            }

    def ensure_player(self, name: str) -> None:
        if name not in self.players:
            self.add_player(name)

    # ------------ soft activations (no caps) ------------
    @staticmethod
    def _exp01(x: float, scale: float) -> float:
        if scale <= 0:
            return 1.0 if x > 0 else 0.0
        return 1.0 - exp(-max(0.0, x) / scale)

    def heat_activation(self, heat: float) -> float:
        """0..1 as player falls behind (heat = Bets − Rewards). No cap."""
        return self._exp01(heat, E_SCALE)

    def advantage_activation(self, heat: float) -> float:
        """0..1 as player moves ahead (profits = Rewards − Bets). No cap."""
        return self._exp01(-heat, E_SCALE * 1.5)

    def pool_activation(self, pool_balance: float) -> float:
        """0..1 as pool increases above 0 (soft)."""
        return self._exp01(pool_balance, P_SCALE)

    # ------------ controller ------------
    def rtp_controller(self) -> Dict[str, float]:
        total_bets = sum(p["total_bets"] for p in self.players.values())
        total_rewards = sum(p["total_rewards"] for p in self.players.values())
        rtp = (total_rewards / total_bets) if total_bets > 0 else RTP_TARGET

        delta = rtp - RTP_TARGET  # negative => paying too little
        adj = max(-0.35, min(0.35, delta * RTP_SENSITIVITY))

        # When RTP low (delta<0): BOOST p_scale/cons_scale; REDUCE skew
        p_scale = min(1.35, max(0.75, 1.0 - adj))
        cons_scale = min(1.40, max(0.60, 1.0 - 0.5 * adj))

        pool_term = (1.0 - self.pool_activation(self.pool)) * 1.0
        low_rtp_relief = max(0.0, -delta) * 4.0
        skew = max(0.6, PRIZE_SKEW_BASE + pool_term * 1.0 - low_rtp_relief)

        return {"rtp": rtp, "p_scale": p_scale, "cons_scale": cons_scale, "skew": skew}

    # --- bet-band guardrail (smooth Gaussian bump around 1k–2k) ---
    @staticmethod
    def _gauss(x: float, mu: float, sigma: float) -> float:
        z = (x - mu) / max(1e-9, sigma)
        return exp(-z * z)  # 0..1

    def bet_guardrail(self, bet: float) -> Dict[str, float]:
        g = self._gauss(bet, BET_GUARD_CENTER, BET_GUARD_WIDTH)  # 0..1
        p_mult = 1.0 - BET_EDGE_AMPLITUDE * g              # up to -6% p-scale
        cons_mult = 1.0 - BET_CONS_REDUCTION * g           # up to -25% cons-scale
        skew_add = BET_SKEW_BOOST * g                      # up to +2 skew
        grant_prob_mult = 1.0 - BET_GRANT_REDUCTION * g    # up to -35% grant prob
        return {"p_mult": p_mult, "cons_mult": cons_mult, "skew_add": skew_add, "grant_prob_mult": grant_prob_mult}

    # ------------ probabilities & payouts (no hard caps) ------------
    def win_probability(self, heat: float, p_scale: float) -> float:
        # Linear composition…
        heat_boost = self.heat_activation(heat) * 0.08
        pool_adjustment = (self.pool_activation(self.pool) - 0.5) * 0.06
        ahead = self.advantage_activation(heat)
        p_linear = (BASE_P + heat_boost + pool_adjustment) * p_scale * (1.0 - AHEAD_P_PENALTY * ahead)
        # …passed through a smooth logistic to land in (0,1) with no hard min/max
        return logistic01(p_linear, center=0.5, slope=6.0)

    def prize_bounds(self, bet: float) -> Tuple[float, float]:
        lower = round((bet * PRIZE_MIN_MULT) / 10) * 10
        upper = round((bet * PRIZE_MAX_MULT) / 10) * 10
        return lower, upper

    def displayed_prize_bounds(self, bet: float, heat: float) -> Tuple[float, float]:
        """UI helper: show effective bounds after soft ahead/behind effects."""
        lo, hi = self.prize_bounds(bet)
        ahead = self.advantage_activation(heat)
        heat_fac = self.heat_activation(heat)
        pool_mult = 0.5 + 0.5 * self.pool_activation(self.pool)
        lo_eff = max(lo, round((bet * BEHIND_MIN_FLOOR * (heat_fac ** 2) * pool_mult) / 10) * 10)
        hi_eff = round((hi * (1.0 - AHEAD_PRIZE_CUT * ahead)) / 10) * 10
        return lo_eff, hi_eff

    def sample_win_prize(
        self,
        bet: float,
        skew: float,
        rng: Optional[random.Random] = None,
        heat: Optional[float] = None
    ) -> Tuple[float, bool]:
        r = rng if rng is not None else random
        lower = bet * PRIZE_MIN_MULT
        upper = bet * PRIZE_MAX_MULT

        h = 0.0 if heat is None else heat
        heat_fac = self.heat_activation(h)          # 0..1 when behind
        ahead = self.advantage_activation(h)        # 0..1 when ahead

        # Ahead penalty: cap top and increase skew
        upper *= (1.0 - AHEAD_PRIZE_CUT * ahead)

        # Behind recovery: reduce skew (bigger wins) and raise a small floor at extreme heat
        skew_eff = max(0.5, skew + AHEAD_SKEW_BOOST * ahead - BEHIND_SKEW_REDUCTION * heat_fac)
        pool_mult = 0.5 + 0.5 * self.pool_activation(self.pool)
        lower_eff = max(lower, bet * BEHIND_MIN_FLOOR * (heat_fac ** 2) * pool_mult)

        u = r.random() ** skew_eff
        raw = lower_eff + (upper - lower_eff) * u
        prize = round(raw / 10) * 10
        is_jp = prize >= JACKPOT_NEAR_MAX_THRESHOLD * (bet * PRIZE_MAX_MULT)
        return prize, is_jp

    def consolation(self, bet: float, heat: float, cons_scale: float, rng: Optional[random.Random] = None) -> float:
        """Consolation (not guaranteed). Stronger when far BEHIND."""
        r = rng if rng is not None else random
        heat_fac = self.heat_activation(heat)  # 0..1 when behind
        guard = self.bet_guardrail(bet)
        ahead = self.advantage_activation(heat)

        # Grant probability: higher if behind, lower if ahead/band
        grant_prob_linear = (0.10 + 0.75 * heat_fac) * cons_scale * guard["grant_prob_mult"] * (1.0 - AHEAD_CONS_GRANT_REDUCTION * ahead)
        grant_prob = logistic01(grant_prob_linear, center=0.5, slope=5.0)  # smooth, no hard cut
        if r.random() >= grant_prob:
            return 0.0

        # Amount cap (soft): scale up with heat and pool
        pool_mult = 0.5 + 0.5 * self.pool_activation(self.pool)
        max_cons = bet * (MAX_CONS_BASE * heat_fac) * pool_mult * cons_scale * guard["cons_mult"] * (1.0 - AHEAD_CONS_SIZE_REDUCTION * ahead)

        raw = (r.random() ** 1.4) * max_cons
        return round(raw / 10) * 10

    # ------------ one round ------------
    def play_round(self, player: str, bet: float, rng: Optional[random.Random] = None) -> Dict[str, float | bool]:
        self.ensure_player(player)
        r = rng if rng is not None else random

        # Pool takes the bet
        self.pool += bet

        ctrl = self.rtp_controller()
        guard = self.bet_guardrail(bet)
        heat = self.players[player]["heat"]

        # Apply guardrail to probability & skew/cons scales
        p_win = self.win_probability(heat, ctrl["p_scale"] * guard["p_mult"])  # bet-aware prob
        won = r.random() < p_win

        prize = 0.0
        consolation_amt = 0.0
        fee_retained = 0.0
        contribution_paid = 0.0  # reserved; no top-up now
        is_jackpot = False

        if won:
            proposed, is_jackpot = self.sample_win_prize(bet, ctrl["skew"] + guard["skew_add"], r, heat=heat)
            gross = round(proposed / 10) * 10  # NOT pool-capped anymore
            profit_threshold = bet * WIN_FEE_THRESHOLD_MULT
            if gross > profit_threshold:
                profit = max(0.0, gross - bet)
                # extra fee when the player is ahead
                fee_retained = profit * HOUSE_FEE * (1.0 + AHEAD_FEE_BOOST * self.advantage_activation(heat))
                prize = round((gross - fee_retained) / 10) * 10
            else:
                prize = gross
            payout = prize
            self.pool -= payout
            self.players[player]["total_rewards"] += payout
            self.players[player]["wins"] += 1
        else:
            # Heat-based consolation — may be ZERO (no top-up)
            base_cons = self.consolation(bet, heat, ctrl["cons_scale"], r)
            payout = base_cons
            consolation_amt = payout
            self.pool -= payout
            self.players[player]["total_rewards"] += payout

        # aggregates
        self.total_rounds += 1
        self.total_house_fees += fee_retained
        self.total_pool_contributions += contribution_paid
        self.players[player]["total_bets"] += bet
        self.players[player]["total_rounds"] += 1

        # recompute Energy & Heat from totals (no caps)
        tb = self.players[player]["total_bets"]
        tr = self.players[player]["total_rewards"]
        self.players[player]["energy"] = tr - tb                # display Energy
        self.players[player]["heat"] = tb - tr                  # engine Heat (unbounded)

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
        """
        Soft “health” for UI only, no caps: logistic around starting_pool.
        ~0.5 ≈ at starting pool; >0.5 above; <0.5 below.
        """
        if self.starting_pool <= 0:
            return 0.5
        x = (self.pool - self.starting_pool) / max(1.0, self.starting_pool)  # normalized deviation
        return 1.0 / (1.0 + math.exp(-x))

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
            "ledger_delta": delta,  # should be ~0
        }


# =======================
# Streamlit UI
# =======================
def build_streamlit_app():
    if "game" not in st.session_state:
        st.session_state.game = StablePoolGame()
        for name in ("UserA", "UserB", "UserC"):
            st.session_state.game.add_player(name)

    game: StablePoolGame = st.session_state.game

    st.title("🎲 Lucky Rewards")

    # Header
    ph = game.pool_health()
    status = "🟢 Above Start" if ph > 0.55 else "🟡 Near Start" if ph > 0.45 else "🔴 Below Start"
    st.markdown(f"**💰 Pool:** {game.pool:,.0f} {status} (Soft health: {ph:.1%})")
    st.caption("No hard caps: Heat/Pool can move freely. P(win) uses a smooth logistic — no fixed floors/ceilings.")

    st.caption(
        f"Win prize range: {int(PRIZE_MIN_MULT*100)}%–{int(PRIZE_MAX_MULT*100)}% of bet • "
        f"RTP target: {RTP_TARGET:.1%} • Fee: {HOUSE_FEE:.0%} on profit above {WIN_FEE_THRESHOLD_MULT:.1f}× "
        f"• Mid-bet guardrail active around ₹1k–₹2k."
    )

    # Controls
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

    # Predictions
    ctrl = game.rtp_controller()
    guard = game.bet_guardrail(bet_choice)
    heat = game.players[user_choice]["heat"]
    p_win = game.win_probability(heat, ctrl["p_scale"] * guard["p_mult"])
    lo, hi = game.displayed_prize_bounds(bet_choice, heat)

    st.markdown(
        f"**Win Probability:** {p_win:.1%} — Prize bounds: **{lo:,.0f}** to **{hi:,.0f}**  "
        f"**Energy (R−B):** {game.players[user_choice]['energy']:,.0f} • **Heat (B−R):** {heat:,.0f}"
    )
    st.info(
        f"""Actual RTP: {game.rtp()*100:.2f}% | Target: {RTP_TARGET*100:.2f}%
Control — p×: {(ctrl['p_scale']*guard['p_mult']):.3f}, cons×: {(ctrl['cons_scale']*guard['cons_mult']):.3f}, skew: {(ctrl['skew']+guard['skew_add']):.2f}"""
    )

    # Play round / autoplay
    play_col, auto_col = st.columns([1, 1])
    with play_col:
        if st.button("🎲 Play Round"):
            res = game.play_round(user_choice, bet_choice)
            outcome = "🎰 JACKPOT WIN!" if (res["won"] and res["is_jackpot"]) else ("WIN 🎉" if res["won"] else "LOSS ❌")
            st.success(
                f"{outcome}  |  Bet: {bet_choice:,} | Received: {int(res['player_received']):,} "
                f"({res['multiplier']:.2f}×)  |  pWin: {res['p_win']:.1%}"
            )
    with auto_col:
        n = st.number_input("Auto-play rounds", 0, 10000, 0, 50)
        seed = st.number_input("Seed (optional)", value=0, step=1)
        if st.button("▶️ Run Auto-play") and n > 0:
            rng = random.Random(None if seed == 0 else int(seed))
            for _ in range(int(n)):
                game.play_round(user_choice, bet_choice, rng)
            st.info(f"Ran {int(n)} rounds for {user_choice} at bet {bet_choice:,}.")

    # Player table
    st.markdown("**👥 Player Performance:**")
    cols = st.columns(3)
    for i, (name, p) in enumerate(game.players.items()):
        net = p["energy"]
        win_rate = (p["wins"] / p["total_rounds"]) if p["total_rounds"] else 0.0
        with cols[i % 3]:
            st.metric(f"{name} — Net (Energy)", f"{net:+,.0f}")
            st.caption(
                f"Bets: {p['total_bets']:,} • Rewards: {p['total_rewards']:,} • "
                f"WR: {win_rate:.1%} ({int(p['wins'])}/{int(p['total_rounds'])}) • Heat: {p['heat']:,}"
            )

    # Ledger snapshot (debug)
    led = game.ledger_snapshot()
    with st.expander("🧾 Ledger snapshot (debug)"):
        st.json({k: (int(v) if isinstance(v, float) else v) for k, v in led.items()})


# Kick off UI
build_streamlit_app()
