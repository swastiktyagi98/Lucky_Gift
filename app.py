import random
import time
from typing import Dict, List

import streamlit as st

# ================================
# Game economics (no-fee version)
# ================================

BASELINE_WIN_PROBABILITY = 0.85
BASELINE_PRIZE_MULTIPLIERS = [
    0.0,
    0.5, 0.6, 0.8, 0.85, 0.9,
    1.0, 1.05,
    1.2, 1.5, 2.0, 5.0, 8.0
]
BASELINE_PRIZE_WEIGHTS = [
    1,
    8, 8, 8, 8, 8, 8, 8,
    6, 6,
    14, 11, 9
]

# 👉 No fee in this version
SYSTEM_FEE_RATE = 0.0

# A round "pays" with a prize with this probability
CASH_WIN_CHANCE = 0.97
MIN_MICRO_WIN = 0.05

# Active prize table (on pays)
PRIZE_MULTIPLIERS = [
    MIN_MICRO_WIN,
    0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9,
    1.0, 1.05,
    1.2, 1.5, 2.0
]
PRIZE_WEIGHTS = [
    1,
    8, 8, 8, 8, 8, 8, 8,
    6, 6,
    14, 11, 9
]

BET_CHOICES = [50, 100, 500, 1_000, 5_000, 10_000, 20_000]
STARTING_POOL = 0.0


# ================================
# Helpers (RTP preservation)
# ================================

def _weighted_avg(mults: List[float], weights: List[int]) -> float:
    tw = sum(weights)
    return sum(m * w for m, w in zip(mults, weights)) / tw if tw else 0.0


BASELINE_EXPECTED_PAYOUT_FACTOR = BASELINE_WIN_PROBABILITY * _weighted_avg(
    BASELINE_PRIZE_MULTIPLIERS, BASELINE_PRIZE_WEIGHTS
)
_active_avg = _weighted_avg(PRIZE_MULTIPLIERS, PRIZE_WEIGHTS)

# Scale the table so EV matches the old baseline despite CASH_WIN_CHANCE changes
PRIZE_SCALE = BASELINE_EXPECTED_PAYOUT_FACTOR / max(CASH_WIN_CHANCE * _active_avg, 1e-9)
PRIZE_SCALE = max(min(PRIZE_SCALE, 2.0), 0.2)  # clamp for stability


# ================================
# Core game object
# ================================

class NoFeePoolGame:
    def __init__(self):
        self.pool = float(STARTING_POOL)
        self.total_bets = 0.0
        self.total_payouts = 0.0
        self.total_rounds = 0

        # energy tracks bet - prize (per player)
        self.players: Dict[str, Dict] = {
            "Alice":   {"energy": 0.0, "total_bet": 0.0, "total_prize": 0.0, "rounds": 0, "wins": 0, "losses": 0},
            "Bob":     {"energy": 0.0, "total_bet": 0.0, "total_prize": 0.0, "rounds": 0, "wins": 0, "losses": 0},
            "Charlie": {"energy": 0.0, "total_bet": 0.0, "total_prize": 0.0, "rounds": 0, "wins": 0, "losses": 0},
        }

        # rolling history (most recent first)
        self.history: List[Dict] = []

    def _select_multiplier(self) -> float:
        # Fallback to uniform choice if a mismatch is ever introduced
        if len(PRIZE_MULTIPLIERS) != len(PRIZE_WEIGHTS):
            return random.choice(PRIZE_MULTIPLIERS)
        return random.choices(PRIZE_MULTIPLIERS, weights=PRIZE_WEIGHTS)[0]

    def play_round(self, player: str, bet_amount: float) -> Dict:
        bet_amount = float(round(bet_amount, 2))
        assert bet_amount > 0, "Bet must be > 0"

        self.total_rounds += 1
        self.total_bets += bet_amount

        p = self.players[player]
        p["total_bet"] += bet_amount
        p["rounds"] += 1

        # Pool receives full bet (no fee)
        self.pool = round(self.pool + bet_amount, 2)

        pays = (random.random() < CASH_WIN_CHANCE)
        prize = 0.0
        eff_mult = 0.0

        if pays:
            base_mult = self._select_multiplier()
            eff_mult = round(base_mult * PRIZE_SCALE, 4)
            prize = round(bet_amount * eff_mult, 2)

            # Pool can go negative
            self.pool = round(self.pool - prize, 2)

        # Track payouts & W/L (loss only when prize == 0)
        self.total_payouts += prize
        p["total_prize"] += prize

        status = "win" if prize > 0 else "loss"
        if status == "win":
            p["wins"] += 1
        else:
            p["losses"] += 1

        # Energy mirrors pool change for this user: bet - prize
        round_energy = round(bet_amount - prize, 2)
        p["energy"] = round(p["energy"] + round_energy, 2)

        result = {
            "round": self.total_rounds,
            "player": player,
            "bet": bet_amount,
            "prize": prize,
            "multiplier": eff_mult if prize > 0 else 0.0,
            "status": status,
            "cash_win": pays,
            "pool_after": self.pool,
            "round_energy": round_energy,
            "total_energy_after": p["energy"],
            "ts": time.time(),
        }

        self.history.insert(0, result)
        self.history = self.history[:200]  # keep last 200

        return result

    def get_stats(self) -> Dict:
        total_energy = sum(pl["energy"] for pl in self.players.values())
        rtp = (self.total_payouts / self.total_bets) if self.total_bets > 0 else 0.0

        return {
            "total_bets": self.total_bets,
            "total_payouts": self.total_payouts,
            "rtp": rtp,
            "pool": self.pool,
            "total_rounds": self.total_rounds,
            "sum_energy": total_energy,
            "invariant_delta": round(self.pool - (STARTING_POOL + total_energy), 2),
        }

    def reset(self):
        self.__init__()


def _last_n_cash_win_rate(history: List[Dict], n=10) -> float:
    if not history:
        return 0.0
    sample = history[:n]  # history is newest first
    wins = sum(1 for r in sample if r["cash_win"])
    return wins / len(sample)


# ================================
# Streamlit UI
# ================================

st.set_page_config(page_title="🎲 Lucky Gift — No-Fee Pool", page_icon="🎲", layout="centered")

if "game" not in st.session_state:
    st.session_state.game = NoFeePoolGame()

game: NoFeePoolGame = st.session_state.game

st.title("🎲 Lucky Gift — No-Fee Pool")
st.caption(
    f"Cash Win Chance: {CASH_WIN_CHANCE:.0%} • Prize scale={PRIZE_SCALE:.3f} • "
    f"Energy = bet − prize • Pool can be negative"
)

# --- Top stats
stats = game.get_stats()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 Pool", f"${stats['pool']:,.2f}")
with col2:
    st.metric("🎮 Rounds", stats["total_rounds"])
with col3:
    st.metric("📈 Player RTP", f"{stats['rtp']:.1%}")
with col4:
    inv_ok = abs(stats["invariant_delta"]) <= 0.02
    st.metric("♾️ Invariant Δ", f"${stats['invariant_delta']:,.2f}",
              help="pool − (startPool + Σ energy)")
st.markdown("**Invariant:** Pool should equal sum of all users’ energy (±¢).")

st.divider()

# --- Controls
st.subheader("🎲 Play a Round")

c1, c2, c3 = st.columns(3)
with c1:
    selected_player = st.selectbox("Choose Player", list(game.players.keys()))
with c2:
    selected_bet = st.selectbox("Bet Amount", BET_CHOICES, index=BET_CHOICES.index(100) if 100 in BET_CHOICES else 0)
with c3:
    if st.button("🔄 Add Random Player"):
        new_name = f"Player {len(game.players)+1}"
        game.players[new_name] = {"energy": 0.0, "total_bet": 0.0, "total_prize": 0.0, "rounds": 0, "wins": 0, "losses": 0}
        selected_player = new_name
        st.rerun()

# Effective prize info
active_avg_unscaled = _weighted_avg(PRIZE_MULTIPLIERS, PRIZE_WEIGHTS)
effective_avg = active_avg_unscaled * PRIZE_SCALE
min_eff = min(PRIZE_MULTIPLIERS) * PRIZE_SCALE
max_eff = max(PRIZE_MULTIPLIERS) * PRIZE_SCALE

min_prize = selected_bet * min_eff
max_prize = selected_bet * max_eff

st.info(
    f"💡 On a pay: avg ≈ {effective_avg:.3f}× (min {min_eff:.2f}×, max {max_eff:.2f}×). "
    f"Prize range (for bet ${selected_bet:,.0f}): ${min_prize:,.2f} – ${max_prize:,.2f}."
)

# Play & Auto buttons
pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    if st.button("🎯 PLAY NOW", use_container_width=True, type="primary"):
        res = game.play_round(selected_player, selected_bet)
        st.session_state.last_result = res
        st.rerun()
with pc2:
    if st.button("⚡ Auto ×10", use_container_width=True):
        for _ in range(10):
            p = random.choice(list(game.players.keys()))
            b = random.choice(BET_CHOICES)
            game.play_round(p, b)
        st.rerun()
with pc3:
    if st.button("⚡ Auto ×50", use_container_width=True):
        for _ in range(50):
            p = random.choice(list(game.players.keys()))
            b = random.choice(BET_CHOICES)
            game.play_round(p, b)
        st.rerun()
with pc4:
    if st.button("🗑️ Reset Game", use_container_width=True):
        game.reset()
        st.session_state.last_result = None
        st.success("Game reset.")
        st.rerun()

# Last result card
if st.session_state.get("last_result"):
    r = st.session_state.last_result
    if r["status"] == "win":
        st.success(f"🎉 WIN! {r['player']} won ${r['prize']:,.2f} ({r['multiplier']:.2f}×)")
    else:
        st.error(f"❌ Loss — {r['player']} prize $0")

    with st.expander("Result details", expanded=False):
        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Bet", f"${r['bet']:,.2f}")
            st.metric("Prize", f"${r['prize']:,.2f}")
        with cB:
            st.metric("Multiplier", f"{r['multiplier']:.2f}×" if r["prize"] > 0 else "—")
            st.metric("Round Energy", f"{r['round_energy']:,.2f}")
        with cC:
            st.metric("Pool After", f"${r['pool_after']:,.2f}")
            st.metric("User Energy After", f"{r['total_energy_after']:,.2f}")

st.divider()

# --- Player statistics
st.subheader("👥 Player Statistics")
for name, pl in game.players.items():
    rounds = pl["rounds"]
    wins = pl["wins"]
    losses = pl["losses"]
    total_bet = pl["total_bet"]
    total_prize = pl["total_prize"]
    rtp = (total_prize / total_bet) if total_bet > 0 else 0.0
    net_spend = total_bet - total_prize  # positive = loss, negative = profit
    avg_energy = (pl["energy"] / rounds) if rounds > 0 else 0.0
    win_rate = (wins / rounds) if rounds > 0 else 0.0

    with st.expander(f"**{name}** — Net Spend: ${net_spend:,.2f}", expanded=(name == selected_player)):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rounds", rounds)
            st.metric("Wins", wins)
        with c2:
            st.metric("Losses", losses)
            st.metric("Win Rate", f"{win_rate:.1%}")
        with c3:
            st.metric("Total Bet", f"${total_bet:,.2f}")
            st.metric("Total Prize", f"${total_prize:,.2f}")
        with c4:
            st.metric("RTP", f"{rtp:.1%}")
            st.metric("Total Energy", f"{pl['energy']:,.2f}", help="Sum of (bet − prize)")
            st.metric("Avg Energy", f"{avg_energy:,.2f}")

# --- System performance
st.divider()
st.subheader("🎯 System Performance")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Player Bets", f"${stats['total_bets']:,.2f}")
with c2:
    st.metric("Total Payouts", f"${stats['total_payouts']:,.2f}")
with c3:
    st.metric("Player RTP", f"{stats['rtp']:.1%}")

# Invariant check details
st.info(
    f"**Invariant check:** pool (${stats['pool']:,.2f}) vs startPool + Σenergy "
    f"(${STARTING_POOL + stats['sum_energy']:,.2f}) → Δ ${stats['invariant_delta']:,.2f}"
)

# Recent cash win rate
if game.history:
    st.divider()
    last10 = _last_n_cash_win_rate(game.history, 10)
    st.write(f"Recent cash win rate (last 10): **{last10:.0%}** (target {CASH_WIN_CHANCE:.0%})")

# Recent rounds (newest first)
if game.history:
    st.divider()
    st.subheader("📝 Recent Rounds")
    for rr in game.history[:20]:
        outcome = f"🎉 WIN ${rr['prize']:,.2f} ({rr['multiplier']:.2f}×)" if rr["prize"] > 0 else "❌ LOSS $0"
        st.text(
            f"Round {rr['round']:>4}: {rr['player']} bet ${rr['bet']:,.2f} → {outcome} | "
            f"Pool {rr['pool_after']:,.2f} | dEnergy {rr['round_energy']:,.2f} | E_after {rr['total_energy_after']:,.2f}"
        )

# Prize distribution
st.divider()
st.subheader("🎁 Prize Distribution (Effective on Pay)")
total_w = sum(PRIZE_WEIGHTS)
colA, colB = st.columns(2)
with colA:
    st.markdown("**Common Prizes:**")
    for mult, weight in zip(PRIZE_MULTIPLIERS[0:10], PRIZE_WEIGHTS[0:10]):
        pct = (weight / total_w) * 100 if total_w else 0
        st.text(f"{mult*PRIZE_SCALE:.2f}×  —  {pct:.1f}% of pays")
with colB:
    st.markdown("**Rare & Bigger Prizes:**")
    for mult, weight in zip(PRIZE_MULTIPLIERS[10:], PRIZE_WEIGHTS[10:]):
        pct = (weight / total_w) * 100 if total_w else 0
        st.text(f"{mult*PRIZE_SCALE:.2f}×  —  {pct:.1f}% of pays")
