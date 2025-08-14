# player_friendly_pool_game.py
"""
Player-Friendly Pool Game

Goal of this version:
- Push Cash Win Rate higher (97%) WITHOUT increasing loss (pool profit stays the same).
- Keep micro-win so every paying round > $0.
- Auto-scale prizes so RTP = baseline (no pool loss increase).
- Persist and show the latest round result under the PLAY button.
- Show last-10-rounds cash win rate to verify behavior quickly.
"""

import random
import streamlit as st
from typing import Dict, List

# =======================
# Baseline (for RTP lock)
# =======================
# These are the OLD settings used to compute the baseline expected payout.
# We lock the new config to this expected payout so pool profit is unchanged.

BASELINE_WIN_PROBABILITY = 0.85
BASELINE_PRIZE_MULTIPLIERS = [
    0.0,
    0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9,
    1.0, 1.05,
    1.2, 1.5, 2.0, 5.0,8,.0
]
BASELINE_PRIZE_WEIGHTS = [
    1,
    8, 8, 8, 8, 8, 8, 8,
    6, 6,
    14, 11, 9
]

def _weighted_avg(mults: List[float], weights: List[int]) -> float:
    tw = sum(weights)
    return sum(m*w for m, w in zip(mults, weights)) / tw if tw else 0.0

BASELINE_EXPECTED_PAYOUT_FACTOR = BASELINE_WIN_PROBABILITY * _weighted_avg(
    BASELINE_PRIZE_MULTIPLIERS, BASELINE_PRIZE_WEIGHTS
)
# ≈ 0.862 of bet paid on average; pool still receives 90% (after system fee).

# =======================
# Active configuration
# =======================

SYSTEM_FEE_RATE = 0.10
BET_CHOICES = [100, 500, 1000, 2000, 5000, 10_000]

# TARGET cash win chance (payout > 0).
CASH_WIN_CHANCE = 0.97  # 97% paying rounds; triple-loss odds ~0.0027%

# Micro-win so every paying round > $0 (pre-scale).
MIN_MICRO_WIN = 0.05  # 5% of bet on the rarest outcome before scaling

# Prize multipliers (we’ll scale with PRIZE_SCALE to hold RTP at baseline)
PRIZE_MULTIPLIERS = [
    MIN_MICRO_WIN,       # used to be 0.0 in baseline
    0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9,
    1.0, 1.05,
    1.2, 1.5, 2.0
]

# Same weights as before (keeps the “feel” of the wheel)
PRIZE_WEIGHTS = [
    1,
    8, 8, 8, 8, 8, 8, 8,
    6, 6,
    14, 11, 9
]

STARTING_POOL = 100_000.0

# =======================
# Derived calibration
# =======================

_active_avg = _weighted_avg(PRIZE_MULTIPLIERS, PRIZE_WEIGHTS)
# Scale so: CASH_WIN_CHANCE * (_active_avg * PRIZE_SCALE) == BASELINE_EXPECTED_PAYOUT_FACTOR
PRIZE_SCALE = BASELINE_EXPECTED_PAYOUT_FACTOR / max(CASH_WIN_CHANCE * _active_avg, 1e-9)
PRIZE_SCALE = max(min(PRIZE_SCALE, 2.0), 0.2)  # safety bounds

# =======================
# Game class
# =======================

class PlayerFriendlyPoolGame:
    """
    Frequent cash wins with pool protection.
    Expected payout is locked to baseline via PRIZE_SCALE → pool profit unchanged.
    """
    def __init__(self):
        self.pool = STARTING_POOL
        self.total_bets = 0.0
        self.total_fees_collected = 0.0
        self.total_payouts = 0.0
        self.total_rounds = 0

        self.players = {
            "Player 1": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0, "hits": 0},
            "Player 2": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0, "hits": 0},
            "Player 3": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0, "hits": 0},
        }

        self.history = []

    def _select_multiplier(self) -> float:
        if len(PRIZE_MULTIPLIERS) != len(PRIZE_WEIGHTS):
            return random.choice(PRIZE_MULTIPLIERS)
        return random.choices(PRIZE_MULTIPLIERS, weights=PRIZE_WEIGHTS)[0]

    def play_round(self, player: str, bet_amount: float) -> Dict:
        system_fee = bet_amount * SYSTEM_FEE_RATE
        effective_bet = bet_amount - system_fee
        self.pool += effective_bet

        self.total_bets += bet_amount
        self.total_fees_collected += system_fee
        self.total_rounds += 1

        p = self.players[player]
        p["total_bet"] += bet_amount
        p["rounds"] += 1

        # Cash win gate
        pays = random.random() < CASH_WIN_CHANCE

        prize = 0.0
        eff_mult = 0.0
        hit = False

        if pays:
            hit = True
            base_mult = self._select_multiplier()
            eff_mult = base_mult * PRIZE_SCALE
            prize = bet_amount * eff_mult

            # Pay out from pool
            if self.pool >= prize:
                self.pool -= prize
            else:
                # Emergency fallback: pay half of remaining pool
                prize = max(0, self.pool * 0.5)
                eff_mult = (prize / bet_amount) if bet_amount > 0 else 0.0
                self.pool -= prize

            p["total_won"] += prize
            p["wins"] += 1
            p["hits"] += 1
            self.total_payouts += prize

        p["balance"] = p["total_won"] - p["total_bet"]

        result = {
            "round": self.total_rounds,
            "player": player,
            "bet": bet_amount,
            "system_fee": system_fee,
            "effective_bet": effective_bet,
            "hit": hit,
            "won": pays,                 # == hit; every hit pays > 0
            "prize": prize,
            "multiplier": eff_mult,      # effective (post-scale) multiplier
            "pool_after": self.pool
        }

        self.history.append(result)
        if len(self.history) > 50:
            self.history.pop(0)

        return result

    def get_stats(self) -> Dict:
        pool_profit = (self.total_bets * (1 - SYSTEM_FEE_RATE)) - self.total_payouts
        return {
            "total_bets": self.total_bets,
            "total_fees": self.total_fees_collected,
            "total_payouts": self.total_payouts,
            "pool_profit": pool_profit,
            "net_pool_change": self.pool - STARTING_POOL,
            "total_rounds": self.total_rounds,
            "effective_rtp": (self.total_payouts / self.total_bets) if self.total_bets > 0 else 0,
            "pool_rtp": (self.total_payouts / (self.total_bets * (1 - SYSTEM_FEE_RATE))) if self.total_bets > 0 else 0
        }

    def reset_game(self):
        self.__init__()

# =======================
# Streamlit UI
# =======================

def _last_n_cash_win_rate(history, n=10) -> float:
    if not history:
        return 0.0
    sample = history[-n:]
    wins = sum(1 for r in sample if r["won"])
    return wins / len(sample)

def main():
    st.set_page_config(page_title="Player-Friendly Pool Game", page_icon="🎊", layout="centered")

    if "game" not in st.session_state:
        st.session_state.game = PlayerFriendlyPoolGame()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    game = st.session_state.game

    # Header
    st.title("🎊 Player-Friendly Pool Game")
    st.caption(
        f"Cash Win Chance: {CASH_WIN_CHANCE:.0%} • RTP locked to baseline "
        f"(scale={PRIZE_SCALE:.3f}, baseline EV={BASELINE_EXPECTED_PAYOUT_FACTOR:.3f})"
    )

    # Stats header
    game_stats = game.get_stats()
    pool_health = "🟢 Profitable" if game_stats["pool_profit"] >= 0 else "🔴 Losing"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Pool", f"${game.pool:,.0f}")
    with col2:
        st.metric("🎮 Rounds", game.total_rounds)
    with col3:
        st.metric("🏦 System Fees", f"${game.total_fees_collected:,.0f}")
    with col4:
        st.metric("📊 Pool Profit", f"${game_stats['pool_profit']:+,.0f}", help="How much the pool has gained/lost")

    st.markdown(f"**Pool Status:** {pool_health}")
    st.divider()

    # Game Controls
    st.subheader("🎲 Play a Round")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_player = st.selectbox("Choose Player", list(game.players.keys()))
    with col2:
        selected_bet = st.selectbox("Bet Amount", BET_CHOICES)
    with col3:
        st.metric("Cash Win Chance", f"{CASH_WIN_CHANCE:.0%}", help="Probability a round pays > $0")

    # Prize info
    active_avg_unscaled = _weighted_avg(PRIZE_MULTIPLIERS, PRIZE_WEIGHTS)
    effective_avg = active_avg_unscaled * PRIZE_SCALE
    min_eff = min(PRIZE_MULTIPLIERS) * PRIZE_SCALE
    max_eff = max(PRIZE_MULTIPLIERS) * PRIZE_SCALE

    min_prize = selected_bet * min_eff
    max_prize = selected_bet * max_eff
    avg_prize = selected_bet * effective_avg

    st.info(
        f"💡 Effective multipliers after scaling: avg ≈ {effective_avg:.3f}× "
        f"(min {min_eff:.2f}×, max {max_eff:.2f}×). "
        f"Prize range: ${min_prize:,.0f}–${max_prize:,.0f}. "
        f"System fee: ${selected_bet * SYSTEM_FEE_RATE:,.0f}."
    )

    # Play button
    play_clicked = st.button("🎲 PLAY ROUND", type="primary", use_container_width=True)
    if play_clicked:
        result = game.play_round(selected_player, selected_bet)
        st.session_state.last_result = result
        st.rerun()

    # Current result (persistent under button)
    if st.session_state.last_result:
        result = st.session_state.last_result
        if result["won"]:
            if result["multiplier"] >= 1.2:
                st.balloons()
                st.success(f"🎉 BIG WIN! {result['player']} won ${result['prize']:,.0f} ({result['multiplier']:.2f}×)")
            elif result["multiplier"] >= 0.8:
                st.success(f"✅ GOOD WIN! {result['player']} won ${result['prize']:,.0f} ({result['multiplier']:.2f}×)")
            else:
                st.success(f"🎊 WIN! {result['player']} won ${result['prize']:,.0f} ({result['multiplier']:.2f}×)")
        else:
            st.error(f"❌ Loss — {result['player']} lost ${result['bet']:,.0f}")

        with st.expander("Result details", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Bet", f"${result['bet']:,.0f}")
                st.metric("System Fee", f"${result['system_fee']:,.0f}")
            with col_b:
                st.metric("Effective to Pool", f"${result['effective_bet']:,.0f}")
                st.metric("Prize", f"${result['prize']:,.0f}")
            with col_c:
                st.metric("Multiplier", f"{result['multiplier']:.2f}×" if result["won"] else "—")
                st.metric("Pool After", f"${result['pool_after']:,.0f}")

    st.divider()

    # Player Statistics
    st.subheader("📊 Player Statistics")

    for player_name, stats in game.players.items():
        cash_win_rate = (stats["wins"] / stats["rounds"]) if stats["rounds"] > 0 else 0
        hit_rate = (stats["hits"] / stats["rounds"]) if stats["rounds"] > 0 else 0  # same as wins now
        balance_color = "green" if stats["balance"] >= 0 else "red"

        with st.expander(f"**{player_name}** — Balance: ${stats['balance']:+,.0f}",
                         expanded=(player_name == selected_player)):

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Bet", f"${stats['total_bet']:,.0f}")
                st.metric("Rounds Played", stats["rounds"])
            with col2:
                st.metric("Total Won", f"${stats['total_won']:,.0f}")
                st.metric("Cash Wins", f"{stats['wins']}")
            with col3:
                st.markdown(
                    f"**Balance:** <span style='color:{balance_color}'>${stats['balance']:+,.0f}</span>",
                    unsafe_allow_html=True
                )
                st.metric("Hit Rate", f"{hit_rate:.1%}", help="Paying rounds")
                st.metric("Cash Win Rate", f"{cash_win_rate:.1%}", help="Rounds with payout > 0")

    # Quick feel check: last-10 cash win rate
    if game.history:
        st.divider()
        last10 = _last_n_cash_win_rate(game.history, 10)
        st.write(f"Recent cash win rate (last 10): **{last10:.0%}** (target {CASH_WIN_CHANCE:.0%})")

    # System Performance
    st.divider()
    st.subheader("🎯 System Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Player RTP", f"{game_stats['effective_rtp']:.1%}", help="Total payouts / Total bets")
        st.metric("Total Player Bets", f"${game_stats['total_bets']:,.0f}")
        st.metric("Total Payouts", f"${game_stats['total_payouts']:,.0f}")
    with col2:
        st.metric("Pool RTP", f"{game_stats['pool_rtp']:.1%}", help="Payouts vs money that went to pool")
        st.metric("Pool Receives", f"${game_stats['total_bets'] * (1 - SYSTEM_FEE_RATE):,.0f}")
        st.metric("Pool Net Gain", f"${game_stats['pool_profit']:+,.0f}")

    if game_stats['total_rounds'] > 0:
        st.info(f"""
        **Why the pool stays profitable:**
        - Cash Win Chance set to {CASH_WIN_CHANCE:.0%} for better feel.
        - We auto-scale prizes (factor {PRIZE_SCALE:.3f}) so expected payout stays at the previous baseline.
        - Pool receives {(1-SYSTEM_FEE_RATE)*100:.0f}% of bets, pays out ~{game_stats['pool_rtp']:.0%}.
        - System keeps {SYSTEM_FEE_RATE:.0%} as operational fee.
        """)

    # Recent History
    if game.history:
        st.divider()
        st.subheader("📝 Recent Rounds")
        for r in reversed(game.history[-10:]):
            outcome = f"🎊 WIN ${r['prize']:,.0f} ({r['multiplier']:.2f}×)" if r["won"] else "❌ LOSS $0"
            st.text(f"Round {r['round']}: {r['player']} bet ${r['bet']:,} → {outcome}")

    # Prize Distribution (effective)
    st.divider()
    st.subheader("🎁 Prize Distribution (Effective on Pay)")
    total_w = sum(PRIZE_WEIGHTS)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Small → Mid Prizes:**")
        for mult, weight in zip(PRIZE_MULTIPLIERS[0:10], PRIZE_WEIGHTS[0:10]):
            pct = (weight / total_w) * 100 if total_w else 0
            st.text(f"{mult*PRIZE_SCALE:.2f}×  —  {pct:.1f}% of pays")
    with col2:
        st.markdown("**Rare & Big Prizes:**")
        for mult, weight in zip(PRIZE_MULTIPLIERS[10:], PRIZE_WEIGHTS[10:]):
            pct = (weight / total_w) * 100 if total_w else 0
            st.text(f"{mult*PRIZE_SCALE:.2f}×  —  {pct:.1f}% of pays")

    # Reset button
    st.divider()
    if st.button("🔄 Reset Game", type="secondary"):
        game.reset_game()
        st.session_state.last_result = None
        st.success("Game has been reset!")
        st.rerun()

if __name__ == "__main__":
    main()
