# player_friendly_pool_game.py
"""
Player-Friendly Pool Game

Key updates in this version:
- Rare 0× (no-prize) remains for pool safety, but is treated as a LOSS by default (cleaner UX).
- Separates "Hit Rate" (landed on prize wheel) from "Cash Win Rate" (payout > 0).
- Keeps pool profitability unchanged.
- Persists and shows the latest round result just under the PLAY button.
"""

import random
import streamlit as st
from typing import Dict

# =======================
# Configuration
# =======================

# System takes 10% of every bet
SYSTEM_FEE_RATE = 0.10

# Available bet amounts
BET_CHOICES = [100, 500, 1000, 2000, 5000, 10_000]

# Base chance to enter the prize wheel ("hit chance")
WIN_PROBABILITY = 0.85

# Whether a 0× outcome should count as a "win" in stats/UX
# Default False => 0× is shown and counted as a LOSS for clarity.
COUNT_ZERO_AS_WIN = False

# Prize multipliers — includes a rare 0× outcome for pool safety
PRIZE_MULTIPLIERS = [
    0.0,                 # Very rare: no prize
    0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9,  # Small prizes (7)
    1.0, 1.05,                            # Near break-even (2)
    1.2, 1.5, 2.0                         # Profit prizes (3)
]

# Weights for prize selection (must match multipliers length)
# Sum = 103. 0× has weight 1 (~0.97% of "hit" outcomes).
PRIZE_WEIGHTS = [
    1,      # 0.0× (very rare)
    8, 8, 8, 8, 8, 8, 8,   # small prizes
    6, 6,                  # near break-even
    14, 11, 9              # profit prizes
]

# Starting pool
STARTING_POOL = 100_000.0

# =======================
# Smart Pool Game Class
# =======================

class PlayerFriendlyPoolGame:
    """
    Game designed to give players frequent hits while protecting the pool.
    Pool always receives more than it pays out on average.
    """
    def __init__(self):
        self.pool = STARTING_POOL
        self.total_bets = 0.0
        self.total_fees_collected = 0.0
        self.total_payouts = 0.0
        self.total_rounds = 0

        # Player stats
        self.players = {
            "Player 1": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0, "hits": 0},
            "Player 2": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0, "hits": 0},
            "Player 3": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0, "hits": 0},
        }

        # Recent history
        self.history = []

    def select_prize_multiplier(self) -> float:
        """Weighted random selection."""
        if len(PRIZE_MULTIPLIERS) != len(PRIZE_WEIGHTS):
            return random.choice(PRIZE_MULTIPLIERS)
        return random.choices(PRIZE_MULTIPLIERS, weights=PRIZE_WEIGHTS)[0]

    def play_round(self, player: str, bet_amount: float) -> Dict:
        """Play one round with a high hit probability and pool-protective prizes."""
        # System fee and effective bet to pool
        system_fee = bet_amount * SYSTEM_FEE_RATE
        effective_bet = bet_amount - system_fee
        self.pool += effective_bet

        # Tracking
        self.total_bets += bet_amount
        self.total_fees_collected += system_fee
        self.total_rounds += 1

        # Player tracking
        p = self.players[player]
        p["total_bet"] += bet_amount
        p["rounds"] += 1

        # Determine if round hits the prize wheel
        hits = random.random() < WIN_PROBABILITY

        prize = 0.0
        multiplier = 0.0
        won_cash = False  # payout > 0

        if hits:
            p["hits"] += 1
            multiplier = self.select_prize_multiplier()
            prize = bet_amount * multiplier

            if prize > 0:
                # Normal cash win
                if self.pool >= prize:
                    self.pool -= prize
                else:
                    # emergency fallback: half of remaining pool
                    prize = max(0, self.pool * 0.5)
                    self.pool -= prize
                    multiplier = (prize / bet_amount) if bet_amount > 0 else 0

                p["total_won"] += prize
                p["wins"] += 1
                self.total_payouts += prize
                won_cash = True
            else:
                # 0× outcome: treat as LOSS unless COUNT_ZERO_AS_WIN is True
                if COUNT_ZERO_AS_WIN:
                    # Counts as a "win" for stats (no payout)
                    p["wins"] += 1
                    won_cash = False  # but prize stays 0

        # Update balance
        p["balance"] = p["total_won"] - p["total_bet"]

        # Compose result
        # "won" means cash was paid out (unless flag counts 0× as win)
        won_flag = (won_cash or (COUNT_ZERO_AS_WIN and hits and multiplier == 0.0))

        result = {
            "round": self.total_rounds,
            "player": player,
            "bet": bet_amount,
            "system_fee": system_fee,
            "effective_bet": effective_bet,
            "hit": hits,              # landed on prize wheel
            "won": won_flag,          # UX win flag
            "cash_win": won_cash,     # strictly prize > 0
            "prize": prize,
            "multiplier": multiplier,
            "pool_after": self.pool
        }

        self.history.append(result)
        if len(self.history) > 50:
            self.history.pop(0)

        return result

    def get_stats(self) -> Dict:
        """Game statistics including pool profitability."""
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
        """Reset the game to initial state."""
        self.__init__()

# =======================
# Helpers
# =======================

def weighted_avg_multiplier() -> float:
    total_w = sum(PRIZE_WEIGHTS)
    return sum(m * w for m, w in zip(PRIZE_MULTIPLIERS, PRIZE_WEIGHTS)) / total_w if total_w else 0.0

def zero_prize_rates():
    """Returns (prob_of_zero_given_hit, prob_of_zero_overall)."""
    total_w = sum(PRIZE_WEIGHTS)
    p_zero_given_hit = (PRIZE_WEIGHTS[0] / total_w) if total_w else 0.0
    p_zero_overall = WIN_PROBABILITY * p_zero_given_hit
    return p_zero_given_hit, p_zero_overall

# =======================
# Streamlit UI
# =======================

def main():
    st.set_page_config(page_title="Player-Friendly Pool Game", page_icon="🎊", layout="centered")

    # Initialize game
    if "game" not in st.session_state:
        st.session_state.game = PlayerFriendlyPoolGame()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    game = st.session_state.game

    # Header
    st.title("🎊 Player-Friendly Pool Game")
    caption_flag = "0× counts as LOSS" if not COUNT_ZERO_AS_WIN else "0× counts as WIN (no payout)"
    st.caption(f"High hit rate ({WIN_PROBABILITY:.0%}) with pool-safe prizes — {caption_flag}.")

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
        st.metric("Hit Chance", f"{WIN_PROBABILITY:.0%}", help="Probability to land on the prize wheel")

    # Prize info
    avg_mult = weighted_avg_multiplier()
    min_prize = selected_bet * min(PRIZE_MULTIPLIERS)
    max_prize = selected_bet * max(PRIZE_MULTIPLIERS)
    avg_prize = selected_bet * avg_mult
    p0_hit, p0_overall = zero_prize_rates()

    st.info(
        f"💡 Prize range: ${min_prize:,.0f} to ${max_prize:,.0f} | "
        f"Avg prize on hit: ~${avg_prize:,.0f} | "
        f"System fee: ${selected_bet * SYSTEM_FEE_RATE:,.0f} | "
        f"0× chance: ~{p0_hit*100:.1f}% of hits (~{p0_overall*100:.1f}% overall)"
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
        if result["cash_win"]:
            if result["multiplier"] >= 1.2:
                st.balloons()
                st.success(f"🎉 BIG WIN! {result['player']} won ${result['prize']:,.0f} ({result['multiplier']:.2f}×)")
            elif result["multiplier"] >= 0.8:
                st.success(f"✅ GOOD WIN! {result['player']} won ${result['prize']:,.0f} ({result['multiplier']:.2f}×)")
            else:
                st.success(f"🎊 WIN! {result['player']} won ${result['prize']:,.0f} ({result['multiplier']:.2f}×)")
        else:
            # Unified $0 branch (loss), including rare 0×
            if result["hit"] and result["multiplier"] == 0.0:
                st.warning(f"🎭 Miss on wheel (0×) — no prize. {result['player']} lost ${result['bet']:,.0f}.")
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
                mult_label = f"{result['multiplier']:.2f}×" if result["hit"] else "—"
                st.metric("Multiplier", mult_label)
                st.metric("Pool After", f"${result['pool_after']:,.0f}")

    st.divider()

    # Player Statistics
    st.subheader("📊 Player Statistics")

    for player_name, stats in game.players.items():
        # Cash win rate (payout > 0)
        cash_win_rate = (stats["wins"] / stats["rounds"]) if stats["rounds"] > 0 else 0
        hit_rate = (stats["hits"] / stats["rounds"]) if stats["rounds"] > 0 else 0
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
                st.metric("Hit Rate", f"{hit_rate:.1%}", help="Landed on prize wheel")
                st.metric("Cash Win Rate", f"{cash_win_rate:.1%}", help="Rounds with payout > 0")

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

    # Why the pool stays profitable
    if game_stats['total_rounds'] > 0:
        st.info(f"""
        **Why the pool stays profitable:**
        - High hit chance ({WIN_PROBABILITY:.0%}) keeps play exciting.
        - Weighted average prize on hits ≈ {weighted_avg_multiplier():.2f}× of bet.
        - Pool receives {(1-SYSTEM_FEE_RATE)*100:.0f}% of bets, pays out ~{game_stats['pool_rtp']:.0%}.
        - Includes a very rare 0× outcome to reduce expected payouts without feeling common.
        - System keeps {SYSTEM_FEE_RATE:.0%} as operational fee.
        """)

    # Recent History
    if game.history:
        st.divider()
        st.subheader("📝 Recent Rounds")
        for r in reversed(game.history[-10:]):
            if r["cash_win"]:
                outcome = f"🎊 WIN ${r['prize']:,.0f} ({r['multiplier']:.2f}×)"
            else:
                if r["hit"] and r["multiplier"] == 0.0:
                    outcome = "🎭 Miss on wheel (0×) — LOSS $0"
                else:
                    outcome = "❌ LOSS $0"
            st.text(f"Round {r['round']}: {r['player']} bet ${r['bet']:,} → {outcome}")

    # Prize Distribution Info (true weighted odds)
    st.divider()
    st.subheader("🎁 Prize Distribution (Odds on Hit)")
    total_w = sum(PRIZE_WEIGHTS)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Small → Mid Prizes:**")
        for mult, weight in zip(PRIZE_MULTIPLIERS[1:10], PRIZE_WEIGHTS[1:10]):
            pct = (weight / total_w) * 100 if total_w else 0
            st.text(f"{mult:.2f}×  —  {pct:.1f}% of hits")
    with col2:
        st.markdown("**Rare & Big Prizes (incl. 0×):**")
        for mult, weight in [(PRIZE_MULTIPLIERS[0], PRIZE_WEIGHTS[0])] + list(zip(PRIZE_MULTIPLIERS[10:], PRIZE_WEIGHTS[10:])):
            label = "0.00× (no prize)" if mult == 0.0 else f"{mult:.2f}×"
            pct = (weight / total_w) * 100 if total_w else 0
            st.text(f"{label}  —  {pct:.1f}% of hits")

    # Reset button
    st.divider()
    if st.button("🔄 Reset Game", type="secondary"):
        game.reset_game()
        st.session_state.last_result = None
        st.success("Game has been reset!")
        st.rerun()

if __name__ == "__main__":
    main()
