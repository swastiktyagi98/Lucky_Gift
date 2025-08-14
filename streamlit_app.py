# organic_pool_game.py
"""
Organic Pool Game - Simple & Fair Approach

Key Features:
- Pure random outcomes (no artificial manipulation)
- 10% system fee on every bet
- Simple multiplier-based prizes
- Clean, minimal UI
- No complex heat/energy systems
- Truly organic gameplay
"""

import random
import streamlit as st
from typing import List, Dict, Tuple

# =======================
# Simple Configuration
# =======================

# System takes 10% of every bet
SYSTEM_FEE_RATE = 0.10

# Available bet amounts
BET_CHOICES = [100, 500, 1000, 2000, 5000, 10_000]

# Prize multipliers (what you can win)
PRIZE_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]

# Base win probability (fair 50/50 minus system fee)
BASE_WIN_PROBABILITY = 0.45  # Slightly below 50% to account for system fee

# Starting pool
STARTING_POOL = 100_000.0

# =======================
# Simple Game Class
# =======================

class OrganicPoolGame:
    """
    Simple, organic pool game with no artificial manipulation.
    Players bet, system takes 10%, winner is determined randomly.
    """
    
    def __init__(self):
        self.pool = STARTING_POOL
        self.total_bets = 0.0
        self.total_fees_collected = 0.0
        self.total_payouts = 0.0
        self.total_rounds = 0
        
        # Player stats
        self.players = {
            "Player 1": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0},
            "Player 2": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0},
            "Player 3": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0},
        }
        
        # Recent history
        self.history = []
    
    def play_round(self, player: str, bet_amount: float) -> Dict:
        """
        Play one round - completely organic and fair.
        """
        # System takes its fee first
        system_fee = bet_amount * SYSTEM_FEE_RATE
        effective_bet = bet_amount - system_fee
        
        # Add to pool
        self.pool += effective_bet
        
        # Update tracking
        self.total_bets += bet_amount
        self.total_fees_collected += system_fee
        self.total_rounds += 1
        
        # Update player stats
        self.players[player]["total_bet"] += bet_amount
        self.players[player]["rounds"] += 1
        
        # Determine if player wins (pure random)
        wins = random.random() < BASE_WIN_PROBABILITY
        
        prize = 0.0
        multiplier = 0.0
        
        if wins:
            # Random multiplier selection
            multiplier = random.choice(PRIZE_MULTIPLIERS)
            prize = bet_amount * multiplier
            
            # Pay out from pool
            if self.pool >= prize:
                self.pool -= prize
                self.players[player]["total_won"] += prize
                self.players[player]["wins"] += 1
                self.total_payouts += prize
            else:
                # Pool can't afford full prize, give what's available
                prize = max(0, self.pool)
                self.pool = 0
                self.players[player]["total_won"] += prize
                self.players[player]["wins"] += 1
                self.total_payouts += prize
                multiplier = prize / bet_amount if bet_amount > 0 else 0
        
        # Update player balance (total won - total bet)
        self.players[player]["balance"] = self.players[player]["total_won"] - self.players[player]["total_bet"]
        
        # Add to history
        result = {
            "round": self.total_rounds,
            "player": player,
            "bet": bet_amount,
            "system_fee": system_fee,
            "effective_bet": effective_bet,
            "won": wins,
            "prize": prize,
            "multiplier": multiplier,
            "pool_after": self.pool
        }
        
        self.history.append(result)
        if len(self.history) > 20:  # Keep last 20 rounds
            self.history.pop(0)
            
        return result
    
    def get_stats(self) -> Dict:
        """Get overall game statistics."""
        return {
            "total_bets": self.total_bets,
            "total_fees": self.total_fees_collected,
            "total_payouts": self.total_payouts,
            "net_pool_change": self.pool - STARTING_POOL,
            "total_rounds": self.total_rounds,
            "effective_rtp": (self.total_payouts / self.total_bets) if self.total_bets > 0 else 0
        }
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.__init__()

# =======================
# Simple Streamlit UI
# =======================

def main():
    st.set_page_config(
        page_title="Organic Pool Game", 
        page_icon="🎯", 
        layout="centered"
    )
    
    # Initialize game
    if "game" not in st.session_state:
        st.session_state.game = OrganicPoolGame()
    
    game = st.session_state.game
    
    # Header
    st.title("🎯 Organic Pool Game")
    st.caption("Simple, fair, and transparent gaming - System takes 10% fee from every bet")
    
    # Pool status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Pool", f"${game.pool:,.0f}")
    with col2:
        st.metric("🎮 Total Rounds", game.total_rounds)
    with col3:
        st.metric("🏦 System Fees", f"${game.total_fees_collected:,.0f}")
    
    st.divider()
    
    # Game Controls
    st.subheader("🎲 Play a Round")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_player = st.selectbox("Choose Player", list(game.players.keys()))
    with col2:
        selected_bet = st.selectbox("Bet Amount", BET_CHOICES)
    with col3:
        win_chance = BASE_WIN_PROBABILITY * 100
        st.metric("Win Chance", f"{win_chance:.0f}%")
    
    # Show what player could win
    min_prize = selected_bet * min(PRIZE_MULTIPLIERS)
    max_prize = selected_bet * max(PRIZE_MULTIPLIERS)
    st.info(f"💡 Possible prizes: ${min_prize:,.0f} to ${max_prize:,.0f} (System keeps ${selected_bet * SYSTEM_FEE_RATE:,.0f})")
    
    # Play button
    if st.button("🎲 PLAY ROUND", type="primary", use_container_width=True):
        result = game.play_round(selected_player, selected_bet)
        
        if result["won"]:
            if result["multiplier"] >= 5.0:
                st.balloons()
                st.success(f"🎉 BIG WIN! {selected_player} won ${result['prize']:,.0f} ({result['multiplier']:.1f}x multiplier)")
            else:
                st.success(f"✅ WIN! {selected_player} won ${result['prize']:,.0f} ({result['multiplier']:.1f}x multiplier)")
        else:
            st.error(f"❌ Loss. {selected_player} lost ${selected_bet:,.0f} (System fee: ${result['system_fee']:,.0f})")
        
        st.rerun()
    
    st.divider()
    
    # Player Statistics
    st.subheader("📊 Player Statistics")
    
    for player_name, stats in game.players.items():
        win_rate = (stats["wins"] / stats["rounds"]) if stats["rounds"] > 0 else 0
        
        # Color code balance
        balance_color = "green" if stats["balance"] >= 0 else "red"
        
        with st.expander(f"**{player_name}** - Balance: ${stats['balance']:+,.0f}", 
                        expanded=(player_name == selected_player)):
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Bet", f"${stats['total_bet']:,.0f}")
                st.metric("Rounds Played", stats["rounds"])
            with col2:
                st.metric("Total Won", f"${stats['total_won']:,.0f}")
                st.metric("Wins", f"{stats['wins']}")
            with col3:
                st.markdown(f"**Balance:** <span style='color:{balance_color}'>${stats['balance']:+,.0f}</span>", 
                           unsafe_allow_html=True)
                st.metric("Win Rate", f"{win_rate:.1%}")
    
    # Game Statistics
    st.divider()
    st.subheader("🎯 Game Statistics")
    
    game_stats = game.get_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Bets Placed", f"${game_stats['total_bets']:,.0f}")
        st.metric("Total Payouts", f"${game_stats['total_payouts']:,.0f}")
        st.metric("Effective RTP", f"{game_stats['effective_rtp']:.1%}")
    
    with col2:
        st.metric("System Fees Collected", f"${game_stats['total_fees']:,.0f}")
        st.metric("Pool Net Change", f"${game_stats['net_pool_change']:+,.0f}")
        theoretical_rtp = (1 - SYSTEM_FEE_RATE) * BASE_WIN_PROBABILITY + (1 - BASE_WIN_PROBABILITY) * 0
        avg_multiplier = sum(PRIZE_MULTIPLIERS) / len(PRIZE_MULTIPLIERS)
        theoretical_rtp_adj = BASE_WIN_PROBABILITY * avg_multiplier
        st.metric("Expected RTP", f"≈{theoretical_rtp_adj:.1%}")
    
    # Recent History
    if game.history:
        st.divider()
        st.subheader("📝 Recent Rounds")
        
        for round_info in reversed(game.history[-10:]):  # Last 10 rounds
            outcome = "🎉 WIN" if round_info["won"] else "❌ LOSS"
            multiplier_text = f" ({round_info['multiplier']:.1f}x)" if round_info["won"] else ""
            
            st.text(f"Round {round_info['round']}: {round_info['player']} bet ${round_info['bet']:,} → "
                   f"{outcome} ${round_info['prize']:,.0f}{multiplier_text}")
    
    # Reset button
    st.divider()
    if st.button("🔄 Reset Game", type="secondary"):
        game.reset_game()
        st.success("Game has been reset!")
        st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    **How it works:**
    - System takes 10% fee from every bet
    - Remaining 90% goes to the pool
    - 45% chance to win (fair odds considering system fee)
    - Winners get random multiplier from the prize list
    - No artificial manipulation - purely organic gameplay
    """)

if __name__ == "__main__":
    main()
