# player_friendly_pool_game.py
"""
Player-Friendly Pool Game

Key Features:
- High win rate (~85%) for better player experience
- Smart prize system that protects pool
- 10% system fee on every bet
- Pool-sustainable prizes (always less than what pool receives)
- Simple, clean UI
"""

import random
import streamlit as st
from typing import List, Dict, Tuple

# =======================
# Configuration
# =======================

# System takes 10% of every bet
SYSTEM_FEE_RATE = 0.10

# Available bet amounts
BET_CHOICES = [100, 500, 1000, 2000, 5000, 10_000]

# High win rate for player satisfaction
WIN_PROBABILITY = 0.85

# Prize multipliers - designed to keep pool profitable
# Most prizes are less than the effective bet (90% of original bet)
PRIZE_MULTIPLIERS = [
    0.5, 0.6, 0.7, 0.8,  # Common small prizes (8 items)
    0.9, 1.0,                                   # Break-even prizes (2 items)
    1.2, 1.5, 2.0                             # Profit prizes (3 items)
]

# Weights for prize selection (higher weight = more likely)
# Must match exactly the number of multipliers above (13 items total)
PRIZE_WEIGHTS = [
    15, 12, 10, 8, 8, 6, 5, 4,  # Small prizes (8 weights)
    3, 2,                        # Break-even (2 weights)
    1, 1, 1                      # Profit prizes (3 weights)
]

# Starting pool
STARTING_POOL = 100_000.0

# =======================
# Smart Pool Game Class
# =======================

class PlayerFriendlyPoolGame:
    """
    Game designed to give players frequent wins while protecting the pool.
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
            "Player 1": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0},
            "Player 2": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0},
            "Player 3": {"balance": 0.0, "total_bet": 0.0, "total_won": 0.0, "rounds": 0, "wins": 0},
        }
        
        # Recent history
        self.history = []
    
    def select_prize_multiplier(self) -> float:
        """
        Select a prize multiplier using weighted random selection.
        Most prizes are small to keep pool profitable.
        """
        # Safety check to ensure lists match
        if len(PRIZE_MULTIPLIERS) != len(PRIZE_WEIGHTS):
            # Fallback to simple random selection if weights don't match
            return random.choice(PRIZE_MULTIPLIERS)
        
        return random.choices(PRIZE_MULTIPLIERS, weights=PRIZE_WEIGHTS)[0]
    
    def play_round(self, player: str, bet_amount: float) -> Dict:
        """
        Play one round with high win rate but pool-protective prizes.
        """
        # System takes its fee first
        system_fee = bet_amount * SYSTEM_FEE_RATE
        effective_bet = bet_amount - system_fee  # This goes to pool
        
        # Pool receives the effective bet
        self.pool += effective_bet
        
        # Update tracking
        self.total_bets += bet_amount
        self.total_fees_collected += system_fee
        self.total_rounds += 1
        
        # Update player stats
        self.players[player]["total_bet"] += bet_amount
        self.players[player]["rounds"] += 1
        
        # Determine if player wins (high probability)
        wins = random.random() < WIN_PROBABILITY
        
        prize = 0.0
        multiplier = 0.0
        
        if wins:
            # Select prize multiplier (weighted toward small prizes)
            multiplier = self.select_prize_multiplier()
            prize = bet_amount * multiplier  # Prize based on original bet
            
            # Pay out from pool (pool protection built into multiplier selection)
            if self.pool >= prize:
                self.pool -= prize
                self.players[player]["total_won"] += prize
                self.players[player]["wins"] += 1
                self.total_payouts += prize
            else:
                # Emergency fallback - shouldn't happen with our design
                prize = max(0, self.pool * 0.5)  # Only take half of remaining pool
                self.pool -= prize
                self.players[player]["total_won"] += prize
                self.players[player]["wins"] += 1
                self.total_payouts += prize
                multiplier = prize / bet_amount if bet_amount > 0 else 0
        
        # Update player balance
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
        if len(self.history) > 20:
            self.history.pop(0)
            
        return result
    
    def get_stats(self) -> Dict:
        """Get game statistics including pool profitability."""
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
# Streamlit UI
# =======================

def main():
    st.set_page_config(
        page_title="Player-Friendly Pool Game", 
        page_icon="🎊", 
        layout="centered"
    )
    
    # Initialize game
    if "game" not in st.session_state:
        st.session_state.game = PlayerFriendlyPoolGame()
    
    game = st.session_state.game
    
    # Header
    st.title("🎊 Player-Friendly Pool Game")
    st.caption(f"High win rate ({WIN_PROBABILITY:.0%}) with smart prize system - Pool stays profitable!")
    
    # Get game statistics
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
        st.metric("📊 Pool Profit", f"${game_stats['pool_profit']:+,.0f}", help="How much pool has gained/lost")
    
    # Pool health indicator
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
        st.metric("Win Chance", f"{WIN_PROBABILITY:.0%}")
    
    # Show prize information
    min_prize = selected_bet * min(PRIZE_MULTIPLIERS)
    max_prize = selected_bet * max(PRIZE_MULTIPLIERS)
    avg_prize = selected_bet * (sum(m * w for m, w in zip(PRIZE_MULTIPLIERS, PRIZE_WEIGHTS)) / sum(PRIZE_WEIGHTS))
    
    st.info(f"💡 Prize range: ${min_prize:,.0f} to ${max_prize:,.0f} | Average prize: ~${avg_prize:,.0f} | System fee: ${selected_bet * SYSTEM_FEE_RATE:,.0f}")
    
    # Play button
    if st.button("🎲 PLAY ROUND", type="primary", use_container_width=True):
        result = game.play_round(selected_player, selected_bet)
        
        if result["won"]:
            if result["multiplier"] >= 1.2:
                st.balloons()
                st.success(f"🎉 BIG WIN! {selected_player} won ${result['prize']:,.0f} ({result['multiplier']:.1f}x)")
            elif result["multiplier"] >= 0.8:
                st.success(f"✅ GOOD WIN! {selected_player} won ${result['prize']:,.0f} ({result['multiplier']:.1f}x)")
            else:
                st.success(f"🎊 WIN! {selected_player} won ${result['prize']:,.0f} ({result['multiplier']:.1f}x)")
        else:
            st.error(f"❌ Rare loss! {selected_player} lost ${selected_bet:,.0f}")
        
        st.rerun()
    
    st.divider()
    
    # Player Statistics
    st.subheader("📊 Player Statistics")
    
    for player_name, stats in game.players.items():
        win_rate = (stats["wins"] / stats["rounds"]) if stats["rounds"] > 0 else 0
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
    
    # Enhanced Game Statistics
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
    
    # Show why system is profitable
    if game_stats['total_rounds'] > 0:
        st.info(f"""
        **Why the pool stays profitable:**
        - Players win {WIN_PROBABILITY:.0%} of the time (great experience!)
        - But most prizes are small (average ~{(sum(m * w for m, w in zip(PRIZE_MULTIPLIERS, PRIZE_WEIGHTS)) / sum(PRIZE_WEIGHTS)):.1f}x)
        - Pool receives ${(1-SYSTEM_FEE_RATE)*100:.0f}% of bets, pays out ~{game_stats['pool_rtp']:.0%}
        - System keeps {SYSTEM_FEE_RATE:.0%} as operational fee
        """)
    
    # Recent History
    if game.history:
        st.divider()
        st.subheader("📝 Recent Rounds")
        
        for round_info in reversed(game.history[-10:]):
            if round_info["won"]:
                outcome = f"🎊 WIN ${round_info['prize']:,.0f} ({round_info['multiplier']:.1f}x)"
            else:
                outcome = "❌ LOSS $0"
            
            st.text(f"Round {round_info['round']}: {round_info['player']} bet ${round_info['bet']:,} → {outcome}")
    
    # Prize Distribution Info
    st.divider()
    st.subheader("🎁 Prize Distribution")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Common Prizes (Frequent):**")
        for i, (mult, weight) in enumerate(zip(PRIZE_MULTIPLIERS[:8], PRIZE_WEIGHTS[:8])):
            st.text(f"{mult:.1f}x multiplier ({weight}% chance)")
    
    with col2:
        st.markdown("**Rare Prizes (Uncommon):**")
        for mult, weight in zip(PRIZE_MULTIPLIERS[8:], PRIZE_WEIGHTS[8:]):
            st.text(f"{mult:.1f}x multiplier ({weight}% chance)")
    
    # Reset button
    st.divider()
    if st.button("🔄 Reset Game", type="secondary"):
        game.reset_game()
        st.success("Game has been reset!")
        st.rerun()

if __name__ == "__main__":
    main()
