# improved_stable_pool_game.py
"""
Improved Stable Pool Economy — Streamlit Only Version

Key Improvements:
- Pool protection: Cannot go below -100,000
- Enhanced UI with better visualization
- Streamlined code (removed CLI and test components)
- Better error handling and validation
- Improved player statistics display
- Real-time pool health monitoring
- Enhanced prize multiplier management
"""

import random
import streamlit as st
from math import exp
from typing import Dict, Tuple, Optional, List

# =======================
# Config / Constants
# =======================
E_SCALE = 300.0
P_SCALE = 50_000.0
POOL_MIN_LIMIT = -100_000.0  # Pool cannot go below this

# RTP & probability control
RTP_TARGET = 0.98
RTP_SENSITIVITY = 0.25
BASE_P = 0.48

# Prize skew — higher => more small wins
PRIZE_SKEW_BASE = 3.0

# Fees
HOUSE_FEE = 0.06
WIN_FEE_THRESHOLD_MULT = 1.2

# Bets menu
BET_CHOICES = [100, 500, 1000, 2000, 5000, 10_000, 20_000]

# Cosmetic: near-max counts as "jackpot" in labeling
JACKPOT_NEAR_MAX_THRESHOLD = 0.95

# Default discrete prize multipliers (editable in UI)
DEFAULT_PRIZE_MULTIPLIERS: List[float] = [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

# =======================
# Core Game Class
# =======================
class StablePoolGame:
    """
    Enhanced stable pool economy game with pool protection and improved features.
    """

    def __init__(self, starting_pool: float = 50_000.0, prize_multipliers: Optional[List[float]] = None):
        self.starting_pool: float = float(starting_pool)
        self.pool: float = float(starting_pool)
        self.players: Dict[str, Dict[str, float]] = {}
        self.total_rounds: int = 0
        self.total_house_fees: float = 0.0
        self.total_pool_contributions: float = 0.0
        self.prize_multipliers: List[float] = self._normalize_multipliers(
            prize_multipliers or DEFAULT_PRIZE_MULTIPLIERS
        )
        self.game_history: List[Dict] = []  # Track recent plays for history

    def add_player(self, name: str) -> None:
        """Add a new player to the game."""
        if name not in self.players:
            self.players[name] = {
                "energy": 0.0,        # Rewards - Bets (display)
                "heat": 0.0,          # Bets - Rewards (control)
                "total_rewards": 0.0,
                "total_bets": 0.0,
                "wins": 0.0,
                "total_rounds": 0.0,
                "biggest_win": 0.0,
                "biggest_loss": 0.0,
            }

    def ensure_player(self, name: str) -> None:
        """Ensure player exists, create if not."""
        if name not in self.players:
            self.add_player(name)

    @staticmethod
    def _exp01_pos(x: float, scale: float) -> float:
        """Exponential activation on positive part only (0..1)."""
        return 1.0 - exp(-max(0.0, x) / max(1.0, scale))

    def heat_activation(self, heat: float) -> float:
        """Calculate activation based on player's heat (losses)."""
        return self._exp01_pos(heat, E_SCALE)

    def pool_activation(self, pool_balance: float) -> float:
        """Calculate activation based on pool health."""
        return self._exp01_pos(pool_balance, P_SCALE)

    @staticmethod
    def _normalize_multipliers(multis: List[float]) -> List[float]:
        """Clean and normalize multiplier list."""
        clean = sorted({m for m in multis if isinstance(m, (int, float)) and m >= 0.0})
        return clean or [0.0]

    def set_prize_multipliers(self, multis: List[float]) -> None:
        """Update prize multipliers list."""
        self.prize_multipliers = self._normalize_multipliers(multis)

    def _nearest_multiplier(self, ratio: float, up_to_one_only: bool = False) -> float:
        """Find nearest multiplier to given ratio."""
        if up_to_one_only:
            pool = [m for m in self.prize_multipliers if m <= 1.0]
            if not pool:
                pool = [0.0]
        else:
            pool = self.prize_multipliers
        return min(pool, key=lambda m: abs(m - ratio))

    def pool_can_afford(self, amount: float) -> bool:
        """Check if pool can afford a payout without going below limit."""
        return (self.pool - amount) >= POOL_MIN_LIMIT

    def rtp_controller(self) -> Dict[str, float]:
        """Calculate RTP control parameters."""
        total_bets = sum(p["total_bets"] for p in self.players.values())
        total_rewards = sum(p["total_rewards"] for p in self.players.values())
        rtp = (total_rewards / total_bets) if total_bets > 0 else RTP_TARGET

        delta = rtp - RTP_TARGET
        adj = delta * RTP_SENSITIVITY

        p_scale = max(0.1, min(2.0, 1.0 - adj))  # Keep reasonable bounds
        cons_scale = max(0.1, min(2.0, 1.0 - 0.7 * adj))
        
        skew = PRIZE_SKEW_BASE + (1.0 - self.pool_activation(self.pool)) * 1.5 + max(0.0, delta) * 5.0
        
        return {
            "rtp": rtp,
            "p_scale": p_scale,
            "cons_scale": cons_scale,
            "skew": skew,
            "delta": delta
        }

    def win_probability(self, heat: float, p_scale: float) -> float:
        """Calculate win probability based on heat and pool state."""
        heat_boost = self.heat_activation(heat) * 0.08
        pool_adjustment = (self.pool_activation(self.pool) - 0.5) * 0.06
        p = (BASE_P + heat_boost + pool_adjustment) * p_scale
        return max(0.01, min(0.99, p))  # Keep within reasonable bounds

    def prize_bounds(self, bet: float) -> Tuple[float, float]:
        """Get minimum and maximum possible prizes."""
        lo = min(self.prize_multipliers) * bet
        hi = max(self.prize_multipliers) * bet
        return lo, hi

    def sample_win_prize(self, bet: float, skew: float, rng: Optional[random.Random] = None) -> Tuple[float, bool]:
        """Sample a win prize from the discrete multiplier list."""
        r = rng if rng is not None else random
        multis = self.prize_multipliers
        n = len(multis)
        if n == 0:
            return 0.0, False
        
        u = (r.random()) ** max(1.0, skew)
        idx = min(int(u * n), n - 1)
        mult = multis[idx]
        prize = bet * mult
        
        # Check if pool can afford this prize
        if not self.pool_can_afford(prize):
            # Find largest affordable prize
            affordable_multis = [m for m in multis if self.pool_can_afford(bet * m)]
            if affordable_multis:
                mult = max(affordable_multis)
                prize = bet * mult
            else:
                mult = 0.0
                prize = 0.0
        
        is_jp = (mult >= JACKPOT_NEAR_MAX_THRESHOLD * max(multis))
        return prize, is_jp

    def consolation(self, bet: float, heat: float, cons_scale: float, rng: Optional[random.Random] = None) -> float:
        """Calculate consolation payout for losses."""
        r = rng if rng is not None else random
        heat_fac = self.heat_activation(heat)
        base_pct = 0.05 + 0.30 * heat_fac
        pool_mult = 0.5 + 0.5 * self.pool_activation(self.pool)
        max_cons = bet * base_pct * pool_mult * cons_scale
        raw = (r.random() ** 1.5) * max_cons
        
        # Ensure pool can afford consolation
        if not self.pool_can_afford(raw):
            available = max(0.0, self.pool - POOL_MIN_LIMIT)
            raw = min(raw, available)
        
        return max(0.0, raw)

    def play_round(self, player: str, bet: float, rng: Optional[random.Random] = None) -> Dict:
        """Play one round of the game."""
        self.ensure_player(player)
        r = rng if rng is not None else random

        # Pool takes the bet
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
            gross = proposed
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
            
            # Track biggest win
            if payout > self.players[player]["biggest_win"]:
                self.players[player]["biggest_win"] = payout
                
            final_mult = payout / bet if bet else 0.0
        else:
            # Loss handling with quantization
            base_cons = self.consolation(bet, heat, ctrl["cons_scale"], r)
            heat_fac = self.heat_activation(heat)
            target_rate = 0.30 + 0.20 * heat_fac
            target_amt = bet * target_rate
            
            if base_cons < target_amt:
                contribution_needed = (target_amt - base_cons) * ctrl["cons_scale"]
                contribution_paid = min(contribution_needed, max(0.0, self.pool - POOL_MIN_LIMIT))
            
            payout = base_cons + contribution_paid

            # Quantize loss payout to nearest multiplier ≤ 1.0
            raw_ratio = (payout / bet) if bet else 0.0
            quant_mult = self._nearest_multiplier(raw_ratio, up_to_one_only=True)
            quant_payout = bet * quant_mult
            
            # Ensure quantized payout is affordable
            if not self.pool_can_afford(quant_payout):
                available = max(0.0, self.pool - POOL_MIN_LIMIT)
                quant_payout = min(quant_payout, available)
                quant_mult = quant_payout / bet if bet else 0.0

            # Adjust components
            if quant_payout >= base_cons:
                contribution_paid = quant_payout - base_cons
                consolation_amt = base_cons
            else:
                contribution_paid = 0.0
                consolation_amt = quant_payout
            
            payout = quant_payout
            self.pool -= payout
            self.players[player]["total_rewards"] += payout
            
            # Track biggest loss (as negative energy change)
            loss_amount = bet - payout
            if loss_amount > self.players[player]["biggest_loss"]:
                self.players[player]["biggest_loss"] = loss_amount
                
            final_mult = payout / bet if bet else 0.0

        # Update aggregates
        self.total_rounds += 1
        self.total_house_fees += fee_retained
        self.total_pool_contributions += contribution_paid
        self.players[player]["total_bets"] += bet
        self.players[player]["total_rounds"] += 1

        # Update Energy & Heat
        tb = self.players[player]["total_bets"]
        tr = self.players[player]["total_rewards"]
        self.players[player]["energy"] = tr - tb
        self.players[player]["heat"] = tb - tr

        # Add to history
        result = {
            "round": self.total_rounds,
            "player": player,
            "bet": bet,
            "won": won,
            "p_win": p_win,
            "prize": prize,
            "consolation": consolation_amt,
            "contribution_paid": contribution_paid,
            "fee_retained": fee_retained,
            "is_jackpot": is_jackpot,
            "player_received": payout,
            "multiplier": final_mult,
            "pool_after": self.pool
        }
        
        self.game_history.append(result)
        if len(self.game_history) > 50:  # Keep last 50 rounds
            self.game_history.pop(0)

        return result

    def totals(self) -> Tuple[float, float]:
        """Get total bets and rewards across all players."""
        total_bets = sum(p["total_bets"] for p in self.players.values())
        total_rewards = sum(p["total_rewards"] for p in self.players.values())
        return total_bets, total_rewards

    def rtp(self) -> float:
        """Calculate current RTP."""
        tb, tr = self.totals()
        return (tr / tb) if tb > 0 else 0.0

    def pool_health(self) -> float:
        """Calculate pool health score."""
        normalized_pool = (self.pool - POOL_MIN_LIMIT) / (self.starting_pool - POOL_MIN_LIMIT)
        return max(0.0, min(1.0, normalized_pool))

    def pool_health_status(self) -> Tuple[str, str]:
        """Get pool health status and color."""
        health = self.pool_health()
        if health > 0.8:
            return "🟢 Excellent", "green"
        elif health > 0.6:
            return "🟡 Good", "orange"
        elif health > 0.4:
            return "🟠 Moderate", "orange"
        elif health > 0.2:
            return "🔴 Low", "red"
        else:
            return "🔴 Critical", "red"

    def reset_game(self, keep_multipliers: bool = True) -> None:
        """Reset the game to initial state."""
        multipliers = self.prize_multipliers if keep_multipliers else DEFAULT_PRIZE_MULTIPLIERS
        self.__init__(self.starting_pool, multipliers)


# =======================
# Utility Functions
# =======================
def parse_multiplier_csv(s: str) -> List[float]:
    """Parse comma-separated multiplier string."""
    try:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        vals = [float(p) for p in parts]
        uniq = sorted({v for v in vals if v >= 0.0})
        return uniq or DEFAULT_PRIZE_MULTIPLIERS
    except Exception:
        return DEFAULT_PRIZE_MULTIPLIERS


# =======================
# Streamlit UI
# =======================
def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Stable Pool Economy",
        page_icon="🎲",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize game
    if "game" not in st.session_state:
        st.session_state.game = StablePoolGame(prize_multipliers=DEFAULT_PRIZE_MULTIPLIERS)
        for name in ("UserA", "UserB", "UserC"):
            st.session_state.game.add_player(name)

    game: StablePoolGame = st.session_state.game

    # Header
    st.title("🎲 Stable Pool Economy")
    st.caption("Enhanced gaming experience with pool protection and discrete multipliers")

    # Pool status
    status, color = game.pool_health_status()
    health = game.pool_health()
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.metric("💰 Pool Balance", f"{game.pool:,.0f}", help=f"Health: {health:.1%}")
    with col2:
        st.metric("🎯 RTP", f"{game.rtp():.2%}", f"{((game.rtp() - RTP_TARGET) * 100):+.2f}%")
    with col3:
        st.metric("🎮 Total Rounds", f"{game.total_rounds:,}")
    with col4:
        st.metric("🏦 House Fees", f"{game.total_house_fees:,.0f}")

    # Pool health indicator
    st.markdown(f"**Pool Status:** {status}")
    progress_val = max(0.0, min(1.0, health))
    st.progress(progress_val)

    if game.pool <= POOL_MIN_LIMIT + 10000:  # Warning when close to limit
        st.warning(f"⚠️ Pool is near the minimum limit of {POOL_MIN_LIMIT:,}!")

    # Sidebar - Game Controls
    with st.sidebar:
        st.header("🎮 Game Controls")
        
        # Player selection
        user_choice = st.selectbox("Select Player", list(game.players.keys()))
        
        # Bet selection
        bet_choice = st.selectbox("Select Bet Amount", BET_CHOICES, index=2)
        
        # Game info for selected player
        player = game.players[user_choice]
        heat = player["heat"]
        ctrl = game.rtp_controller()
        p_win = game.win_probability(heat, ctrl["p_scale"])
        lo, hi = game.prize_bounds(bet_choice)
        
        st.markdown("### 📊 Current Stats")
        st.markdown(f"**Win Probability:** {p_win:.1%}")
        st.markdown(f"**Prize Range:** {lo:,.0f} - {hi:,.0f}")
        st.markdown(f"**Player Energy:** {player['energy']:,.0f}")
        st.markdown(f"**Player Heat:** {heat:,.0f}")
        
        # Play button
        if st.button("🎲 Play Round", type="primary", use_container_width=True):
            result = game.play_round(user_choice, bet_choice)
            
            if result["won"]:
                if result["is_jackpot"]:
                    st.balloons()
                    st.success(f"🎰 JACKPOT! Won {result['player_received']:,.0f} ({result['multiplier']:.2f}×)")
                else:
                    st.success(f"🎉 WIN! Received {result['player_received']:,.0f} ({result['multiplier']:.2f}×)")
            else:
                st.info(f"💔 Loss. Received {result['player_received']:,.0f} ({result['multiplier']:.2f}×)")
            
            st.rerun()
        
        st.divider()
        
        # Multiplier editor
        st.header("⚙️ Prize Multipliers")
        current_mult_str = ",".join(str(m) for m in game.prize_multipliers)
        new_mult_str = st.text_area(
            "Edit Multipliers (comma-separated)",
            value=current_mult_str,
            help="Example: 0,0.2,0.5,0.8,1,1.5,2,3,5"
        )
        
        if st.button("Update Multipliers"):
            new_multipliers = parse_multiplier_csv(new_mult_str)
            game.set_prize_multipliers(new_multipliers)
            st.success("Multipliers updated!")
            st.rerun()
        
        st.divider()
        
        # Reset game
        if st.button("🔄 Reset Game", type="secondary"):
            game.reset_game()
            st.success("Game reset!")
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("👥 Player Statistics")
        
        # Player stats table
        for name, p in game.players.items():
            net = p["energy"]
            win_rate = (p["wins"] / p["total_rounds"]) if p["total_rounds"] else 0.0
            
            with st.expander(f"**{name}** - Net: {net:+,.0f}", expanded=(name == user_choice)):
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric("Total Bets", f"{p['total_bets']:,.0f}")
                    st.metric("Win Rate", f"{win_rate:.1%}")
                
                with metrics_col2:
                    st.metric("Total Rewards", f"{p['total_rewards']:,.0f}")
                    st.metric("Biggest Win", f"{p['biggest_win']:,.0f}")
                
                with metrics_col3:
                    st.metric("Energy (R-B)", f"{p['energy']:+,.0f}")
                    st.metric("Biggest Loss", f"{p['biggest_loss']:,.0f}")
                
                # Progress bar for energy
                if p['total_bets'] > 0:
                    energy_ratio = p['energy'] / p['total_bets']
                    energy_progress = max(0.0, min(1.0, (energy_ratio + 1) / 2))  # Normalize to 0-1
                    st.progress(energy_progress, text=f"Energy Ratio: {energy_ratio:+.2%}")

    with col2:
        st.header("📈 System Health")
        
        # Control parameters
        st.subheader("Control Parameters")
        st.markdown(f"**P Scale:** {ctrl['p_scale']:.3f}")
        st.markdown(f"**Consolation Scale:** {ctrl['cons_scale']:.3f}")
        st.markdown(f"**Prize Skew:** {ctrl['skew']:.2f}")
        st.markdown(f"**RTP Delta:** {ctrl['delta']:+.3f}")
        
        # Current multipliers
        st.subheader("Active Multipliers")
        mult_cols = st.columns(3)
        for i, mult in enumerate(game.prize_multipliers):
            col_idx = i % 3
            mult_cols[col_idx].metric(f"M{i+1}", f"{mult:.1f}×")
        
        # Recent history
        if game.game_history:
            st.subheader("Recent Plays")
            recent_history = game.game_history[-10:]  # Last 10 plays
            
            for play in reversed(recent_history):
                outcome = "🎰 JP" if play.get("is_jackpot") else ("✅ Win" if play["won"] else "❌ Loss")
                st.text(f"{outcome} | {play['player']} | {play['bet']:,} → {play['player_received']:,.0f}")

    # Footer info
    st.divider()
    st.markdown(
        f"""
        **System Info:** Pool Limit: {POOL_MIN_LIMIT:,} | RTP Target: {RTP_TARGET:.1%} | 
        House Fee: {HOUSE_FEE:.0%} | Fee Threshold: {WIN_FEE_THRESHOLD_MULT:.1f}×
        """
    )


if __name__ == "__main__":
    main()
