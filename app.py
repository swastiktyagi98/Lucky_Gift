import random
import math
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

# Constants
SYSTEM_FEE_RATE = 0.0
MIN_MICRO_WIN = 0.0
PRIZE_MULTIPLIERS: List[float] = [MIN_MICRO_WIN, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Energy-based win chance configuration
ENERGY_THRESHOLDS = {
    'very_low': -10000,   # Energy <= -10000 (heavy losses)
    'low': 0,             # Energy <= 0 (some losses)
    'medium': 5000,       # Energy <= 5000 (break even to small wins)
    'high': 15000,        # Energy <= 15000 (good wins)
    'very_high': 25000    # Energy > 25000 (big winners)
}

# Win chance configurations for different energy levels
# Higher energy = lower win chance, lower energy = higher win chance
WIN_CHANCE_CONFIGS = {
    'very_low': {
        'weights': [1, 15, 12, 10, 8, 6, 4, 3],   # ~87% win chance - help recover losses
        'description': 'Recovery Mode - Very High Win Chance'
    },
    'low': {
        'weights': [2, 12, 10, 8, 6, 4, 3, 2],    # ~81% win chance - good recovery
        'description': 'Favorable Odds - High Win Chance'
    },
    'medium': {
        'weights': [4, 8, 7, 6, 5, 4, 3, 2],      # ~69% win chance - balanced
        'description': 'Balanced Odds - Medium Win Chance'
    },
    'high': {
        'weights': [6, 6, 5, 4, 3, 2, 2, 1],      # ~52% win chance - slightly unfavorable
        'description': 'Cooling Down - Low Win Chance'
    },
    'very_high': {
        'weights': [8, 4, 3, 2, 2, 1, 1, 1],      # ~36% win chance - prevent big streaks
        'description': 'Hot Streak Protection - Very Low Win Chance'
    }
}

def get_energy_tier(energy: float) -> str:
    """Determine energy tier based on current energy level"""
    if energy <= ENERGY_THRESHOLDS['very_low']:  # <= -10000
        return 'very_low'
    elif energy <= ENERGY_THRESHOLDS['low']:     # <= 0
        return 'low'
    elif energy <= ENERGY_THRESHOLDS['medium']:  # <= 5000
        return 'medium'
    elif energy <= ENERGY_THRESHOLDS['high']:    # <= 15000
        return 'high'
    else:                                        # > 15000
        return 'very_high'

def calculate_dynamic_weights(energy: float, bet_amount: float) -> List[int]:
    """
    Calculate dynamic weights based on energy level and bet size
    Lower energy = better odds, higher energy = worse odds
    """
    base_tier = get_energy_tier(energy)
    base_weights = WIN_CHANCE_CONFIGS[base_tier]['weights'].copy()
    
    # Optional: Add bet size influence (larger bets get slightly better odds)
    bet_bonus = min(0.1, bet_amount / 10000)  # Max 10% bonus for large bets
    
    # Apply bet bonus to winning multipliers (skip index 0 which is loss)
    for i in range(1, len(base_weights)):
        base_weights[i] = int(base_weights[i] * (1 + bet_bonus))
    
    return base_weights

def select_multiplier_with_energy(energy: float, bet_amount: float) -> tuple[float, str]:
    """Select multiplier based on energy level"""
    weights = calculate_dynamic_weights(energy, bet_amount)
    
    if len(PRIZE_MULTIPLIERS) == len(weights):
        multiplier = random.choices(PRIZE_MULTIPLIERS, weights=weights)[0]
    else:
        multiplier = random.choice(PRIZE_MULTIPLIERS)
    
    tier = get_energy_tier(energy)
    return multiplier, WIN_CHANCE_CONFIGS[tier]['description']

# Models
class PlayRequest(BaseModel):
    userId: str = Field(..., description="Caller user id")
    betAmount: float = Field(..., gt=0, description="User bet amount (> 0)")
    currentPool: float = Field(..., description="Current pool before this round")
    userEnergy: Optional[float] = Field(0.0, description="User total energy before this round (client-maintained)")
    
    @field_validator("betAmount", "currentPool")
    @classmethod
    def round_values(cls, v: float) -> float:
        return round(float(v), 2)
    
    @field_validator("userEnergy")
    @classmethod
    def round_energy(cls, v: Optional[float]) -> float:
        return round(float(v or 0.0), 2)

class PlayResponse(BaseModel):
    userId: str
    status: str
    prizeAmount: float
    multiplier: float
    poolAfter: float
    cashWin: bool
    systemFee: float
    effectiveBet: float
    roundEnergy: float
    totalEnergyAfter: float
    energyTier: str  # New field to show energy tier
    winChanceDescription: str  # New field to describe win chance

# Core logic
def resolve_round(user_id: str, bet_amount: float, pool_before: float, user_energy_before: float) -> PlayResponse:
    available = max(0.0, round(pool_before + bet_amount, 2))
    
    # Use energy-based multiplier selection
    drawn_mult, win_desc = select_multiplier_with_energy(user_energy_before, bet_amount)
    drawn_prize = round(bet_amount * drawn_mult, 2)
    
    if drawn_prize <= available:
        prize, eff_mult = drawn_prize, drawn_mult
    else:
        min_prize = round(bet_amount * MIN_MICRO_WIN, 2)
        prize, eff_mult = (min_prize, MIN_MICRO_WIN) if min_prize <= available else (0.0, 0.0)
    
    pool_after = round(available - prize, 2)
    round_energy = round(bet_amount - prize, 2)
    total_energy_after = round(float(user_energy_before or 0.0) + round_energy, 2)
    
    return PlayResponse(
        userId=user_id,
        status="win" if prize > 0 else "loss",
        prizeAmount=prize,
        multiplier=eff_mult,
        poolAfter=pool_after,
        cashWin=prize > 0,
        systemFee=0.0,
        effectiveBet=bet_amount,
        roundEnergy=round_energy,
        totalEnergyAfter=total_energy_after,
        energyTier=get_energy_tier(user_energy_before),
        winChanceDescription=win_desc
    )

# FastAPI app
app = FastAPI(
    title="Energy-Based Pool Game API",
    version="3.2.0",
    description="Pool game with dynamic win chances based on user energy levels. Lower energy = higher win chance."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {
        "ok": True,
        "message": "Energy-based pool game API is running",
        "multipliers": PRIZE_MULTIPLIERS,
        "feeRate": SYSTEM_FEE_RATE,
        "energyTiers": ENERGY_THRESHOLDS,
        "winChanceConfigs": {
            tier: {
                "weights": config["weights"],
                "description": config["description"]
            }
            for tier, config in WIN_CHANCE_CONFIGS.items()
        }
    }

@app.post("/play", response_model=PlayResponse)
def play_round(req: PlayRequest):
    return resolve_round(req.userId, req.betAmount, req.currentPool, req.userEnergy or 0.0)

@app.get("/energy-analysis/{energy}")
def analyze_energy_tier(energy: float):
    """Analyze what tier an energy level falls into and expected win chances"""
    tier = get_energy_tier(energy)
    config = WIN_CHANCE_CONFIGS[tier]
    weights = config['weights']
    
    # Calculate actual win percentages
    total_weight = sum(weights)
    win_weights = sum(weights[1:])  # Exclude loss weight (index 0)
    win_percentage = (win_weights / total_weight) * 100
    
    # Calculate expected multiplier
    expected_mult = sum(PRIZE_MULTIPLIERS[i] * weights[i] for i in range(len(weights))) / total_weight
    
    return {
        "energy": energy,
        "tier": tier,
        "description": config['description'],
        "winPercentage": round(win_percentage, 1),
        "expectedMultiplier": round(expected_mult, 2),
        "weights": weights,
        "recommendation": get_energy_recommendation(tier)
    }

def get_energy_recommendation(tier: str) -> str:
    """Provide recommendations based on energy tier"""
    recommendations = {
        'very_low': "Great time to play! You have the highest win chances.",
        'low': "Good time to play with favorable odds.",
        'medium': "Moderate win chances - play with caution.",
        'high': "Consider taking a break or playing smaller bets.",
        'very_high': "Very unfavorable odds - recommended to pause playing."
    }
    return recommendations.get(tier, "Unknown energy tier")
