import random
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

# =========================
# Config
# =========================
SYSTEM_FEE_RATE = 0.0
MIN_MICRO_WIN = 0.0
PRIZE_MULTIPLIERS: List[float] = [MIN_MICRO_WIN, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Energy buckets (you can tweak numbers freely)
ENERGY_THRESHOLDS = {
    # Extremely losing player
    'very_low': -20000,
    # Losing player
    'low': -5000,
    # Around break-even / modest wins
    'medium': 10000,
    # Strong winner
    'high': 30000,
    # Big winner
    'very_high': 50000
}

# Win weights per energy tier (index 0 is "loss", others are multipliers in PRIZE_MULTIPLIERS order)
# --> Higher energy = higher overall win chance.
WIN_CHANCE_CONFIGS = {
    # ~30% win
    'very_low': {
        'weights': [20, 3, 2, 1, 1, 1, 1, 0],
        'description': 'House Protection Mode - Very Low Win Chance'
    },
    # ~48% win
    'low': {
        'weights': [16, 5, 3, 2, 2, 1, 1, 1],
        'description': 'House Defense Mode - Low Win Chance'
    },
    # ~68% win
    'medium': {
        'weights': [12, 8, 6, 4, 3, 2, 1, 1],
        'description': 'Balanced Mode - Good Win Chance'
    },
    # ~85% win
    'high': {
        'weights': [8, 12, 10, 8, 6, 4, 3, 2],
        'description': 'Reward Mode - Very High Win Chance'
    },
    # ~92% win
    'very_high': {
        'weights': [5, 15, 12, 10, 8, 6, 4, 3],
        'description': 'VIP Mode - Excellent Win Chance'
    }
}

def get_energy_tier(energy: float) -> str:
    """
    Map energy to a tier.
    Higher energy means the player has been losing more to the house (contributing more),
    so we reward them with higher win chances.
    """
    if energy <= ENERGY_THRESHOLDS['very_low']:
        return 'very_low'
    elif energy <= ENERGY_THRESHOLDS['low']:
        return 'low'
    elif energy <= ENERGY_THRESHOLDS['medium']:
        return 'medium'
    elif energy <= ENERGY_THRESHOLDS['high']:
        return 'high'
    else:
        return 'very_high'

def calculate_dynamic_weights(energy: float, bet_amount: float) -> List[int]:
    """
    Calculate weights from the energy tier and bet amount.
    Higher energy = better odds. Lower energy = worse odds.
    """
    base_tier = get_energy_tier(energy)
    base = WIN_CHANCE_CONFIGS[base_tier]['weights'].copy()

    # Slight bet-size bonus to winning outcomes (capped at +10%)
    bet_bonus = min(0.10, max(0.0, bet_amount) / 10000.0)
    for i in range(1, len(base)):  # skip index 0 (loss)
        base[i] = max(1, int(round(base[i] * (1 + bet_bonus))))
    return base

def select_multiplier_with_energy(energy: float, bet_amount: float) -> tuple[float, str]:
    weights = calculate_dynamic_weights(energy, bet_amount)
    multiplier = random.choices(PRIZE_MULTIPLIERS, weights=weights, k=1)[0]
    tier = get_energy_tier(energy)
    return multiplier, WIN_CHANCE_CONFIGS[tier]['description']

# =========================
# Models
# =========================
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
    energyTier: str
    winChanceDescription: str

# =========================
# Core logic
# =========================
def resolve_round(user_id: str, bet_amount: float, pool_before: float, user_energy_before: float) -> PlayResponse:
    available = max(0.0, round(pool_before + bet_amount, 2))

    drawn_mult, win_desc = select_multiplier_with_energy(user_energy_before, bet_amount)
    drawn_prize = round(bet_amount * drawn_mult, 2)

    if drawn_prize <= available:
        prize, eff_mult = drawn_prize, drawn_mult
    else:
        min_prize = round(bet_amount * MIN_MICRO_WIN, 2)
        prize, eff_mult = (min_prize, MIN_MICRO_WIN) if min_prize <= available else (0.0, 0.0)

    pool_after = round(available - prize, 2)
    round_energy = round(bet_amount - prize, 2)  # positive when player loses; negative when player wins big
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

# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="Energy-Based Pool Game API",
    version="3.3.0",
    description="Pool game with dynamic win chances based on user energy: higher energy ⇒ higher win chance."
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
                "weights": cfg["weights"],
                "description": cfg["description"]
            } for tier, cfg in WIN_CHANCE_CONFIGS.items()
        }
    }

@app.post("/play", response_model=PlayResponse)
def play_round(req: PlayRequest):
    return resolve_round(req.userId, req.betAmount, req.currentPool, req.userEnergy or 0.0)

def get_energy_recommendation(tier: str) -> str:
    recommendations = {
        'very_low': "Protection active — your odds are intentionally low after strong recent wins.",
        'low': "Defense mode — lower odds due to recent profitability.",
        'medium': "Balanced odds — neutral state.",
        'high': "Reward mode — strong odds as a valued contributor.",
        'very_high': "VIP treatment — excellent odds."
    }
    return reco
