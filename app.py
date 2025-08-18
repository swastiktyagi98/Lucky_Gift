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

# Index 0 = loss (0.0), the rest are winning multipliers (keep order in sync with weights)
PRIZE_MULTIPLIERS: List[float] = [0.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Energy buckets
# Interpretation: higher (more positive) energy means the user has lost more overall (contributed to the house)
# and should be rewarded with higher win odds and bigger multipliers.
ENERGY_THRESHOLDS = {
    'very_low': -20000,   # heavy winner (house is down)
    'low': -5000,         # recent winner
    'medium': 10000,      # around breakeven
    'high': 30000,        # losing to house
    'very_high': 50000    # losing heavily to house (VIP)
}

# Weights per tier (length must equal len(PRIZE_MULTIPLIERS)).
# Index meaning: [loss, x2, x5, x6, x7, x8, x9, x10]
# These weights jointly control:
#   1) overall win probability (loss bucket vs others)
#   2) distribution across prize multipliers when the user *does* win.
WIN_CHANCE_CONFIGS = {
    # Very-low energy (user has been winning a lot) -> few wins, mostly tiny payouts when they do win
    'very_low': {
        'weights': [40, 6, 3, 1, 0, 0, 0, 0],   # ~19% win, mostly x2, x5 rare, no jackpots
        'description': 'House Protection - Tiny wins only'
    },
    # Low energy (still profitable) -> low win odds, skewed to small multipliers
    'low': {
        'weights': [25, 8, 5, 3, 2, 1, 0, 0],   # ~40% win; x2 common; x5/x6 sometimes; jackpots rare/none
        'description': 'House Defense - Mostly small wins'
    },
    # Medium energy (neutral) -> balanced odds and a full spread of multipliers
    'medium': {
        'weights': [15, 10, 8, 6, 4, 3, 2, 1],  # ~68% win; distribution favors x2–x6 with rare jackpots
        'description': 'Balanced Mode - Fair odds'
    },
    # High energy (losing) -> strong win odds and better access to higher multipliers
    'high': {
        'weights': [10, 12, 10, 8, 6, 5, 3, 2], # ~80% win; x7–x10 show up
        'description': 'Reward Mode - Higher multipliers appear more often'
    },
    # Very-high energy (losing heavily) -> best odds and biggest multipliers most accessible
    'very_high': {
        'weights': [5, 14, 12, 10, 8, 7, 5, 4], # ~91% win; strong tail on x7–x10
        'description': 'VIP Mode - Best chance and bigger multipliers'
    }
}

# -------------------------
# Helpers
# -------------------------
def get_energy_tier(energy: float) -> str:
    """
    Map energy to a tier.
    Higher energy (more positive) ⇒ user has been losing to the house ⇒ reward with higher odds/bigger multipliers.
    Lower/negative energy ⇒ user has been winning ⇒ protect the house.
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
    Start from the tier weights, then add a small (capped) bet-size bonus
    to *winning* buckets (keeps loss bucket unchanged).
    """
    base_tier = get_energy_tier(energy)
    base = WIN_CHANCE_CONFIGS[base_tier]['weights'].copy()

    # Gentle bet-size bonus up to +10% on winning weights only
    bet_bonus = min(0.10, max(0.0, bet_amount) / 10000.0)
    for i in range(1, len(base)):  # skip loss index 0
        # Slight progressive boost for higher multipliers (encourage bigger prizes on bigger bets)
        progressive = 1.0 + bet_bonus * (1.0 + i / (len(base) - 1))
        base[i] = max(0, int(round(base[i] * progressive)))

    # Ensure at least one winning bucket stays > 0 when original was > 0
    if sum(base[1:]) == 0 and sum(WIN_CHANCE_CONFIGS[base_tier]['weights'][1:]) > 0:
        base[1] = 1

    return base

def select_multiplier_with_energy(energy: float, bet_amount: float) -> tuple[float, str, List[int]]:
    weights = calculate_dynamic_weights(energy, bet_amount)
    multiplier = random.choices(PRIZE_MULTIPLIERS, weights=weights, k=1)[0]
    tier = get_energy_tier(energy)
    return multiplier, WIN_CHANCE_CONFIGS[tier]['description'], weights

# -------------------------
# Models
# -------------------------
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

# -------------------------
# Core round resolution
# -------------------------
def resolve_round(user_id: str, bet_amount: float, pool_before: float, user_energy_before: float) -> PlayResponse:
    available = max(0.0, round(pool_before + bet_amount, 2))

    drawn_mult, win_desc, _weights = select_multiplier_with_energy(user_energy_before, bet_amount)
    drawn_prize = round(bet_amount * drawn_mult, 2)

    if drawn_prize <= available:
        prize, eff_mult = drawn_prize, drawn_mult
    else:
        # Respect pool constraint; fall back to micro/zero prize if not enough liquidity
        min_prize = round(bet_amount * MIN_MICRO_WIN, 2)
        prize, eff_mult = (min_prize, MIN_MICRO_WIN) if min_prize <= available else (0.0, 0.0)

    pool_after = round(available - prize, 2)
    # Positive roundEnergy means user lost that amount to the house; negative means user won more than bet
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

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="Energy-Scaled Win & Multiplier Pool Game API",
    version="3.4.0",
    description=(
        "Win chance and prize multipliers both scale with user energy. "
        "Higher energy ⇒ higher win chance and larger multipliers. "
        "Lower/negative energy ⇒ lower win chance and smaller multipliers."
    ),
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
    return {
        'very_low': "Protection active — odds low and prizes small after big recent wins.",
        'low': "Defense mode — reduced odds; big multipliers rare.",
        'medium': "Balanced odds and prize sizes.",
        'high': "Reward mode — higher odds and larger multipliers.",
        'very_high': "VIP — best odds and biggest multipliers."
    }.get(tier, "Unknown energy tier")

@app.get("/energy-analysis/{energy}")
def analyze_energy_tier(energy: float):
    """
    Inspect tier mapping, implied win rate, expected multiplier, and EV per $1 bet (ignoring pool cap).
    """
    tier = get_energy_tier(energy)
    cfg = WIN_CHANCE_CONFIGS[tier]
    weights = cfg['weights']

    total = sum(weights)
    win_weights = sum(weights[1:])  # exclude loss bucket
    win_percentage = round((win_weights / total) * 100, 1) if total else 0.0

    expected_mult = (
        sum(PRIZE_MULTIPLIERS[i] * weights[i] for i in range(len(weights))) / total
        if total else 0.0
    )
    # EV per $1 bet (before pool/liquidity constraints and fees)
    ev_per_dollar = round(expected_mult, 4)

    return {
        "energy": energy,
        "tier": tier,
        "description": cfg['description'],
        "weights": weights,
        "winPercentage": win_percentage,
        "expectedMultiplier": round(expected_mult, 3),
        "expectedValuePerDollar": ev_per_dollar,
        "recommendation": get_energy_recommendation(tier)
    }
