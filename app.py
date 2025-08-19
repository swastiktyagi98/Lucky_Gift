import random
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

SYSTEM_FEE_RATE = 0.0
MIN_MICRO_WIN = 0.0

PRIZE_MULTIPLIERS: List[float] = [MIN_MICRO_WIN, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]

ENERGY_THRESHOLDS = {
    "ultra_low": -20000,   # huge negative energy, big recent winner
    "very_low": -10000,    # heavy winner
    "low": -5000,          # moderate winner
    "medium": 0,           # breakeven
    "mid_high": 1000,      # slight loser
    "high": 5000,          # losing more
    "very_high": 10000,    # losing heavily
    "ultra_high": 20000    # VIP big loser
}


WIN_CHANCE_CONFIGS = {
    "ultra_low": {
        "weights": [70, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0],
        "description": "Super Protection - almost no wins"
    },
    "very_low": {
        "weights": [55, 8, 5, 3, 2, 1, 0, 0, 0, 0, 0],
        "description": "House Protection - very low odds"
    },
    "low": {
        "weights": [40, 10, 7, 5, 3, 2, 1, 0, 0, 0, 0],
        "description": "Defense Mode - low odds"
    },
    "medium": {
        "weights": [25, 12, 9, 7, 5, 4, 3, 2, 1, 1, 0],
        "description": "Balanced Mode - fair odds"
    },
    "mid_high": {
        "weights": [18, 14, 11, 9, 7, 6, 5, 4, 3, 2, 1],
        "description": "Slight Reward - above average odds"
    },
    "high": {
        "weights": [12, 15, 13, 11, 9, 7, 6, 5, 4, 3, 2],
        "description": "Reward Mode - high odds and bigger multipliers"
    },
    "very_high": {
        "weights": [8, 16, 14, 12, 10, 8, 6, 5, 4, 3, 2],
        "description": "VIP Mode - excellent odds and multipliers"
    },
    "ultra_high": {
        "weights": [5, 18, 16, 14, 12, 10, 8, 6, 5, 4, 3],
        "description": "Super VIP - best odds and maximum multipliers"
    }
}


def get_energy_tier(energy: float) -> str:
    if energy <= ENERGY_THRESHOLDS["very_low"]:
        return "very_low"
    elif energy <= ENERGY_THRESHOLDS["low"]:
        return "low"
    elif energy <= ENERGY_THRESHOLDS["medium"]:
        return "medium"
    elif energy <= ENERGY_THRESHOLDS["high"]:
        return "high"
    else:
        return "very_high"

def calculate_dynamic_weights(energy: float, bet_amount: float) -> List[int]:
    base_tier = get_energy_tier(energy)
    base = WIN_CHANCE_CONFIGS[base_tier]["weights"].copy()
    bet_bonus = min(0.10, max(0.0, bet_amount) / 10000.0)
    for i in range(1, len(base)):
        progressive = 1.0 + bet_bonus * (1.0 + i / (len(base) - 1))
        base[i] = max(0, int(round(base[i] * progressive)))
    if sum(base[1:]) == 0 and sum(WIN_CHANCE_CONFIGS[base_tier]["weights"][1:]) > 0:
        base[1] = 1
    return base

def select_multiplier_with_energy(energy: float, bet_amount: float) -> tuple[float, str, List[int]]:
    weights = calculate_dynamic_weights(energy, bet_amount)
    multiplier = random.choices(PRIZE_MULTIPLIERS, weights=weights, k=1)[0]
    tier = get_energy_tier(energy)
    return multiplier, WIN_CHANCE_CONFIGS[tier]["description"], weights

class PlayRequest(BaseModel):
    userId: str = Field(..., description="Caller user id")
    betAmount: float = Field(..., gt=0, description="User bet amount (> 0)")
    currentPool: float = Field(..., description="Current pool before this round")
    userEnergy: Optional[float] = Field(0.0, description="User total energy before this round")

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

def resolve_round(user_id: str, bet_amount: float, pool_before: float, user_energy_before: float) -> PlayResponse:
    available = max(0.0, round(pool_before + bet_amount, 2))
    drawn_mult, win_desc, _weights = select_multiplier_with_energy(user_energy_before, bet_amount)
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
        winChanceDescription=win_desc,
    )

app = FastAPI(
    title="Energy-Scaled Win & Multiplier Pool Game API",
    version="3.5.0",
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
            tier: {"weights": cfg["weights"], "description": cfg["description"]}
            for tier, cfg in WIN_CHANCE_CONFIGS.items()
        },
    }

@app.post("/play", response_model=PlayResponse)
def play_round(req: PlayRequest):
    return resolve_round(req.userId, req.betAmount, req.currentPool, req.userEnergy or 0.0)

def get_energy_recommendation(tier: str) -> str:
    return {
        "very_low": "Protection active — odds low and prizes small after big recent wins.",
        "low": "Defense mode — reduced odds; big multipliers rare.",
        "medium": "Balanced odds and prize sizes.",
        "high": "Reward mode — higher odds and larger multipliers.",
        "very_high": "VIP — best odds and biggest multipliers.",
    }.get(tier, "Unknown energy tier")

@app.get("/energy-analysis/{energy}")
def analyze_energy_tier(energy: float):
    tier = get_energy_tier(energy)
    cfg = WIN_CHANCE_CONFIGS[tier]
    weights = cfg["weights"]
    total = sum(weights)
    win_weights = sum(weights[1:])
    win_percentage = round((win_weights / total) * 100, 1) if total else 0.0
    expected_mult = (
        sum(PRIZE_MULTIPLIERS[i] * weights[i] for i in range(len(weights))) / total
        if total
        else 0.0
    )
    ev_per_dollar = round(expected_mult, 4)
    return {
        "energy": energy,
        "tier": tier,
        "description": cfg["description"],
        "weights": weights,
        "winPercentage": win_percentage,
        "expectedMultiplier": round(expected_mult, 3),
        "expectedValuePerDollar": ev_per_dollar,
        "recommendation": get_energy_recommendation(tier),
    }
