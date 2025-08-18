import random
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Game economics (same base tables)
# ----------------------------
BASELINE_WIN_PROBABILITY = 0.85
BASELINE_PRIZE_MULTIPLIERS = [
    0.0,
    0.5, 0.6, 0.8, 0.85, 0.9,
    1.0, 1.05,
    1.2, 1.5, 2.0, 5.0, 8.0,
]
BASELINE_PRIZE_WEIGHTS = [
    1,
    8, 8, 8, 8, 8, 8, 8,
    6, 6,
    14, 11, 9,
]

# No fee
SYSTEM_FEE_RATE = 0.0

# We keep a micro prize in the distribution
MIN_MICRO_WIN = 0.05

PRIZE_MULTIPLIERS = [
    MIN_MICRO_WIN,
    0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9,
    1.0, 1.05,
    1.2, 1.5, 2.0,
]
PRIZE_WEIGHTS = [
    1,
    8, 8, 8, 8, 8, 8, 8,
    6, 6,
    14, 11, 9,
]

# ----------------------------
# RTP helper and scale (same logic)
# ----------------------------
def _weighted_avg(mults: List[float], weights: List[float]) -> float:
    tw = sum(weights)
    return sum(m * w for m, w in zip(mults, weights)) / tw if tw else 0.0

BASELINE_EXPECTED_PAYOUT_FACTOR = BASELINE_WIN_PROBABILITY * _weighted_avg(
    BASELINE_PRIZE_MULTIPLIERS, BASELINE_PRIZE_WEIGHTS
)
_active_avg = _weighted_avg(PRIZE_MULTIPLIERS, PRIZE_WEIGHTS)
PRIZE_SCALE = BASELINE_EXPECTED_PAYOUT_FACTOR / max(_active_avg, 1e-9)
PRIZE_SCALE = max(min(PRIZE_SCALE, 2.0), 0.2)  # clamp for stability


def _select_multiplier() -> float:
    if len(PRIZE_MULTIPLIERS) != len(PRIZE_WEIGHTS):
        return random.choice(PRIZE_MULTIPLIERS)
    return random.choices(PRIZE_MULTIPLIERS, weights=PRIZE_WEIGHTS)[0]


# ----------------------------
# API models (unchanged schema)
# ----------------------------
class PlayRequest(BaseModel):
    userId: str = Field(..., description="Caller user id")
    betAmount: float = Field(..., gt=0, description="User bet amount (> 0)")
    currentPool: float = Field(..., description="Current pool before this round")
    userEnergy: Optional[float] = Field(
        0.0, description="User total energy before this round (client-maintained)"
    )

    @field_validator("betAmount")
    @classmethod
    def _round_bet(cls, v: float) -> float:
        return round(float(v), 2)

    @field_validator("currentPool")
    @classmethod
    def _round_pool(cls, v: float) -> float:
        return round(float(v), 2)

    @field_validator("userEnergy")
    @classmethod
    def _round_energy(cls, v: Optional[float]) -> float:
        return round(float(v or 0.0), 2)


class PlayResponse(BaseModel):
    userId: str
    status: str               # "win" | "loss" (loss only if prizeAmount == 0)
    prizeAmount: float
    multiplier: float
    poolAfter: float
    cashWin: bool
    # compatibility fields (fee is 0; effectiveBet == bet)
    systemFee: float
    effectiveBet: float
    # energy matches pool delta (no fee):
    roundEnergy: float        # bet - prize
    totalEnergyAfter: float   # previous userEnergy + roundEnergy


# ----------------------------
# Core round resolution
#   - Pool never goes negative
#   - User loses ONLY if available funds < theoretical prize
# ----------------------------
def resolve_round(
    user_id: str,
    bet_amount: float,
    pool_before: float,
    user_energy_before: float,
) -> PlayResponse:
    system_fee = 0.0
    effective_bet = bet_amount

    # Safety: if client ever sends negative pool (legacy), clamp to 0
    pool_before_safe = max(0.0, round(pool_before, 2))

    # Available funds after adding the bet
    available = round(pool_before_safe + bet_amount, 2)

    # Always attempt to pay; only fail if insufficient funds
    base_mult = _select_multiplier()
    wanted_mult = round(base_mult * PRIZE_SCALE, 4)
    theoretical_prize = round(bet_amount * wanted_mult, 2)

    if theoretical_prize <= available:
        prize = theoretical_prize
        eff_mult = wanted_mult
    else:
        # Insufficient pool -> loss (prize 0)
        prize = 0.0
        eff_mult = 0.0

    pool_after = round(available - prize, 2)  # >= 0

    # Energy = bet - prize
    round_energy = round(bet_amount - prize, 2)
    total_energy_after = round(float(user_energy_before or 0.0) + round_energy, 2)

    status = "win" if prize > 0 else "loss"
    cash_win = (prize > 0)

    return PlayResponse(
        userId=user_id,
        status=status,
        prizeAmount=round(prize, 2),
        multiplier=eff_mult,
        poolAfter=pool_after,
        cashWin=cash_win,
        systemFee=system_fee,
        effectiveBet=effective_bet,
        roundEnergy=round_energy,
        totalEnergyAfter=total_energy_after,
    )


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="Pool Game API (Loss only if insufficient pool)",
    version="2.3.0",
    description=(
        "No-fee pool game. Pool never goes below 0. "
        "User only loses when the available funds are less than the theoretical prize. "
        "Energy: roundEnergy = bet - prize; totalEnergyAfter = userEnergy + roundEnergy."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["health"])
def health():
    return {
        "ok": True,
        "message": "Pool game API is running (loss only if insufficient pool).",
        # We no longer use a random win-chance gate, so leave this null for the UI.
        "cashWinChance": None,
        "prizeScale": PRIZE_SCALE,
        "winCondition": "Pays full theoretical prize if available >= theoretical; otherwise prize=0 (loss).",
        "energyDefinition": "roundEnergy = bet - prize; totalEnergy = sum(roundEnergy)",
        "feeRate": SYSTEM_FEE_RATE,
    }

@app.post("/play", response_model=PlayResponse, tags=["gameplay"])
def play_round(req: PlayRequest):
    """
    Resolve a single round (stateless):
    - available = max(0, currentPool) + bet
    - if theoretical_prize <= available: pay it; else prize = 0 (loss)
    - poolAfter = available - prize  (never negative)
    - Energy: roundEnergy = bet - prize. Client persists totalEnergyAfter per user.
    """
    return resolve_round(
        user_id=req.userId,
        bet_amount=req.betAmount,
        pool_before=req.currentPool,
        user_energy_before=req.userEnergy or 0.0,
    )
