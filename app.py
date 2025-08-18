import random
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Game economics (unchanged)
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

# 👉 No fee anymore
SYSTEM_FEE_RATE = 0.0

# Probability a round pays > 0
CASH_WIN_CHANCE = 0.97
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
# RTP helper
# ----------------------------
def _weighted_avg(mults: List[float], weights: List[float]) -> float:
    tw = sum(weights)
    return sum(m * w for m, w in zip(mults, weights)) / tw if tw else 0.0


BASELINE_EXPECTED_PAYOUT_FACTOR = BASELINE_WIN_PROBABILITY * _weighted_avg(
    BASELINE_PRIZE_MULTIPLIERS, BASELINE_PRIZE_WEIGHTS
)

_active_avg = _weighted_avg(PRIZE_MULTIPLIERS, PRIZE_WEIGHTS)
PRIZE_SCALE = BASELINE_EXPECTED_PAYOUT_FACTOR / max(CASH_WIN_CHANCE * _active_avg, 1e-9)
PRIZE_SCALE = max(min(PRIZE_SCALE, 2.0), 0.2)  # clamp for stability


def _select_multiplier() -> float:
    if len(PRIZE_MULTIPLIERS) != len(PRIZE_WEIGHTS):
        return random.choice(PRIZE_MULTIPLIERS)
    return random.choices(PRIZE_MULTIPLIERS, weights=PRIZE_WEIGHTS)[0]


# ----------------------------
# API models
# ----------------------------
class PlayRequest(BaseModel):
    userId: str = Field(..., description="Caller user id")
    betAmount: float = Field(..., gt=0, description="User bet amount (> 0)")
    currentPool: float = Field(..., description="Current pool before this round (can be negative)")
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
    # kept for compatibility (now: systemFee=0, effectiveBet=bet)
    systemFee: float
    effectiveBet: float
    # energy mirrors pool change exactly (no fee):
    roundEnergy: float        # bet - prize
    totalEnergyAfter: float   # previous userEnergy + roundEnergy


# ----------------------------
# Core round resolution
# ----------------------------
def resolve_round(
    user_id: str,
    bet_amount: float,
    pool_before: float,
    user_energy_before: float,
) -> PlayResponse:
    # No fee
    system_fee = 0.0
    effective_bet = bet_amount

    # Pool receives full bet (can go negative after payout)
    pool = round(pool_before + bet_amount, 2)

    # Determine if this round pays (cash win)
    pays = (random.random() < CASH_WIN_CHANCE)

    prize = 0.0
    eff_mult = 0.0

    if pays:
        base_mult = _select_multiplier()
        eff_mult = round(base_mult * PRIZE_SCALE, 4)
        prize = round(bet_amount * eff_mult, 2)
        pool = round(pool - prize, 2)  # allow negative pool

    status = "win" if prize > 0 else "loss"

    # Energy = bet - prize  (pool change this round)
    round_energy = round(bet_amount - prize, 2)
    total_energy_after = round(float(user_energy_before or 0.0) + round_energy, 2)

    return PlayResponse(
        userId=user_id,
        status=status,
        prizeAmount=prize,
        multiplier=eff_mult if prize > 0 else 0.0,
        poolAfter=pool,
        cashWin=pays,
        systemFee=system_fee,
        effectiveBet=effective_bet,
        roundEnergy=round_energy,
        totalEnergyAfter=total_energy_after,
    )


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(
    title="Player-Friendly Pool Game API (No-Fee)",
    version="2.0.0",
    description=(
        "No-fee pool game. Pool can go negative. "
        "Energy definition: roundEnergy = bet - prize; totalEnergyAfter = userEnergy + roundEnergy. "
        "With startPool = 0, pool == sum(all users' energy)."
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
        "message": "Pool game API is running (no fee, pool can go negative).",
        "cashWinChance": CASH_WIN_CHANCE,
        "prizeScale": PRIZE_SCALE,
        "energyDefinition": "roundEnergy = bet - prize; totalEnergy = sum(roundEnergy)",
        "feeRate": SYSTEM_FEE_RATE,
    }

@app.post("/play", response_model=PlayResponse, tags=["gameplay"])
def play_round(req: PlayRequest):
    """
    Resolve a single round (stateless):
    - Pool += bet, then pool -= prize (if any). Pool may be negative.
    - Loss only when prizeAmount == 0.
    - Energy: roundEnergy = bet - prize. Client persists totalEnergyAfter per user.
    """
    return resolve_round(
        user_id=req.userId,
        bet_amount=req.betAmount,
        pool_before=req.currentPool,
        user_energy_before=req.userEnergy or 0.0,
    )
