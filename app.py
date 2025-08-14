import random
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware


# ----------------------------
# Original game economics
# ----------------------------

BASELINE_WIN_PROBABILITY = 0.85
BASELINE_PRIZE_MULTIPLIERS = [
    0.0,
    0.5, 0.6, 0.8, 0.85, 0.9,
    1.0, 1.05,
    1.2, 1.5, 2.0, 5.0, 8.0
]
BASELINE_PRIZE_WEIGHTS = [
    1,
    8, 8, 8, 8, 8, 8, 8,
    6, 6,
    14, 11, 9
]

SYSTEM_FEE_RATE = 0.10
CASH_WIN_CHANCE = 0.97
MIN_MICRO_WIN = 0.05

PRIZE_MULTIPLIERS = [
    MIN_MICRO_WIN,
    0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9,
    1.0, 1.05,
    1.2, 1.5, 2.0
]

PRIZE_WEIGHTS = [
    1,
    8, 8, 8, 8, 8, 8, 8,
    6, 6,
    14, 11, 9
]


# ----------------------------
# Helpers to preserve baseline RTP
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
    # Defensive: if configs drift
    if len(PRIZE_MULTIPLIERS) != len(PRIZE_WEIGHTS):
        return random.choice(PRIZE_MULTIPLIERS)
    return random.choices(PRIZE_MULTIPLIERS, weights=PRIZE_WEIGHTS)[0]


# ----------------------------
# API models
# ----------------------------

class PlayRequest(BaseModel):
    userId: str = Field(..., description="Caller user id")
    betAmount: float = Field(..., gt=0, description="User bet amount (must be > 0)")
    currentPool: float = Field(..., ge=0, description="Current pool balance before this round")
    # User total energy BEFORE this round (start at 0.0 for new users)
    userEnergy: Optional[float] = Field(
        0.0,
        description="User total energy before this round. Start at 0.0 and pass back each time."
    )

    @field_validator("betAmount")
    @classmethod
    def _round_bet(cls, v: float) -> float:
        # Keep cents precision stable
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
    status: str  # "win" | "loss" (loss only if prizeAmount == 0)
    prizeAmount: float
    multiplier: float
    poolAfter: float
    # transparency fields
    systemFee: float
    effectiveBet: float
    cashWin: bool
    # energy accounting
    roundEnergy: float           # betAmount - prizeAmount
    totalEnergyAfter: float      # previous userEnergy + roundEnergy


# ----------------------------
# Core round resolution
# ----------------------------

def resolve_round(user_id: str, bet_amount: float, pool_before: float, user_energy_before: float) -> PlayResponse:
    # Fees & pool inflow
    system_fee = round(bet_amount * SYSTEM_FEE_RATE, 2)
    effective_bet = round(bet_amount - system_fee, 2)
    pool = round(pool_before + effective_bet, 2)

    # Determine if this round pays (cash win)
    pays = (random.random() < CASH_WIN_CHANCE)

    prize = 0.0
    eff_mult = 0.0

    if pays:
        base_mult = _select_multiplier()
        eff_mult = round(base_mult * PRIZE_SCALE, 4)
        theoretical_prize = bet_amount * eff_mult

        if pool >= theoretical_prize:
            prize = theoretical_prize
            pool -= prize
        else:
            # Not enough in the pool; degrade prize to keep pool non-negative
            prize = max(0.0, pool * 0.5)
            eff_mult = round((prize / bet_amount) if bet_amount > 0 else 0.0, 4)
            pool -= prize

        pool = round(pool, 2)
        prize = round(prize, 2)

    status = "win" if prize > 0 else "loss"

    # ----------------------------
    # ENERGY ACCOUNTING (as requested)
    # current round energy = bet - prize
    # new total energy = previous total energy + current round energy
    # ----------------------------
    round_energy = round(bet_amount - prize, 2)
    total_energy_after = round(float(user_energy_before or 0.0) + round_energy, 2)

    return PlayResponse(
        userId=user_id,
        status=status,
        prizeAmount=prize,
        multiplier=eff_mult if prize > 0 else 0.0,
        poolAfter=pool,
        systemFee=system_fee,
        effectiveBet=effective_bet,
        cashWin=pays,
        roundEnergy=round_energy,
        totalEnergyAfter=total_energy_after
    )


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(
    title="Player-Friendly Pool Game API",
    version="1.2.0",
    description=(
        "Pool-based round resolver. Send userId, betAmount, currentPool, userEnergy (user's total energy before this round). "
        "Energy is accounted as: roundEnergy = bet - prize; totalEnergyAfter = userEnergy + roundEnergy. "
        "Loss occurs only when prizeAmount == 0."
    ),
)

# CORS (adjust origins as needed)
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
        "message": "Player-Friendly Pool Game API is running.",
        "cashWinChance": CASH_WIN_CHANCE,
        "prizeScale": PRIZE_SCALE,
        "energyDefinition": "roundEnergy = bet - prize; totalEnergyAfter = prevUserEnergy + roundEnergy"
    }


@app.post("/play", response_model=PlayResponse, tags=["gameplay"])
def play_round(req: PlayRequest):
    """
    Resolve a single round using the current pool.

    - Loss is only when prizeAmount == 0.
    - Pool is updated by taking in (bet - systemFee) and paying out the prize.
    - ENERGY:
        * roundEnergy = betAmount - prizeAmount
        * totalEnergyAfter = req.userEnergy + roundEnergy
      (Client should persist totalEnergyAfter and pass it as userEnergy on the next call.)
    - Endpoint is stateless aside from values you pass in.
    """
    return resolve_round(
        user_id=req.userId,
        bet_amount=req.betAmount,
        pool_before=req.currentPool,
        user_energy_before=req.userEnergy or 0.0
    )
