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

def _weighted_avg(mults: List[float], weights: List[int]) -> float:
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
    userEnergy: Optional[int] = Field(
        None, ge=0, le=100,
        description="0..100. Currently informational; does not alter odds."
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


class PlayResponse(BaseModel):
    userId: str
    status: str  # "win" | "loss"
    prizeAmount: float
    multiplier: float
    poolAfter: float
    # extra transparency (optional for clients to use)
    systemFee: float
    effectiveBet: float
    cashWin: bool


# ----------------------------
# Core round resolution
# ----------------------------

def resolve_round(user_id: str, bet_amount: float, pool_before: float) -> PlayResponse:
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

    return PlayResponse(
        userId=user_id,
        status=status,
        prizeAmount=prize,
        multiplier=eff_mult if prize > 0 else 0.0,
        poolAfter=pool,
        systemFee=system_fee,
        effectiveBet=effective_bet,
        cashWin=pays
    )


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(
    title="Player-Friendly Pool Game API",
    version="1.0.0",
    description=(
        "API to resolve a pool-based round. "
        "Send userId, betAmount, currentPool, userEnergy. "
        "Returns status ('win'|'loss') and prizeAmount (0 means loss), "
        "plus poolAfter and transparency fields."
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
        "prizeScale": PRIZE_SCALE
    }


@app.post("/play", response_model=PlayResponse, tags=["gameplay"])
def play_round(req: PlayRequest):
    """
    Resolve a single round using the current pool.
    - Loss is only when prizeAmount == 0.
    - The pool is updated by taking in (bet - systemFee) and paying out the prize.
    - The endpoint is **stateless**: you must persist `poolAfter` on your side.
    """
    # NOTE: userEnergy is accepted for future tuning. Current odds honor the original design.
    return resolve_round(
        user_id=req.userId,
        bet_amount=req.betAmount,
        pool_before=req.currentPool
    )
