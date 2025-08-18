# app.py
import random
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Fixed multipliers (no baseline / no scaling)
# ----------------------------
SYSTEM_FEE_RATE = 0.0          # no fee

# Minimum multiplier used as fallback when pool can't afford the drawn prize
MIN_MICRO_WIN: float = 0.9

PRIZE_MULTIPLIERS: List[float] = [
    MIN_MICRO_WIN,            
    1.2,1.3, 1.5, 2.0, 5.0
]
PRIZE_WEIGHTS: List[int] = [
    1,                    
    8, 8, 8, 8, 8, 8, 8,
    6, 6,
    14, 11, 9,
]

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
    # energy mirrors pool delta (no fee):
    roundEnergy: float        # bet - prize
    totalEnergyAfter: float   # previous userEnergy + roundEnergy


# ----------------------------
# Core round resolution
#   - Pool never goes negative
#   - Try drawn multiplier; if unaffordable, try MIN_MICRO_WIN; else prize=0 (loss)
#   - No fee
# ----------------------------
def resolve_round(
    user_id: str,
    bet_amount: float,
    pool_before: float,
    user_energy_before: float,
) -> PlayResponse:
    system_fee = 0.0
    effective_bet = bet_amount

    # Safety: if client sends negative pool (legacy), clamp to 0
    pool_before_safe = max(0.0, round(pool_before, 2))

    # Available funds after adding the bet
    available = round(pool_before_safe + bet_amount, 2)

    # Draw a multiplier from the fixed distribution
    drawn_mult = _select_multiplier()
    drawn_prize = round(bet_amount * drawn_mult, 2)

    prize = 0.0
    eff_mult = 0.0

    if drawn_prize <= available:
        # Pay the drawn prize
        prize = drawn_prize
        eff_mult = drawn_mult
    else:
        # Not enough for drawn prize — try minimum multiplier as fallback
        min_prize = round(bet_amount * MIN_MICRO_WIN, 2)
        if MIN_MICRO_WIN > 0 and min_prize <= available:
            prize = min_prize
            eff_mult = MIN_MICRO_WIN
        else:
            # Still not enough — loss (prize 0)
            prize = 0.0
            eff_mult = 0.0

    pool_after = round(available - prize, 2)  # >= 0

    # Energy = bet - prize (matches pool delta)
    round_energy = round(bet_amount - prize, 2)
    total_energy_after = round(float(user_energy_before or 0.0) + round_energy, 2)

    status = "win" if prize > 0 else "loss"
    cash_win = (prize > 0)

    return PlayResponse(
        userId=user_id,
        status=status,
        prizeAmount=prize,
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
    title="Pool Game API (Fixed Multipliers + Min-Multiplier Fallback)",
    version="3.1.0",
    description=(
        "No-fee pool game with fixed prize multipliers and weights. "
        "Pool never goes below 0. "
        "If the drawn prize exceeds available funds, the game tries the minimum multiplier; "
        "if that's still unaffordable, prize=0 (loss). "
        "Energy: roundEnergy = bet - prize; totalEnergyAfter = userEnergy + roundEnergy."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["health"])
def health():
    return {
        "ok": True,
        "message": "Pool game API is running (fixed multipliers, min-multiplier fallback).",
        "cashWinChance": None,      # outcomes depend on affordability, not fixed chance
        "prizeScale": 1.0,          # no scaling
        "winCondition": (
            "Pay drawn multiplier if affordable; otherwise try minimum multiplier; "
            "if still unaffordable, prize=0 (loss)."
        ),
        "minimumMultiplier": MIN_MICRO_WIN,
        "multipliers": PRIZE_MULTIPLIERS,
        "weights": PRIZE_WEIGHTS,
        "energyDefinition": "roundEnergy = bet - prize; totalEnergy = sum(roundEnergy)",
        "feeRate": SYSTEM_FEE_RATE,
    }

@app.post("/play", response_model=PlayResponse, tags=["gameplay"])
def play_round(req: PlayRequest):
    """
    Resolve a single round (stateless):
    - available = max(0, currentPool) + bet
    - drawn multiplier picked from fixed distribution
    - if drawn_prize <= available: pay it
      else if bet*MIN_MICRO_WIN <= available: pay minimum multiplier
      else: prize = 0 (loss)
    - poolAfter = available - prize  (never negative)
    - Energy: roundEnergy = bet - prize. Client persists totalEnergyAfter per user.
    """
    return resolve_round(
        user_id=req.userId,
        bet_amount=req.betAmount,
        pool_before=req.currentPool,
        user_energy_before=req.userEnergy or 0.0,
    )
