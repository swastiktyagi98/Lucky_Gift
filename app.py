import random
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

# Constants
SYSTEM_FEE_RATE = 0.0
MIN_MICRO_WIN = 0.9

PRIZE_MULTIPLIERS: List[float] = [MIN_MICRO_WIN, 1.2, 1.3, 1.5, 2.0, 5.0]
PRIZE_WEIGHTS: List[int] = [1, 8, 8, 8, 8, 8, 8, 8, 6, 6, 14, 11, 9]

def _select_multiplier() -> float:
    return random.choices(PRIZE_MULTIPLIERS, weights=PRIZE_WEIGHTS)[0] if len(PRIZE_MULTIPLIERS) == len(PRIZE_WEIGHTS) else random.choice(PRIZE_MULTIPLIERS)

# Models
class PlayRequest(BaseModel):
    userId: str = Field(..., description="Caller user id")
    betAmount: float = Field(..., gt=0, description="User bet amount (> 0)")
    currentPool: float = Field(..., description="Current pool before this round")
    userEnergy: Optional[float] = Field(0.0, description="User total energy before this round (client-maintained)")

    @field_validator("betAmount", "currentPool")
    @classmethod
    def _round_values(cls, v: float) -> float:
        return round(float(v), 2)

    @field_validator("userEnergy")
    @classmethod
    def _round_energy(cls, v: Optional[float]) -> float:
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

# Core logic
def resolve_round(user_id: str, bet_amount: float, pool_before: float, user_energy_before: float) -> PlayResponse:
    available = max(0.0, round(pool_before + bet_amount, 2))
    drawn_mult = _select_multiplier()
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
    )

# FastAPI app
app = FastAPI(
    title="Pool Game API",
    version="3.1.0",
    description="No-fee pool game with fixed prize multipliers. Pool never goes below 0."
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
        "message": "Pool game API is running",
        "multipliers": PRIZE_MULTIPLIERS,
        "weights": PRIZE_WEIGHTS,
        "feeRate": SYSTEM_FEE_RATE,
    }

@app.post("/play", response_model=PlayResponse)
def play_round(req: PlayRequest):
    return resolve_round(req.userId, req.betAmount, req.currentPool, req.userEnergy or 0.0)
