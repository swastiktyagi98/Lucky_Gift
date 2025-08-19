import random
from collections import deque
from typing import List, Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware

# =========================
# House/Engagement Controls
# =========================

SYSTEM_FEE_RATE = 0.00      # 0–0.05 typical if you want a fee; kept 0 to match your "no-fee" spec
MIN_MICRO_WIN = 0.0         # if >0, fallback micro prize when pool-limited

# RTP governor (keeps the house from losing long term)
TARGET_RTP = 0.92           # target return-to-player across a rolling window
RTP_WINDOW = 600            # how many recent rounds to consider
RTP_ADJ_GAIN = 0.08         # how aggressively to nudge weights (0.03–0.12 reasonable)
MAX_HOUSE_BIAS = 0.35       # caps how far we bias weights to protect RTP (safety)

# Engagement knobs
ENABLE_VOLATILITY_SPIKES = True  # rare “spiky” rounds (more high multipliers) when safe
ENABLE_LDW = True                # Losses-Disguised-as-Wins via <1.0 multipliers (responsible toggle)

# =========================
# Prize Space
# =========================
# Index 0 = loss (0.0). Include sub-1.0 multipliers only if ENABLE_LDW is True.
BASE_MULTS: List[float] = [0.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]
LDW_MULTS:  List[float] = [0.0, 0.5, 0.8, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0]

PRIZE_MULTIPLIERS: List[float] = LDW_MULTS if ENABLE_LDW else BASE_MULTS

# =========================
# Energy tiers (8 stages)
# =========================
# Your earlier 5-tier config extended to 8 for finer control.
ENERGY_THRESHOLDS = {
    "ultra_low":  -20000,
    "very_low":   -10000,
    "low":        -5000,
    "medium":          0,
    "mid_high":     1000,
    "high":         5000,
    "very_high":   10000,
    "ultra_high":  20000
}

# Helper: length alignment
W = len(PRIZE_MULTIPLIERS)

def pad_or_trim(weights: List[int]) -> List[int]:
    if len(weights) == W:
        return weights
    # expand with trailing zeros or trim
    if len(weights) < W:
        return weights + [0] * (W - len(weights))
    return weights[:W]

# Baseline weights per tier (index 0 is loss bucket).
# These control BOTH win chance and which multiplier appears.
if ENABLE_LDW:
    # With small (<1.0) multipliers included: allow frequent “wins” that are still net-negative.
    _cfg = {
        "ultra_low":  [70,  8, 6,  5, 3, 2, 1, 1, 0, 0, 0, 0, 0],  # heavy winner -> strong protection
        "very_low":   [55, 10, 8,  6, 5, 3, 2, 1, 1, 1, 0, 0, 0],
        "low":        [42, 12, 9,  7, 6, 5, 4, 3, 2, 1, 1, 0, 0],
        "medium":     [28, 10, 8,  6, 6, 5, 4, 3, 2, 2, 1, 1, 1],
        "mid_high":   [22,  8, 6,  6, 6, 6, 5, 4, 4, 3, 2, 2, 1],
        "high":       [16,  6, 5,  5, 6, 7, 7, 6, 6, 5, 4, 3, 2],
        "very_high":  [12,  5, 4,  4, 5, 6, 7, 7, 7, 6, 5, 4, 3],
        "ultra_high": [ 8,  4, 3,  3, 4, 5, 6, 7, 8, 7, 6, 5, 4]
    }
else:
    # No LDW: low tiers still lose often; wins tend to be small, big tails appear at high tiers.
    _cfg = {
        "ultra_low":  [70, 5,  2, 1, 0, 0, 0, 0, 0, 0, 0],
        "very_low":   [55, 8,  5, 3, 2, 1, 0, 0, 0, 0, 0],
        "low":        [40,10,  7, 5, 3, 2, 1, 0, 0, 0, 0],
        "medium":     [25,12,  9, 7, 5, 4, 3, 2, 1, 1, 0],
        "mid_high":   [18,14, 11, 9, 7, 6, 5, 4, 3, 2, 1],
        "high":       [12,15, 13,11, 9, 7, 6, 5, 4, 3, 2],
        "very_high":  [ 8,16, 14,12,10, 8, 6, 5, 4, 3, 2],
        "ultra_high": [ 5,18, 16,14,12,10, 8, 6, 5, 4, 3],
    }

WIN_CHANCE_CONFIGS = {
    tier: {
        "weights": pad_or_trim(weights),
        "description": desc
    } for tier, weights, desc in [
        ("ultra_low",  _cfg["ultra_low"],  "Super Protection — almost no big wins"),
        ("very_low",   _cfg["very_low"],   "House Protection — very low odds"),
        ("low",        _cfg["low"],        "Defense — low odds"),
        ("medium",     _cfg["medium"],     "Balanced — fair odds"),
        ("mid_high",   _cfg["mid_high"],   "Slight Reward — above average odds"),
        ("high",       _cfg["high"],       "Reward — high odds & bigger multipliers"),
        ("very_high",  _cfg["very_high"],  "VIP — excellent odds"),
        ("ultra_high", _cfg["ultra_high"], "Super VIP — best odds & biggest multipliers"),
    ]
}

# =========================
# RTP Governor State
# =========================
recent_rounds = deque(maxlen=RTP_WINDOW)   # store tuples: (bet, prize)

def observed_rtp() -> float:
    if not recent_rounds:
        return TARGET_RTP
    bet_sum = sum(b for b, _ in recent_rounds)
    if bet_sum <= 0:
        return TARGET_RTP
    prize_sum = sum(p for _, p in recent_rounds)
    return prize_sum / bet_sum

def rtp_bias_factor() -> float:
    """
    Returns a bias in [-MAX_HOUSE_BIAS, +MAX_HOUSE_BIAS].
    Positive = make outcomes richer (more wins), Negative = tighten (more losses).
    """
    rtp = observed_rtp()
    error = TARGET_RTP - rtp
    bias = max(-MAX_HOUSE_BIAS, min(MAX_HOUSE_BIAS, RTP_ADJ_GAIN * error))
    return bias

def apply_house_bias(weights: List[int]) -> List[int]:
    """
    Nudge the distribution based on RTP error.
    - If RTP > target (house losing), increase loss/low buckets and shave high multipliers.
    - If RTP < target, relax.
    """
    bias = rtp_bias_factor()
    if abs(bias) < 1e-6:
        return weights

    w = weights[:]
    # Index map: 0=loss, 1..k = multipliers from small to large (LDW lists have sub-1 first)
    # Shift mass from tail to head when bias is negative; invert when positive.
    total = max(1, sum(w))
    # Scale ends more than middle
    for i in range(len(w)):
        pos = i / (len(w) - 1 if len(w) > 1 else 1)
        edge_intensity = 4 * (pos - 0.5) ** 2  # 0..1 bell at edges
        if i == 0:
            # Loss bucket: grow when we need to tighten (negative bias), shrink when relaxing.
            factor = 1.0 - bias * 1.5  # bias>0 => smaller loss; bias<0 => larger loss
        else:
            # Non-loss: tilt tail according to bias (more weight on higher multipliers when bias>0)
            tailness = pos  # closer to 1 for big multipliers
            factor = 1.0 + bias * (0.4 + 1.1 * tailness) * (0.5 + 0.5 * edge_intensity)
        w[i] = max(0, int(round(w[i] * factor)))

    # Keep at least one winning bucket if any existed
    if sum(w[1:]) == 0 and sum(weights[1:]) > 0:
        w[1] = max(1, w[1])

    # Normalize scale (optional). We’ll leave absolute scale since random.choices only needs relative.
    return w

def maybe_add_volatility_spike(weights: List[int]) -> List[int]:
    if not ENABLE_VOLATILITY_SPIKES:
        return weights
    # Small chance to spike, but only if RTP is below target (safe to be generous)
    if random.random() < 0.06 and observed_rtp() < TARGET_RTP:
        w = weights[:]
        # tilt towards bigger multipliers (upper half)
        mid = len(w) // 2
        for i in range(mid + 1, len(w)):
            w[i] = int(round(w[i] * 1.25))  # +25% on tail
        # keep loss weight slightly reduced
        w[0] = int(round(w[0] * 0.9))
        if sum(w[1:]) == 0 and sum(weights[1:]) > 0:
            w[1] = max(1, w[1])
        return w
    return weights

# =========================
# Core helpers
# =========================
def get_energy_tier(energy: float) -> str:
    # Walk thresholds in order
    for tier, thresh in ENERGY_THRESHOLDS.items():
        if energy <= thresh:
            return tier
    return "ultra_high"

def calculate_dynamic_weights(energy: float, bet_amount: float) -> Tuple[List[int], str]:
    base_tier = get_energy_tier(energy)
    base = WIN_CHANCE_CONFIGS[base_tier]["weights"].copy()

    # Bet-size progressive bonus on winning buckets only
    bet_bonus = min(0.10, max(0.0, bet_amount) / 10000.0)
    for i in range(1, len(base)):
        progressive = 1.0 + bet_bonus * (1.0 + i / (len(base) - 1))
        base[i] = max(0, int(round(base[i] * progressive)))

    # Safety: at least one winning bucket if there used to be one
    if sum(base[1:]) == 0 and sum(WIN_CHANCE_CONFIGS[base_tier]["weights"][1:]) > 0:
        base[1] = 1

    # Apply RTP bias
    biased = apply_house_bias(base)
    # Optional volatility spike
    spiky = maybe_add_volatility_spike(biased)

    return spiky, WIN_CHANCE_CONFIGS[base_tier]["description"]

def select_multiplier_with_energy(energy: float, bet_amount: float) -> Tuple[float, str, List[int]]:
    weights, desc = calculate_dynamic_weights(energy, bet_amount)
    multiplier = random.choices(PRIZE_MULTIPLIERS, weights=weights, k=1)[0]
    return multiplier, desc, weights

# =========================
# Models
# =========================
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

# =========================
# Round resolution
# =========================
def resolve_round(user_id: str, bet_amount: float, pool_before: float, user_energy_before: float) -> PlayResponse:
    fee = round(bet_amount * SYSTEM_FEE_RATE, 2)
    effective_bet = round(bet_amount - fee, 2)
    available = max(0.0, round(pool_before + effective_bet, 2))

    drawn_mult, win_desc, _weights = select_multiplier_with_energy(user_energy_before, effective_bet)
    drawn_prize = round(effective_bet * drawn_mult, 2)

    if drawn_prize <= available:
        prize, eff_mult = drawn_prize, drawn_mult
    else:
        # Respect pool constraint; fallback to micro/zero prize if not enough liquidity
        min_prize = round(effective_bet * MIN_MICRO_WIN, 2)
        prize, eff_mult = (min_prize, MIN_MICRO_WIN) if min_prize <= available else (0.0, 0.0)

    pool_after = round(available - prize, 2)
    # Energy mirrors net house gain: bet (after fee) - prize
    round_energy = round(effective_bet - prize, 2)
    total_energy_after = round(float(user_energy_before or 0.0) + round_energy, 2)

    # Update RTP stats
    recent_rounds.append((effective_bet, prize))

    return PlayResponse(
        userId=user_id,
        status="win" if prize > 0 else "loss",
        prizeAmount=prize,
        multiplier=eff_mult,
        poolAfter=pool_after,
        cashWin=prize > 0,
        systemFee=fee,
        effectiveBet=effective_bet,
        roundEnergy=round_energy,
        totalEnergyAfter=total_energy_after,
        energyTier=get_energy_tier(user_energy_before),
        winChanceDescription=win_desc
    )

# =========================
# FastAPI App
# =========================
app = FastAPI(
    title="Energy-Scaled Pool (RTP-Governed)",
    version="4.0.0",
    description=(
        "Higher energy ⇒ higher win chance & larger multipliers. "
        "RTP governor keeps long-run RTP near target. "
        "Optional LDW and volatility spikes for engagement."
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
        "targetRTP": TARGET_RTP,
        "observedRTP": round(observed_rtp(), 4),
        "rtpWindow": RTP_WINDOW,
        "ldwEnabled": ENABLE_LDW,
        "volatilitySpikes": ENABLE_VOLATILITY_SPIKES,
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
        "ultra_low":  "Protection max — odds low; prizes tiny.",
        "very_low":   "Protection — lower odds; small prizes.",
        "low":        "Defense — reduced odds; big multipliers rare.",
        "medium":     "Balanced — fair odds and sizes.",
        "mid_high":   "Slight reward — above-average odds.",
        "high":       "Reward — higher odds and larger multipliers.",
        "very_high":  "VIP — excellent odds; strong tails.",
        "ultra_high": "Super VIP — best odds; biggest tails.",
    }.get(tier, "Unknown energy tier")

@app.get("/energy-analysis/{energy}")
def analyze_energy_tier(energy: float):
    tier = get_energy_tier(energy)
    cfg = WIN_CHANCE_CONFIGS[tier]
    base_weights = cfg["weights"]
    # Show user the current live-biased weights too
    biased = apply_house_bias(base_weights)
    spiky = maybe_add_volatility_spike(biased)

    def metrics(weights: List[int]):
        total = sum(weights)
        if total <= 0:
            return 0.0, 0.0
        win_weights = sum(weights[1:])
        win_pct = (win_weights / total) * 100
        exp_mult = sum(PRIZE_MULTIPLIERS[i] * weights[i] for i in range(len(weights))) / total
        return round(win_pct, 2), round(exp_mult, 4)

    win_pct_base, exp_mult_base = metrics(base_weights)
    win_pct_live, exp_mult_live = metrics(spiky)

    return {
        "energy": energy,
        "tier": tier,
        "description": cfg["description"],
        "targetRTP": TARGET_RTP,
        "observedRTP": round(observed_rtp(), 4),
        "weightsBase": base_weights,
        "weightsLive": spiky,
        "winPercentageBase": win_pct_base,
        "expectedMultiplierBase": exp_mult_base,
        "winPercentageLive": win_pct_live,
        "expectedMultiplierLive": exp_mult_live,
        "recommendation": get_energy_recommendation(tier)
    }
