import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

# -----------------------
# Constants
# -----------------------
PRICE_MIN = 1.0
PRICE_MAX = 100.0

INITIAL_THRESHOLD = 40   # grid exploration length
T_PHASE1 = 50         # end of Phase 1
T_PHASE2 = 300           # end of Phase 2

# Log-spaced grid (better for logistic demand)
EXPLORATION_GRID = [round(float(p), 2) for p in np.linspace(2.0, PRICE_MAX, INITIAL_THRESHOLD)]

# -----------------------
# Data Loading
# -----------------------
def load_data():
    try:
        demand_df = pd.read_csv("test_historical_demands.csv", header=None)
        prices_df = pd.read_csv("test_historical_prices.csv", header=None)

        my_prices = demand_df.iloc[:, 0].values.astype(float)
        outcomes  = demand_df.iloc[:, 1].values.astype(float)

        TEAM_IDX = 0
        all_prices = prices_df.values.astype(float)
        comp_prices = np.delete(all_prices, TEAM_IDX, axis=1)

        n = min(len(my_prices), len(comp_prices))
        my_prices   = my_prices[:n]
        outcomes    = outcomes[:n]
        comp_prices = comp_prices[:n]

        mask = (
            (my_prices >= PRICE_MIN) &
            (my_prices <= PRICE_MAX) &
            np.isin(outcomes, [0, 1])
        )

        return {
            "t": np.sum(mask),
            "my_prices": my_prices[mask],
            "outcomes": outcomes[mask],
            "comp_prices": comp_prices[mask]
        }

    except:
        return {
            "t": 0,
            "my_prices": np.array([]),
            "outcomes": np.array([]),
            "comp_prices": np.array([])
        }

# -----------------------
# Warm-up Model (for Phase 2 later)
# -----------------------

'''
might be helpful for phase 2 if we want to do some kind of 
competitor-aware model, but for now we can just use it as a sanity check on the data 
and a fallback if we have enough data but can't fit a model

def fit_model(prices, outcomes):
    if len(np.unique(outcomes)) < 2:
        return None

    X = prices.reshape(-1, 1)

    try:
        model = LogisticRegression(fit_intercept=False, max_iter=2000, C=1e6)
        model.fit(X, outcomes)

        beta = float(-model.coef_[0][0])
        return beta if beta > 0 else None

    except:
        return None

'''

def optimal_price(beta):
    def neg_revenue(p):
        return -p / (1 + np.exp(beta * p))

    res = minimize_scalar(
        neg_revenue,
        bounds=(PRICE_MIN, PRICE_MAX),
        method='bounded'
    )

    return float(np.clip(res.x, PRICE_MIN, PRICE_MAX))


# -----------------------
# Phase 1: Exploration 
# -----------------------
def phase1_strategy(prices, outcomes):
    decisions = len(prices)

    if decisions == 0:
        return 50.0

    # ---- Phase 1: systematic cycling through exploration grid ----
    # Cycle through grid to ensure diverse exploration
    grid_index = decisions % len(EXPLORATION_GRID)
    return float(EXPLORATION_GRID[grid_index])

# -----------------------
# Phase 2: 
# -----------------------
def phase2_strategy(prices, outcomes):
    # TODO: add competitor-aware model + Thompson Sampling

    if len(prices) == 0:
        return 50.0

    # Continue exploring with mix of grid and uniform
    if np.random.rand() < 0.6:
        return float(np.random.choice(EXPLORATION_GRID))
    else:
        return float(np.random.uniform(PRICE_MIN, PRICE_MAX))


# -----------------------
# Phase 3: 
# -----------------------
def phase3_strategy(prices, outcomes):
    # TODO: nonparametric / smoothing method

    return phase2_strategy(prices, outcomes)


# -----------------------
# Main Strategy
# -----------------------
def strategy():
    try:
        data = load_data()
        prices = data["my_prices"]
        outcomes = data["outcomes"]
        t = data["t"]

        if t < T_PHASE1:
            return phase1_strategy(prices, outcomes)

        elif t < T_PHASE2:
            return phase2_strategy(prices, outcomes)

        else:
            return phase3_strategy(prices, outcomes)

    except Exception:
        return 50.0