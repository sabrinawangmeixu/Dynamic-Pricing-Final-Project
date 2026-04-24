import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# -----------------------
# Constants
# -----------------------
PRICE_MIN = 1.0 # prices are constrained to the range [1,100]
PRICE_MAX = 100.0

T_PHASE1 = 25    # end of Phase 1 (exploration)
T_PHASE2 = 100   # end of Phase 2; Phase 3 activates at t≥100, reachable in 168-period game

EXPLORATION_GRID = [round(float(p), 2) for p in np.linspace(1.0, 100.0, T_PHASE1)]

# -----------------------
# Data Loading
# -----------------------
def load_data():
    try:
        # read historical data from csvs 
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

    if decisions == 0: # if no data, return 50 as default
        return 50.0

    # ---- Phase 1: systematic cycling through exploration grid ----
    # Cycle through grid to ensure diverse exploration
    grid_index = decisions % len(EXPLORATION_GRID)
    return float(EXPLORATION_GRID[grid_index])

# -----------------------
# Phase 2: 
# -----------------------
def phase2_strategy(prices, outcomes, comp_prices):
    try:
        if len(prices) == 0:
            return 50.0

        competitor_median_array = np.median(comp_prices, axis=1)
        X = np.column_stack([prices, competitor_median_array])
        y = outcomes

        # fit_intercept=True lets sklearn handle the constant term properly
        model = LogisticRegression(fit_intercept=True, C=10.0, max_iter=1000)
        model.fit(X, y)

        coef = model.coef_[0]         # [price_coef, comp_median_coef]
        intercept = model.intercept_[0]

        # Use mean historical competitor median — last period alone is one noisy draw
        mean_comp_median = float(np.mean(competitor_median_array))

        best_p = 50.0
        max_rev = -1.0

        for p_test in np.linspace(1.0, 100.0, 200):
            logit = intercept + coef[0] * p_test + coef[1] * mean_comp_median
            prob_buy = 1.0 / (1.0 + np.exp(-logit))
            expected_revenue = p_test * prob_buy
            if expected_revenue > max_rev:
                max_rev = expected_revenue
                best_p = p_test

        return round(float(best_p), 2)

    except Exception:
        return 50.0


# -----------------------
# Phase 3: 
# -----------------------

"""
At time t,
1) summarize current competitor context
- for each period t, from comp_prices[t], compute:competitor median/min/max//std
2) fit or update global contextual model
- for candidate price p, features include both competitor context and how our price compares to that context
3) build local neighborhood of past periods with similar competitor context
4) for each candidate price p, estimate purchase probability in 2 ways:
    - global estimate
    - local estimate
5) combine these estimates
6) choose price maximizing expected revenue

for candidate price p: q^​hybrid​(p)=λq^​global​(p)+(1−λ)q^​local​(p)
and then choose p* = arg max_p [p * q^​hybrid​(p ]
"""

# Helper function to summarize competitor prices row-wise
def summarize_competitors(comp_prices):
    comp_med = np.median(comp_prices, axis=1)
    comp_min = np.min(comp_prices, axis=1)
    comp_max = np.max(comp_prices, axis=1)
    comp_mean = np.mean(comp_prices, axis=1)
    comp_std = np.std(comp_prices, axis=1)
    return comp_med, comp_min, comp_max, comp_mean, comp_std

# Helper function to build historical feature matrix
def build_features(prices, comp_prices):
    comp_med, comp_min, comp_max, comp_mean, comp_std = summarize_competitors(comp_prices)
    gap_med = prices - comp_med
    gap_min = prices - comp_min
    gap_max = prices - comp_max

    is_cheapest = (prices <= comp_min).astype(float)

    # proportion of competitors cheaper than our team
    prop_below = np.mean(comp_prices < prices.reshape(-1,1), axis=1)

    X = np.column_stack([
        prices,
        comp_med,
        comp_min,
        comp_max,
        comp_std,
        gap_med,
        gap_min,
        gap_max,
        gap_med ** 2,
        gap_min ** 2,
        is_cheapest,
        prop_below
    ])
    return X 

# Helper function to build 1 candidate feature row
def build_candidate_feature(p, current_comp_row):
    comp_med = float(np.median(current_comp_row))
    comp_min = float(np.min(current_comp_row))
    comp_max = float(np.max(current_comp_row))
    comp_std = float(np.std(current_comp_row))

    gap_med = p - comp_med
    gap_min = p - comp_min
    gap_max = p - comp_max

    is_cheapest = float(p <= comp_min)
    frac_below = float(np.mean(current_comp_row < p))

    x = np.array([[
        p,
        comp_med,
        comp_min,
        comp_max,
        comp_std,
        gap_med,
        gap_min,
        gap_max,
        gap_med ** 2,
        gap_min ** 2,
        is_cheapest,
        frac_below
    ]], dtype=float)

    return x

# Helper function: compute local probability estimate 

def local_kernel_prob(candidate_price, hist_prices, hist_outcomes, hist_comp_prices, current_comp_row, k_neighbors = 80, price_bandwidth = 6.0):
    """ 
    Estimate local acceptance probability at candidate_price by:
    1) selecting historical periods with similar competitor context
    2) kernel-smoothing over our historical prices within that neighborhood
    """
    
    n = len(hist_prices)
    if n < 20:
        return None

    # Historical competitor context
    hist_med = np.median(hist_comp_prices, axis=1)
    hist_min = np.min(hist_comp_prices, axis=1)
    hist_max = np.max(hist_comp_prices, axis=1)
    hist_std = np.std(hist_comp_prices, axis=1)

    hist_context = np.column_stack([hist_med, hist_min, hist_max, hist_std])

    # Current competitor context
    curr_med = float(np.median(current_comp_row))
    curr_min = float(np.min(current_comp_row))
    curr_max = float(np.max(current_comp_row))
    curr_std = float(np.std(current_comp_row))
    curr_context = np.array([curr_med, curr_min, curr_max, curr_std], dtype=float)

    # Standardize context dimensions so one variable doesn't dominate
    means = np.mean(hist_context, axis=0)
    stds = np.std(hist_context, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)

    hist_z = (hist_context - means) / stds
    curr_z = (curr_context - means) / stds

    dists = np.linalg.norm(hist_z - curr_z, axis=1)

    k_eff = min(k_neighbors, n)
    idx = np.argsort(dists)[:k_eff]

    local_prices = hist_prices[idx]
    local_outcomes = hist_outcomes[idx]

    # Gaussian kernel in our own price dimension
    price_diff = local_prices - candidate_price
    weights = np.exp(-0.5 * (price_diff / price_bandwidth) ** 2)

    # Tiny floor so we don't divide by zero
    if np.sum(weights) < 1e-10:
        return None

    q_local = float(np.sum(weights * local_outcomes) / np.sum(weights))
    q_local = float(np.clip(q_local, 0.0, 1.0))
    return q_local


def phase3_strategy(prices, outcomes, comp_prices):
    try:
        n = len(prices)
        # Need both classes present to fit classifier
        if len(np.unique(outcomes)) < 2:
            return phase2_strategy(prices, outcomes, comp_prices)

        # Build historical training features
        X = build_features(prices, comp_prices)
        y = outcomes.astype(int)

        # Global model
        model = HistGradientBoostingClassifier(
            loss='log_loss',
            learning_rate=0.03,
            max_iter=150,
            max_depth=3,
            min_samples_leaf=8,
            random_state=0
        )
        model.fit(X, y)

        # Use mean historical competitor row — last period alone is one noisy draw
        current_comp_row = np.mean(comp_prices, axis=0)

        # Candidate price search
        coarse_grid = np.linspace(1.0, 100.0, 50)

        best_p = 50.0
        best_score = -1.0

        for p in coarse_grid:
            x_candidate = build_candidate_feature(float(p), current_comp_row)

            q_global = float(model.predict_proba(x_candidate)[0, 1])

            k = min(40, n // 2)
            q_local = local_kernel_prob(
                candidate_price=float(p),
                hist_prices=prices,
                hist_outcomes=outcomes,
                hist_comp_prices=comp_prices,
                current_comp_row=current_comp_row,
                k_neighbors=k,
                price_bandwidth=6.0
            )

            # Blend global + local
            if q_local is None:
                q_hybrid = q_global
            else:
                lam = 0.6 if n < 150 else 0.35
                q_hybrid = lam * q_global + (1.0 - lam) * q_local

            q_hybrid = float(np.clip(q_hybrid, 0.0, 1.0))
            score = float(p) * q_hybrid

            if score > best_score:
                best_score = score
                best_p = float(p)

        # Fine search around the best coarse price
        low = max(PRICE_MIN, best_p - 5.0)
        high = min(PRICE_MAX, best_p + 5.0)
        fine_grid = np.linspace(low, high, 41)

        for p in fine_grid:
            x_candidate = build_candidate_feature(float(p), current_comp_row)

            q_global = float(model.predict_proba(x_candidate)[0, 1])

            k = min(40, n // 2)
            q_local = local_kernel_prob(
                candidate_price=float(p),
                hist_prices=prices,
                hist_outcomes=outcomes,
                hist_comp_prices=comp_prices,
                current_comp_row=current_comp_row,
                k_neighbors=k,
                price_bandwidth=6.0
            )

            if q_local is None:
                q_hybrid = q_global
            else:
                lam = 0.6 if n < 150 else 0.35
                q_hybrid = lam * q_global + (1.0 - lam) * q_local

            q_hybrid = float(np.clip(q_hybrid, 0.0, 1.0))
            score = float(p) * q_hybrid

            if score > best_score:
                best_score = score
                best_p = float(p)
        return round(float(np.clip(best_p, PRICE_MIN, PRICE_MAX)), 2)

    except Exception as e:
        print("Phase 3 Error:", e)
        return 50.0

# -----------------------
# Main Strategy:
# -----------------------
def strategy():
    try:
        data = load_data()
        prices = data["my_prices"]
        outcomes = data["outcomes"]
        comp_prices = data["comp_prices"]
        t = data["t"]

        if t <= T_PHASE1:
            return phase1_strategy(prices, outcomes)

        elif t <= T_PHASE2:
            return phase2_strategy(prices, outcomes, comp_prices)

        else:
            return phase3_strategy(prices, outcomes, comp_prices)

    except Exception as e: 
        return 50.0
