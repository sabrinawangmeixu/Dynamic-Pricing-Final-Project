import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.ensemble import HistGradientBoostingClassifier

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
        # read historical data from csvs (following professor's example)
        demands = pd.read_csv("historical_demands.csv", header=None)
        prices = pd.read_csv("../historical_prices.csv", header=None)
        
        # Set column names for prices: team 1, 2, 3, ...
        prices.columns = np.arange(len(prices.columns)) + 1

        # demands: single column of demand counts (0, 1, 2, 3, ...)
        # Original: outcomes = demands.iloc[:, 0].values.astype(float)
        # Temporarily convert counts to binary (0 = no demand, >0 = some demand)
        # TODO: Once TA clarifies the scale, convert to actual demand/purchase ratios
        demand_counts = demands.iloc[:, 0].values.astype(float)
        #outcomes = (demand_counts > 0).astype(float)
        outcomes = demand_counts

        # prices: row i = [my_price, price_team_2, price_team_3, ...]
        my_prices = prices.iloc[:, 0].values.astype(float)
        # comp_prices = prices from all teams except team 1
        all_prices = prices.values.astype(float)
        comp_prices = all_prices[:, 1:]  # all columns except team 1

        n = min(len(my_prices), len(comp_prices), len(outcomes))
        my_prices = my_prices[:n]
        outcomes = outcomes[:n]
        comp_prices = comp_prices[:n]

        mask = (
            (my_prices >= PRICE_MIN) &
            (my_prices <= PRICE_MAX)
        )

        return {
            "t": int(np.sum(mask)),
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

def need_catchup_exploration(prices, min_unique=20):
    if len(prices) == 0:
        return True

    unique_prices = np.unique(np.round(prices, 2))

    # If we mostly only played 50, we did not explore enough
    return len(unique_prices) < min_unique

# -----------------------
# Phase 1: Exploration 
# -----------------------
# def phase1_strategy(prices, outcomes):
#     decisions = len(prices)

#     if decisions == 0: # if no data, return 50 as default
#         return 50.0

#     # ---- Phase 1: systematic cycling through exploration grid ----
#     # Cycle through grid to ensure diverse exploration
#     grid_index = decisions % len(EXPLORATION_GRID)
#     return float(EXPLORATION_GRID[grid_index])

def phase1_strategy(prices, outcomes):
    unique_prices = set(np.round(prices, 2))

    for p in EXPLORATION_GRID:
        if round(p, 2) not in unique_prices:
            return float(p)

    # If all grid points already tried, fallback
    return 50.0

# -----------------------
# Phase 2: 
# -----------------------
def phase2_strategy(prices, outcomes, comp_prices):
    try:
        t = len(prices)
        if t == 0:
            return 50.0

        competitor_median_array = np.median(comp_prices, axis=1)

        '''
        # MLE logistic regression
        X = np.column_stack([prices, competitor_median_array])
        y = outcomes

        model = LogisticRegression(fit_intercept=True, C=10.0, max_iter=1000)
        model.fit(X, y)

        coef = model.coef_[0]
        intercept = model.intercept_[0]

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
        '''

        # Thompson Sampling with Laplace approximation
        y_log = np.log(outcomes + 1.0) #in case demand = 0
        X_log = np.column_stack([
            np.ones(t), 
            np.log(prices), 
            np.log(competitor_median_array + 1e-5) # prevending error
        ])

        # get the distribution of parameters 
        precision_n = X_log.T @ X_log + np.eye(3) * 0.1
        cov_n = np.linalg.inv(precision_n)
        mu_n = cov_n @ (X_log.T @ y_log)

        # Thompson Sampling: sampling one beta from the parameter distribution 
        beta = np.random.multivariate_normal(mu_n, cov_n)
        
        if t < 60:
            beta[1] = min(beta[1], -0.05)
        else:
            beta[1] = min(beta[1], -0.02)
        # if beta[1] > 0: 
        #     beta[1] = -0.01 
            # make sure the coefficient of our own price is negative

        # Part 2: get opt price
        last_comp_median_p = competitor_median_array[-1]

        best_p = 50.0
        max_rev = -1.0

        # find the optimal price between 1 and 100 that max price * prob of buying 
        for p_test in np.linspace(1.0, 100.0, 100):
            log_demand = beta[0] + beta[1] * np.log(p_test) + beta[2] * np.log(last_comp_median_p) 
            expected_demand = np.exp(log_demand)

            expected_revenue = p_test * expected_demand
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

def local_kernel_demand(candidate_price, hist_prices, hist_demands, hist_comp_prices,
                        current_comp_row, k_neighbors=80, price_bandwidth=6.0):

    n = len(hist_prices)
    if n < 20:
        return None

    hist_med = np.median(hist_comp_prices, axis=1)
    hist_min = np.min(hist_comp_prices, axis=1)
    hist_max = np.max(hist_comp_prices, axis=1)
    hist_std = np.std(hist_comp_prices, axis=1)

    hist_context = np.column_stack([hist_med, hist_min, hist_max, hist_std])

    curr_context = np.array([
        np.median(current_comp_row),
        np.min(current_comp_row),
        np.max(current_comp_row),
        np.std(current_comp_row)
    ], dtype=float)

    means = np.mean(hist_context, axis=0)
    stds = np.std(hist_context, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)

    hist_z = (hist_context - means) / stds
    curr_z = (curr_context - means) / stds

    dists = np.linalg.norm(hist_z - curr_z, axis=1)

    k_eff = min(k_neighbors, n)
    idx = np.argsort(dists)[:k_eff]

    local_prices = hist_prices[idx]
    local_demands = hist_demands[idx]

    weights = np.exp(-0.5 * ((local_prices - candidate_price) / price_bandwidth) ** 2)

    if np.sum(weights) < 1e-10:
        return None

    demand_local = float(np.sum(weights * local_demands) / np.sum(weights))

    return max(demand_local, 0.0)

def phase3_strategy(prices, outcomes, comp_prices):
    try:
        n = len(prices)

        # outcomes are now demand counts
        demands = outcomes.astype(float)

        if n < 30:
            return phase2_strategy(prices, outcomes, comp_prices)

        X = build_features(prices, comp_prices)
        y = demands

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.03,
            max_iter=150,
            max_depth=3,
            min_samples_leaf=8,
            random_state=0
        )

        model.fit(X, y)

        current_comp_row = comp_prices[-1]

        best_p = 50.0
        best_score = -1.0

        coarse_grid = np.linspace(1.0, 100.0, 50)

        for p in coarse_grid:
            x_candidate = build_candidate_feature(float(p), current_comp_row)

            demand_global = float(model.predict(x_candidate)[0])
            demand_global = max(demand_global, 0.0)

            k = min(40, n // 2)

            demand_local = local_kernel_demand(
                candidate_price=float(p),
                hist_prices=prices,
                hist_demands=demands,
                hist_comp_prices=comp_prices,
                current_comp_row=current_comp_row,
                k_neighbors=k,
                price_bandwidth=6.0
            )

            if demand_local is None:
                demand_hybrid = demand_global
            else:
                lam = 0.6 if n < 150 else 0.35
                demand_hybrid = lam * demand_global + (1.0 - lam) * demand_local

            demand_hybrid = max(float(demand_hybrid), 0.0)

            score = float(p) * demand_hybrid

            if score > best_score:
                best_score = score
                best_p = float(p)

        low = max(PRICE_MIN, best_p - 5.0)
        high = min(PRICE_MAX, best_p + 5.0)
        fine_grid = np.linspace(low, high, 41)

        for p in fine_grid:
            x_candidate = build_candidate_feature(float(p), current_comp_row)

            demand_global = float(model.predict(x_candidate)[0])
            demand_global = max(demand_global, 0.0)

            k = min(40, n // 2)

            demand_local = local_kernel_demand(
                candidate_price=float(p),
                hist_prices=prices,
                hist_demands=demands,
                hist_comp_prices=comp_prices,
                current_comp_row=current_comp_row,
                k_neighbors=k,
                price_bandwidth=6.0
            )

            if demand_local is None:
                demand_hybrid = demand_global
            else:
                lam = 0.6 if n < 150 else 0.35
                demand_hybrid = lam * demand_global + (1.0 - lam) * demand_local

            demand_hybrid = max(float(demand_hybrid), 0.0)

            score = float(p) * demand_hybrid

            if score > best_score:
                best_score = score
                best_p = float(p)

        return round(float(np.clip(best_p, PRICE_MIN, PRICE_MAX)), 2)

    except Exception as e:
        print("Phase 3 Error:", e, flush=True)
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

        # Catch-up exploration since we missed Phase 1
        if need_catchup_exploration(prices, min_unique=20):
            return phase1_strategy(prices, outcomes)

        # if t <= T_PHASE1:
        #     return phase1_strategy(prices, outcomes)

        elif t <= T_PHASE2:
            return phase2_strategy(prices, outcomes, comp_prices)

        else:
            return phase3_strategy(prices, outcomes, comp_prices)

    except Exception as e: 
        return 50.0
    

