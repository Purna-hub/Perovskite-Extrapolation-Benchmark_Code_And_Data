"""
Evaluation protocols, model zoo and split-conformal
uncertainty quantification.

quantification for the HOIP multi-fidelity benchmark.

Split protocols
---------------
random   : i.i.d. 5-fold. The protocol used by essentially all published HOIP
           band-gap ML studies. Polymorphs of the same ABX3 composition are
           distributed across train and test folds.
group    : 5-fold GroupKFold grouped on the 192 ABX3 compositions. Removes
           polymorph leakage but keeps all 16 cations / 3 metals / 4 halides
           represented in training.
LOCO     : leave-one-cation-out (16 folds). Tests extrapolation to an unseen
           A-site organic cation.
LOBO     : leave-one-B-out (3 folds). Unseen group-IV metal.
LOHO     : leave-one-halide-out (4 folds). Unseen halide chemistry -- the
           hardest and most discovery-relevant regime.

Part of the reproducibility package for:
  Large language model descriptors and multi-fidelity learning for HSE06 band gaps
  of hybrid halide perovskites under chemical extrapolation.

Paths are resolved through paths.py, so this runs from any working directory.
"""
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import lightgbm as lgb
import xgboost as xgb


# ----------------------------------------------------------------- splits ---
def make_splits(df, seed=0):
    n = len(df)
    idx = np.arange(n)
    splits = {}
    splits["random"] = list(KFold(5, shuffle=True, random_state=seed).split(idx))
    splits["group"] = list(GroupKFold(5).split(idx, groups=df.composition))
    for name, col in (("LOCO", "A_cation"), ("LOBO", "B"), ("LOHO", "X")):
        splits[name] = [(np.where(df[col] != v)[0], np.where(df[col] == v)[0])
                        for v in sorted(df[col].unique())]
    return splits


# ------------------------------------------------------------------ models ---
def _pipe(est, scale=True):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("est", est))
    return Pipeline(steps)


def model_zoo(seed=0):
    return {
        "Ridge": _pipe(RidgeCV(alphas=np.logspace(-4, 4, 40))),
        "KRR-RBF": _pipe(KernelRidge(kernel="rbf", alpha=1e-2, gamma=1e-2)),
        "_skipGPR": _pipe(GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(1e-2),
            normalize_y=True, alpha=1e-6, random_state=seed)),
        "MLP": _pipe(MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=400,
                                  early_stopping=True, n_iter_no_change=40,
                                  learning_rate_init=3e-3, alpha=1e-3,
                                  random_state=seed)),
        "RF": _pipe(RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_depth=20,
                                          n_jobs=1, random_state=seed), scale=False),
        "_skipET": _pipe(ExtraTreesRegressor(n_estimators=300, n_jobs=1,
                                                random_state=seed), scale=False),
        "XGBoost": _pipe(xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                          max_depth=6, subsample=0.85,
                                          colsample_bytree=0.8, reg_lambda=1.0,
                                          n_jobs=1, random_state=seed,
                                          tree_method="hist"), scale=False),
        "LightGBM": _pipe(lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                            num_leaves=31, min_child_samples=8,
                                            subsample=0.85, subsample_freq=1,
                                            colsample_bytree=0.8, reg_lambda=1.0,
                                            n_jobs=1, random_state=seed,
                                            verbose=-1), scale=False),
    }


# ------------------------------------------------------------------ metrics --
def metrics(y, yp):
    err = y - yp
    ss = ((y - y.mean()) ** 2).sum()
    return dict(MAE=float(np.abs(err).mean()),
                RMSE=float(np.sqrt((err ** 2).mean())),
                R2=float(1 - (err ** 2).sum() / ss),
                MaxAE=float(np.abs(err).max()))


def cross_validate(model_fn, X, y, split, delta_base=None, y_true=None):
    """Out-of-fold predictions. If delta_base is given, the model targets the
    residual y (= dEg) and predictions are reconstructed as base + dEg_hat."""
    yp = np.full(len(y), np.nan)
    for tr, te in split:
        m = model_fn()
        m.fit(X[tr], y[tr])
        yp[te] = m.predict(X[te])
    if delta_base is not None:
        yp = yp + delta_base
        y = y_true
    return y, yp


# -------------------------------------------------------- conformal UQ ------
def split_conformal(model_fn, X, y, split, alpha=0.1, cal_frac=0.3, seed=0,
                    delta_base=None, y_true=None, groups=None):
    """Split-conformal prediction intervals evaluated out-of-fold.

    Within each training fold a calibration subset is held out to estimate the
    (1-alpha) quantile of absolute residuals; the resulting half-width is applied
    to the test fold. When `groups` is supplied the calibration split is made
    group-wise so that calibration residuals are themselves out-of-composition,
    which is what makes the interval valid under the grouped/OOD protocols.
    """
    rng = np.random.default_rng(seed)
    lo = np.full(len(y), np.nan)
    hi = np.full(len(y), np.nan)
    width = np.full(len(y), np.nan)
    for tr, te in split:
        if groups is None:
            perm = rng.permutation(tr)
            ncal = max(20, int(cal_frac * len(tr)))
            cal, fit = perm[:ncal], perm[ncal:]
        else:
            g = np.array(groups)[tr]
            ug = rng.permutation(np.unique(g))
            ncalg = max(2, int(cal_frac * len(ug)))
            calg = set(ug[:ncalg])
            mask = np.array([gi in calg for gi in g])
            cal, fit = tr[mask], tr[~mask]
            if len(cal) < 20 or len(fit) < 20:
                perm = rng.permutation(tr)
                ncal = max(20, int(cal_frac * len(tr)))
                cal, fit = perm[:ncal], perm[ncal:]
        m = model_fn()
        m.fit(X[fit], y[fit])
        res = np.abs(y[cal] - m.predict(X[cal]))
        n = len(res)
        q = np.quantile(res, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
        pred = m.predict(X[te])
        lo[te], hi[te] = pred - q, pred + q
        width[te] = 2 * q
    if delta_base is not None:
        lo, hi = lo + delta_base, hi + delta_base
        y = y_true
    cov = float(((y >= lo) & (y <= hi)).mean())
    return dict(coverage=cov, mean_width=float(np.nanmean(width)),
                nominal=1 - alpha), lo, hi
