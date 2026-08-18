"""Feature-based Bradley-Terry with a positional bias.

Single source of truth for the fit used by the laptop notebooks. It grew up as a copy
pasted between four notebooks that then drifted apart (different constructor arguments,
and two different names for the intercept); this is the union of those copies.

The model, for one ordered pair (a, b):

    score_a - score_b  =  beta0 + sum_f [ w_f(a_f) - w_f(b_f) ]  +  eps

fitted by OLS on the margins. Two things about the result are easy to get wrong:

1. Weights are only identified up to a constant per feature, so each feature is pinned to
   mean zero. A weight then reads as "above / below the average level of this feature",
   and no level is an arbitrary reference point.

2. `beta0_` is NOT the positional bias. Every constant lands in the intercept, including
   the PMI offset C that the old collection code subtracted from every stored margin:

       beta0 = gamma - C     ->     gamma = beta0 + C

   Pass the recovered `pmi_c` and `gamma_` is the real positional bias. For runs collected
   after PMI was turned off (2026-08-12), C is 0 and the two coincide.

Full notation and a worked synthetic example: Notebooks/explainer_metrics.ipynb.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


class FeatureBT:
    """Fit one run. See the module docstring for what the parameters mean.

    Parameters
    ----------
    pmi_c : float
        The PMI constant C for this run, so that `gamma_ = beta0_ + pmi_c`. Use 0.0 for
        runs collected with `normalize_pmi=False`; recover it from saturation otherwise.
    r2_warn : float or None
        Warn when the fit's R^2 falls below this. None disables the check.
    pvalue_warn : float or None
        Warn (only when `verbose`) about levels whose coefficient p-value exceeds this.
        Those p-values are relative to the dropped reference level, not to the centred
        weights, so they answer "is this level different from the reference?".
    verbose : bool
        Enable the per-level p-value warning.

    Attributes
    ----------
    weights_ : dict[str, dict[str, float]]
        weights_[feature][level], centred to mean 0 within each feature. Every level of
        every feature is present, including the one OLS dropped (it is added back at 0
        before centring).
    beta0_ : float
        The fitted intercept, = gamma - C. Also available as `gamma_fit_`.
    gamma_ : float
        beta0_ + pmi_c: the positional bias itself.
    r2_, resid_sd_, result_, pvalues_, n_obs_
        Fit diagnostics. `result_` is the raw statsmodels result.
    """

    def __init__(self, pmi_c=0.0, r2_warn=0.5, pvalue_warn=None, verbose=False):
        self.pmi_c = pmi_c
        self.r2_warn = r2_warn
        self.pvalue_warn = pvalue_warn
        self.verbose = verbose

    def fit(self, df, a_prefix="a_", b_prefix="b_",
            score_a_col="score_a", score_b_col="score_b"):
        n = len(df)
        Y = (df[score_a_col] - df[score_b_col]).values

        cols_a = [c for c in df.columns if c.startswith(a_prefix)]
        cols_b = [c.replace(a_prefix, b_prefix, 1) for c in cols_a]
        self.features_ = [c[len(a_prefix):] for c in cols_a]

        side_a = df[cols_a].rename(columns=dict(zip(cols_a, self.features_)))
        side_b = df[cols_b].rename(columns=dict(zip(cols_b, self.features_)))
        combined = pd.concat([side_a, side_b], axis=0)

        # drop_first=True pins one level per feature to 0 so the design matrix is full
        # rank; with full dummies the columns of each feature sum to 1 and OLS has no
        # unique solution. The dropped level is added back and everything re-centred.
        dummies = pd.get_dummies(combined, drop_first=True, dtype=float)
        X = sm.add_constant(dummies.iloc[:n].values - dummies.iloc[n:].values)

        self.result_ = sm.OLS(Y, X).fit()
        self.beta0_ = self.result_.params[0]
        self.gamma_ = self.beta0_ + self.pmi_c
        self.r2_ = self.result_.rsquared
        self.resid_sd_ = float(np.std(self.result_.resid))
        self.n_obs_ = n

        names = dummies.columns.tolist()
        raw = dict(zip(names, self.result_.params[1:]))
        self.pvalues_ = dict(zip(names, self.result_.pvalues[1:]))

        self.weights_ = {}
        for feat in self.features_:
            w = {lvl: raw.get(f"{feat}_{lvl}", 0.0) for lvl in combined[feat].unique()}
            mean = np.mean(list(w.values()))
            self.weights_[feat] = {lvl: v - mean for lvl, v in w.items()}

        if self.r2_warn is not None and self.r2_ < self.r2_warn:
            warnings.warn(f"low fit: R^2 = {self.r2_:.3f} < {self.r2_warn}")
        if self.verbose and self.pvalue_warn is not None:
            weak = {k: p for k, p in self.pvalues_.items() if p > self.pvalue_warn}
            if weak:
                warnings.warn("weak levels (vs reference): "
                              + ", ".join(f"{k} p={p:.3f}" for k, p in weak.items()))
        return self

    # `gamma_fit_` is the older name for the intercept, kept so existing notebooks keep
    # working. `beta0_` is preferred: it is the OLS intercept, and calling it a gamma is
    # exactly the confusion that hid the PMI bug for months.
    @property
    def gamma_fit_(self):
        return self.beta0_

    def utility(self, item):
        """u(x) for one laptop, e.g. {"brand": "Apple", "screen": "14-inch", "ram": "16GB"}.

        An unknown feature or level raises KeyError on purpose - it is nearly always a
        typo, and silently returning 0 would hide it.
        """
        return sum(self.weights_[f][item[f]] for f in self.features_)

    def predict(self, item_a, item_b):
        """P(A beats B), position-free (gamma excluded).

        The ORDER is trustworthy; the calibration is not. The weight scale is a nuisance
        parameter here (docs/progress.md 2026-08-12), so do not quote this as a
        calibrated probability.
        """
        return 1.0 / (1.0 + np.exp(-(self.utility(item_a) - self.utility(item_b))))

    def range(self, feat):
        """Best level minus worst level, for one feature."""
        w = self.weights_[feat]
        return max(w.values()) - min(w.values())

    def total_spread(self):
        """S: the sum of the per-feature ranges.

        Equivalently u(best laptop) - u(worst laptop), exactly, when the item set is a
        full factorial - which is how to read it.
        """
        return sum(self.range(f) for f in self.features_)

    def normalised(self, feat, level):
        """w / S: this level's share of everything the model cares about.

        The scale-free quantity to compare ACROSS models; raw weights are not comparable
        because a louder model is not a model with different tastes.
        """
        return self.weights_[feat][level] / self.total_spread()
