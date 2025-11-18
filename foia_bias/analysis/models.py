"""Regression helpers for hypotheses tests."""
from __future__ import annotations

import statsmodels.formula.api as smf


def run_wrongdoing_model(df, include_agency_fe: bool = True, include_year_fe: bool = True):
    formula = "wrongdoing_any ~ same_party"
    if include_agency_fe:
        formula += " + C(agency)"
    if include_year_fe:
        formula += " + C(year)"
    return smf.logit(formula=formula, data=df).fit(disp=0)


def run_favorability_model(df, include_agency_fe: bool = True, include_year_fe: bool = True):
    formula = "fav_diff ~ same_party"
    if include_agency_fe:
        formula += " + C(agency)"
    if include_year_fe:
        formula += " + C(year)"
    return smf.ols(formula=formula, data=df).fit()
