from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis
import pandas as pd
import numpy as np
from typing import Tuple, Any
import matplotlib.pyplot as plt

def Cox_regression(X_train : pd.DataFrame,
                   Y_train : pd.DataFrame,
                   X_test : pd.DataFrame,
                   title: str
                   ) -> Tuple[pd.DataFrame,
                              np.ndarray,
                              np.ndarray,
                              np.ndarray]:
    
    alphas = 10.0 ** np.linspace(-4,4, 50)
    
    betas = dict()
    
    chp = CoxPHSurvivalAnalysis()
    
    for alpha in alphas:
        chp.set_params(alpha=alpha)
        chp.fit(X_train, Y_train)
        key = round(alpha, 5)
        betas[key] = chp.coef_
    
    betas = (pd.DataFrame.from_dict(betas)
             .rename_axis(index="feature", columns="alpha")
             .set_index(X_train.columns))
    
    chp_predict = chp.predict(X_test)
    chp_survival_curve = chp.predict_survival_function(X_test)
    chp_risk_curve = chp.predict_cumulative_hazard_function(X_test)
    for fn in chp_survival_curve:
        plt.step(fn.x, fn(fn.x), where="post")
    plt.title(f"Survival curve for {title}")
    plt.xlabel("Days")
    plt.ylim(0, 1)
    plt.ylabel("% of survival")
    plt.show()
    
    for fn in chp_risk_curve:
        plt.step(fn.x, fn(fn.x), where="post")
    plt.title(f"Risk curve for {title}")
    plt.xlabel("Months")
    plt.ylim(0, 1)
    plt.ylabel("% of risk")
    plt.show()
    
    
    return (betas, chp_predict, chp_survival_curve, chp_risk_curve)

        
def p_values_Cox_regression(df: pd.DataFrame, 
                            event_col : str, 
                            duration_col : str) -> pd.DataFrame:
    pvalue_Cox = CoxPHFitter()
    pvalue_Cox.fit(df, event_col=event_col, duration_col=duration_col)
    
    return pvalue_Cox.summary
    
    
    
def plot_coefficients(coefs, n_highlight, title:str):
    _, ax = plt.subplots(figsize=(9, 6))
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
    plt.title(title)    
