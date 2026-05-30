from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
import numpy as np
from typing import Tuple, Any, Optional
from pandas import Series
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def Cox_regression(X_train : pd.DataFrame,
                   Y_train : pd.DataFrame,
                   X_test : pd.DataFrame,
                   draw_plot : bool,
                   title: Optional[str] = None,
                   ) -> Tuple[pd.DataFrame,
                              np.ndarray,
                              np.ndarray,
                              np.ndarray]:
    
    
    chp = CoxPHSurvivalAnalysis()

    chp.fit(X_train, Y_train)

    betas = pd.DataFrame(
            chp.coef_,
            index=X_train.columns,
            columns=["beta"]
            )
    
    chp_predict = chp.predict(X_test)
    chp_survival_curve = chp.predict_survival_function(X_test)
    chp_risk_curve = chp.predict_cumulative_hazard_function(X_test)
    
    
    if draw_plot:
        for fn in chp_survival_curve:
            plt.step(fn.x, fn(fn.x), where="post") # type: ignore
        plt.title(f"Survival curve for {title}")
        plt.xlabel("Days")
        plt.ylim(0, 1)
        plt.ylabel("% of survival")
        plt.show()
        
        for fn in chp_risk_curve:
            plt.step(fn.x, fn(fn.x), where="post") # type: ignore
        plt.title(f"Risk curve for {title}")
        plt.xlabel("Months")
        plt.ylim(0, 1)
        plt.ylabel("% of risk")
        plt.show()
        
    
    return (betas, chp_predict, chp_survival_curve, chp_risk_curve)


        
def Cox_l2_regression(X_train : pd.DataFrame,
                   Y_train : pd.DataFrame,
                   X_test : pd.DataFrame,
                   draw_plot : bool,
                   title: Optional[str] = None,
                   ) -> Tuple[pd.DataFrame,
                              np.ndarray,
                              np.ndarray,
                              np.ndarray]:
    
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index, columns=X_train.columns
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),   # ¡transform, NO fit_transform!
        index=X_test.index, columns=X_test.columns
    )
    alphas = 10.0 ** np.linspace(-2, 4, 40)
    
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
    
    if draw_plot:
        for fn in chp_survival_curve:
            plt.step(fn.x, fn(fn.x), where="post") # type: ignore
        plt.title(f"Survival curve for {title}")
        plt.xlabel("Days")
        plt.ylim(0, 1)
        plt.ylabel("% of survival")
        plt.show()
        
        for fn in chp_risk_curve:
            plt.step(fn.x, fn(fn.x), where="post") # type: ignore
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
    
def p_values_log_rank(df_merged : pd.DataFrame) -> Tuple:
    """
    Fuction for taking the p-value and log rank test for different types of genes
    df_merged -> The merged df of mrna-seq and clinical data
    """
    df_life_gene = df_merged[["expression", "event", "Overall Survival (Months)"]].copy()
    
    df_life_gene["expression"] = pd.to_numeric(df_life_gene["expression"], errors="coerce")
    
    df_life_gene["Overall Survival (Months)"] = pd.to_numeric(df_life_gene["Overall Survival (Months)"], errors="coerce")
    
    df_life_gene = df_life_gene.dropna(subset=["expression", "event", "Overall Survival (Months)"])
    
    df_life_gene["time_60"] = np.minimum(df_life_gene["Overall Survival (Months)"], 60)
    
    df_life_gene["event_60"] = df_life_gene["event"].copy()
    
    df_life_gene.loc[df_life_gene["Overall Survival (Months)"] > 60, "event_60"] = False
    
    df_life_gene = df_life_gene[["expression", "time_60", "event_60"]].copy()
    
    p_value = p_values_Cox_regression(
        df_life_gene,
        event_col="event_60",
        duration_col="time_60"
    )
    
    thr = df_merged["expression"].median()
    low_group = df_merged[df_merged["expression"] < thr]
    high_group = df_merged[df_merged["expression"] >= thr]

    results = logrank_test(
            durations_A=low_group["time_60"],
            durations_B=high_group["time_60"],
            event_observed_A=low_group["event_60"],
            event_observed_B=high_group["event_60"]
        )
   
    return (p_value, results)
    
    
def evaluate_model_path(alphas, coefficients, X, Y, time_col="time_60", event_col="event_60"):
    scores = []
    for i in range(len(alphas)):
        risk = X @ coefficients[:, i]
        c_index = concordance_index_censored(Y[event_col], Y[time_col], risk)[0]
        scores.append(c_index)
    
    best_idx = np.argmax(scores)
    return best_idx, scores[best_idx]


def select_coxnet_alpha_cv(model, X_train, Y_train, n_splits=5, random_state=42,
                           time_col="time_60", event_col="event_60"):
    """Pick the Coxnet penalty alpha by k-fold CV on the training set only.

    Avoids the optimistic bias of choosing alpha on the test set: for every alpha
    on the fitted model's path we average the validation C-index across folds and
    return the alpha with the highest mean CV C-index.
    """
    alphas = model.alphas_
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    score_sum = np.zeros(len(alphas))
    score_cnt = np.zeros(len(alphas))
    for tr, va in kf.split(X_train):
        fold_model = CoxnetSurvivalAnalysis(
            l1_ratio=model.l1_ratio, alpha_min_ratio=model.alpha_min_ratio, alphas=alphas
        )
        try:
            fold_model.fit(X_train.iloc[tr], Y_train[tr])
        except Exception:
            continue
        for j, a in enumerate(alphas):
            try:
                risk = fold_model.predict(X_train.iloc[va], alpha=a)
                c = concordance_index_censored(
                    Y_train[event_col][va], Y_train[time_col][va], risk
                )[0]
                score_sum[j] += c
                score_cnt[j] += 1
            except Exception:
                continue
    mean_scores = np.divide(score_sum, score_cnt, out=np.zeros_like(score_sum),
                            where=score_cnt > 0)
    best_j = int(np.argmax(mean_scores))
    return alphas[best_j], mean_scores[best_j]


def select_rsf_params_cv(X_train, Y_train, grid, n_splits=5, random_state=42):
    """Pick RandomSurvivalForest hyper-parameters by k-fold CV on the training set.

    Selecting on the held-out test set (as the original grid search did) inflates
    the reported C-index; this scores every grid point by mean cross-validated
    C-index on the training split instead.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = -1
    best_params = None
    results = []
    for n_estimators, min_samples_split, min_samples_leaf in grid:
        fold_scores = []
        for tr, va in kf.split(X_train):
            try:
                m = RandomSurvivalForest(
                    n_estimators=n_estimators,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                )
                m.fit(X_train.iloc[tr], Y_train[tr])
                fold_scores.append(m.score(X_train.iloc[va], Y_train[va]))
            except Exception:
                continue
        if not fold_scores:
            continue
        score = float(np.mean(fold_scores))
        results.append({
            "n_estimators": n_estimators,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "c_index": score,
        })
        if score > best_score:
            best_score = score
            best_params = {
                "n_estimators": n_estimators,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
            }
    return best_params, best_score, results


def calculate_multiple_C_index(coef_series : Series, X_train_columns : pd.DataFrame, Y_train : pd.DataFrame) -> Series:
    individual_c_indexes = {}
    for gene in X_train_columns.columns:
        coef_sign = np.sign(coef_series[gene])
        individual_risk = X_train_columns[gene] * coef_sign
        
        c_val = concordance_index_censored(Y_train["event_60"], Y_train["time_60"], individual_risk)[0]
        
        individual_c_indexes[gene] = c_val

    c_index_series = pd.Series(individual_c_indexes).sort_values(ascending=False)

    print("C-index individual por cada gen:")
    return c_index_series
    
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
