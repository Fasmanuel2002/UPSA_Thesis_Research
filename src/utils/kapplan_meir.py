import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
from sksurv.compare import compare_survival
from numpy import ndarray
from typing import Tuple

def k_m_cox(list_risk : list, durations : Tensor, events : Tensor):
    df = pd.DataFrame()

    thr = np.median(list_risk)
    df["group"] = np.where(list_risk > thr, "High", "Low")
    df["duration"] = durations # type: ignore
    df["event"] = events # type: ignore

    for g in ["High", "Low"]:
            sub = df[df["group"] == g]
            event = sub["event"].to_numpy(dtype=bool)
            time = sub["duration"].to_numpy(dtype=float)
                
            time_points_months = np.array([0, 12, 24, 36, 48, 60])
            at_risk = [(time > tp).sum() for tp in time_points_months]
            print(f"\nFor this group the risk {g}\n")
            x = [(time, risk) for time,risk in zip(time_points_months, at_risk)]
            for i, _ in enumerate(x):
                print(f"For the time {x[i][0]} the risk group is {x[i][1]}")
                    
            event_60_month = event.copy()
            event_60_month[time > 60] = False
            time_60_months = np.minimum(time, 60)
            time, prob_survival, conf_int = kaplan_meier_estimator(event_60_month, time_60_months, conf_type="log-log")  # type: ignore # event True=evento, False=censura
            plt.step(time, prob_survival, where="post", label=f"{g} (n={len(sub)})")
            plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.xlabel("Months")
    plt.ylabel("Survival probability")
    plt.title(f"Kaplan–Meier por (mediana)")
    plt.legend()
    plt.show()
    
    time_all = df["duration"].to_numpy(dtype=float)
    event_all = df["event"].to_numpy(dtype=bool)
    event_60_months = event_all.copy()
    event_60_months[time_all > 60] = False
    time_60_months = np.minimum(time_all, 60)

    y = Surv.from_arrays(event_60_months, time_60_months)
    chisq, pvalue = compare_survival(y, df["group"].to_numpy()) # type: ignore
    print(f"Log-rank: chi2={chisq:.3f}, p={pvalue:.4g}")
    return chisq, pvalue


def k_m_surv(risk : ndarray, Y_test) -> Tuple:
    df = pd.DataFrame()
    thr = np.median(risk)

    df["group"] = np.where(risk > thr, "High", "Low")
    df["duration"] = Y_test["time_60"]
    df["event"] = Y_test["event_60"].astype(bool)
    
    for g in ["High", "Low"]:
        sub = df[df["group"] == g]
        event = sub["event"].to_numpy(dtype=bool)
        time = sub["duration"].to_numpy(dtype=float)
            
        time_points_months = np.array([0, 12, 24, 36, 48, 60])
        at_risk = [(time > tp).sum() for tp in time_points_months]
        print(f"\nFor this group the risk {g}\n")
        x = [(time, risk) for time,risk in zip(time_points_months, at_risk)]
        for i, _ in enumerate(x):
            print(f"For the time {x[i][0]} the risk group is {x[i][1]}")
                
        event_60_month = event.copy()
        event_60_month[time > 60] = False
        time_60_months = np.minimum(time, 60)
        time, prob_survival, conf_int = kaplan_meier_estimator(event_60_month, time_60_months, conf_type="log-log")  # type: ignore # event True=evento, False=censura
        plt.step(time, prob_survival, where="post", label=f"{g} (n={len(sub)})")
        plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.xlabel("Months")
    plt.ylabel("Survival probability")
    plt.title(f"Kaplan–Meier por (mediana)")
    plt.legend()
    plt.show()
    
    
    time_all = df["duration"].to_numpy(dtype=float)
    event_all = df["event"].to_numpy(dtype=bool)
    event_60_months = event_all.copy()
    event_60_months[time_all > 60] = False
    time_60_months = np.minimum(time_all, 60)

    y = Surv.from_arrays(event_60_months, time_60_months)
    chisq, pvalue = compare_survival(y, df["group"].to_numpy()) # type: ignore
    print(f"Log-rank: chi2={chisq:.3f}, p={pvalue:.4g}")

    return chisq, pvalue
    
