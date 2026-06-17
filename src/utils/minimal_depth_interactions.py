"""
Minimal Depth + Maximal Subtree para detectar pares de genes que interaccionan
sobre un Random Survival Forest entrenado con scikit-survival.

Replica la lógica de randomForestSRC::find.interaction(method="maxsubtree")
en Python puro, recorriendo los arboles del bosque que ya tienes entrenado.

Base metodologica:
  Ishwaran, Kogalur, Gorodeski, Minn, Lauer (2010), JASA 105:205-217.
  Ishwaran, Kogalur, Chen, Minn (2011), Stat. Anal. Data Min. 4:115-132.

Idea:
  - minimal depth de una variable v = profundidad del nodo mas alto que divide
    por v (raiz = 0). Menor profundidad -> variable mas predictiva.
  - maximal subtree de i = el subarbol mas alto cuya raiz divide por i.
  - interaccion [i][j] = profundidad minima de j DENTRO del maximal subtree de i.
    Valores PEQUEÑOS de [i][j] -> interaccion fuerte entre i y j.

NOTA: la normalizacion exacta de randomForestSRC es algo mas elaborada. Esta
implementacion es fiel al concepto y suficiente para rankear pares; si necesitas
rigor numerico total, valida contra randomForestSRC en un ejemplo pequeño.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Minimal depth univariante
# ---------------------------------------------------------------------------

def _tree_minimal_depths(tree, n_features):
    """Profundidad minima de cada feature en un arbol (raiz = 0).
    Devuelve (array_minimal_depths, profundidad_maxima_del_arbol).
    Las features ausentes quedan como np.inf."""
    md = np.full(n_features, np.inf)
    max_depth = 0
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        f = tree.feature[node]
        if f >= 0:  # nodo de division (las hojas tienen feature = -2)
            if depth < md[f]:
                md[f] = depth
            stack.append((tree.children_left[node], depth + 1))
            stack.append((tree.children_right[node], depth + 1))
    return md, max_depth


def minimal_depth(forest, feature_names):
    """Minimal depth promedio por feature sobre todo el bosque.
    Las features ausentes en un arbol reciben (profundidad_maxima + 1) en ese
    arbol, como penalizacion (en linea con el tratamiento de Ishwaran).

    Devuelve una pd.Series ordenada de menor a mayor (menor = mas predictiva).
    """
    n_features = len(feature_names)
    acc = np.zeros(n_features)
    for est in forest.estimators_:
        md, max_depth = _tree_minimal_depths(est.tree_, n_features)
        penalty = max_depth + 1
        md = np.where(np.isinf(md), penalty, md)
        acc += md
    acc /= len(forest.estimators_)
    return pd.Series(acc, index=feature_names).sort_values()


# ---------------------------------------------------------------------------
# 2. Maximal subtree e interacciones por pares
# ---------------------------------------------------------------------------

def _shallowest_node_for(tree, i):
    """Nodo raiz del maximal subtree mas cercano de la feature i.
    Devuelve el id de nodo, o None si i no aparece en el arbol."""
    best_node, best_depth = None, np.inf
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        f = tree.feature[node]
        if f >= 0:
            if f == i and depth < best_depth:
                best_depth, best_node = depth, node
            stack.append((tree.children_left[node], depth + 1))
            stack.append((tree.children_right[node], depth + 1))
    return best_node


def _min_depth_of_j_in_subtree(tree, root, j):
    """Profundidad minima de j dentro del subarbol con raiz `root` (root = prof 0).
    Devuelve (min_depth_de_j, profundidad_maxima_del_subarbol)."""
    best_j = np.inf
    max_depth = 0
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        f = tree.feature[node]
        if f >= 0:
            if f == j and depth > 0 and depth < best_j:
                best_j = depth
            stack.append((tree.children_left[node], depth + 1))
            stack.append((tree.children_right[node], depth + 1))
    return best_j, max_depth


def interaction_matrix(forest, feature_names, subset=None):
    """Matriz de interacciones [i][j] por minimal depth + maximal subtree.

    [i][j] = profundidad media (sobre arboles) de j dentro del maximal subtree
    de i. Valores PEQUEÑOS indican interaccion fuerte. Se promedia solo sobre
    los arboles donde i aparece; si j no esta en el subarbol de i, se penaliza
    con (profundidad_maxima_del_subarbol + 1).

    `subset`: lista de genes a evaluar. RECOMENDADO pasar el top-k por minimal
    depth: la matriz completa p x p es O(p^2) por arbol y con miles de genes
    es inviable.
    """
    if subset is None:
        subset = list(feature_names)
    name_to_idx = {n: k for k, n in enumerate(feature_names)}
    idx = [name_to_idx[n] for n in subset]
    p = len(idx)

    sums = np.zeros((p, p))
    counts = np.zeros((p, p))

    for est in forest.estimators_:
        tree = est.tree_
        # cache del maximal subtree de cada i en este arbol
        roots = {a: _shallowest_node_for(tree, i) for a, i in enumerate(idx)}
        for a, i in enumerate(idx):
            root_node = roots[a]
            if root_node is None:
                continue
            for b, j in enumerate(idx):
                if i == j:
                    continue
                dj, sub_max = _min_depth_of_j_in_subtree(tree, root_node, j)
                if np.isinf(dj):
                    dj = sub_max + 1
                sums[a, b] += dj
                counts[a, b] += 1

    with np.errstate(invalid="ignore"):
        mat = np.where(counts > 0, sums / counts, np.nan)
    return pd.DataFrame(mat, index=subset, columns=subset)


def top_interacting_pairs(inter_df, n=20):
    """Extrae los pares (gen_i, gen_j) con menor valor -> interaccion mas fuerte.
    Simetriza promediando [i][j] y [j][i]."""
    pairs = []
    for a, gi in enumerate(inter_df.index):
        for b, gj in enumerate(inter_df.columns):
            if a < b:  # par no ordenado: evita duplicados y diagonal
                score = np.nanmean([inter_df.iloc[a, b], inter_df.iloc[b, a]])
                pairs.append((gi, gj, score))
    return (pd.DataFrame(pairs, columns=["gen_i", "gen_j", "interaccion"])
              .dropna()
              .sort_values("interaccion")
              .head(n)
              .reset_index(drop=True))


# ---------------------------------------------------------------------------
# 3. Uso (con tu modelo ya entrenado: random_forest_model, X_train)
# ---------------------------------------------------------------------------
