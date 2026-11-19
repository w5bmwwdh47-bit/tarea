"""Modelo predictivo - Momento 3 (Clase 13).
Pregunta: ¿cuántas toneladas producirá un cultivo en un municipio dentro de 3 años?
Tipo: regresión (la respuesta es un número de toneladas).
Modelo: árboles de decisión potenciados (HistGradientBoosting): muchos árboles pequeños
        en cadena, cada uno corrige el error del anterior.
Uso: python modelo.py  -> imprime resultados y guarda resultados_modelo.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from preparacion import limpiar

SEMILLA = 42
HORIZONTE = 3  # años hacia adelante
NOMBRES = {"anio": "Año", "prod_hace_3": "Cosecha hace 3 años", "prod_hace_4": "Cosecha hace 4 años",
           "prod_hace_5": "Cosecha hace 5 años", "tendencia_local": "Tendencia del municipio",
           "tendencia_depto": "Tendencia del cultivo en el Valle", "Municipio": "Municipio",
           "Cultivo": "Cultivo", "Subregion": "Subregión"}


def construir_tabla(d):
    """Una fila por municipio-cultivo-año. Solo usa información conocida 3 años antes
    del año a pronosticar, así que no hay fuga de datos."""
    t = (d.groupby(["Municipio", "Subregion", "Cultivo", "anio"], as_index=False)
           ["Produccion_toneladas"].sum())
    full = t.set_index(["Municipio", "Cultivo", "anio"])["Produccion_toneladas"]
    dep = t.groupby(["Cultivo", "anio"])["Produccion_toneladas"].sum()
    lag = lambda k: full.reindex(list(zip(t.Municipio, t.Cultivo, t.anio - k))).values
    H = HORIZONTE
    t["prod_hace_3"], t["prod_hace_4"], t["prod_hace_5"] = lag(H), lag(H + 1), lag(H + 2)
    t = t.dropna(subset=["prod_hace_3"]).copy()
    t["prod_hace_4"] = t["prod_hace_4"].fillna(t["prod_hace_3"])
    t["prod_hace_5"] = t["prod_hace_5"].fillna(t["prod_hace_4"])
    t["tendencia_local"] = t["prod_hace_3"] - t["prod_hace_5"]
    d1 = dep.reindex(list(zip(t.Cultivo, t.anio - H))).values
    d2 = dep.reindex(list(zip(t.Cultivo, t.anio - H - 2))).values
    with np.errstate(divide="ignore", invalid="ignore"):
        t["tendencia_depto"] = pd.Series(d1 / d2, index=t.index).replace([np.inf, -np.inf], np.nan).fillna(1).clip(0, 5)
    for c in ["Municipio", "Cultivo", "Subregion"]:
        t[c] = t[c].astype("category")
    return t


COLS = ["anio", "prod_hace_3", "prod_hace_4", "prod_hace_5", "tendencia_local", "tendencia_depto",
        "Municipio", "Cultivo", "Subregion"]


def nuevo_modelo(prof):
    return HistGradientBoostingRegressor(loss="absolute_error", max_depth=prof, learning_rate=0.05,
                                         max_iter=400, categorical_features="from_dtype", random_state=SEMILLA)


def entrenar(verbose=True):
    d, _ = limpiar("frutales.csv")
    t = construir_tabla(d)
    X, y = t[COLS], t["Produccion_toneladas"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEMILLA)

    curva = []
    for p in range(1, 9):
        m = nuevo_modelo(p).fit(Xtr, ytr)
        curva.append({"profundidad": p, "train": round(r2_score(ytr, m.predict(Xtr)), 3),
                      "test": round(r2_score(yte, m.predict(Xte)), 3),
                      "mae": round(mean_absolute_error(yte, m.predict(Xte)), 1)})
    validas = [c for c in curva if c["train"] - c["test"] <= 0.10]
    p = min(validas, key=lambda c: c["mae"])["profundidad"]

    modelo = nuevo_modelo(p).fit(Xtr, ytr)
    pred = modelo.predict(Xte)
    tonto = DummyRegressor(strategy="mean").fit(Xtr, ytr).predict(Xte)
    simple = Xte["prod_hace_3"]  # regla simple: se repite la cosecha de hace 3 años

    pi = permutation_importance(modelo, Xte, yte, n_repeats=5, random_state=SEMILLA,
                                scoring="neg_mean_absolute_error")
    imp = pd.Series(np.clip(pi.importances_mean, 0, None), index=COLS)
    imp = (imp / imp.sum()).rename(NOMBRES).sort_values(ascending=False)

    # Prueba en el tiempo: entrenar hasta 2019 y pronosticar 2020-2022
    tr, te = t["anio"] <= 2019, t["anio"] >= 2020
    mt = nuevo_modelo(p).fit(X[tr], y[tr])

    res = {
        "horizonte_anios": HORIZONTE, "filas": len(t), "train": len(Xtr), "test": len(Xte), "profundidad": p,
        "curva": curva,
        "r2_train": round(r2_score(ytr, modelo.predict(Xtr)), 3), "r2_test": round(r2_score(yte, pred), 3),
        "gap": round(r2_score(ytr, modelo.predict(Xtr)) - r2_score(yte, pred), 3),
        "mae_modelo": round(mean_absolute_error(yte, pred), 1),
        "mae_tonto": round(mean_absolute_error(yte, tonto), 1),
        "mae_simple": round(mean_absolute_error(yte, simple), 1),
        "mejora_vs_simple_%": round((1 - mean_absolute_error(yte, pred) / mean_absolute_error(yte, simple)) * 100, 1),
        "importancias": imp.round(3).to_dict(),
        "tiempo_mae_modelo": round(mean_absolute_error(y[te], mt.predict(X[te])), 1),
        "tiempo_mae_simple": round(mean_absolute_error(y[te], X["prod_hace_3"][te]), 1),
    }
    if verbose:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    return res, modelo, COLS


def agrupar_municipios(k=3):
    """Clustering (no supervisado): municipios parecidos según tamaño,
    crecimiento y peso de los 4 cultivos en ascenso."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    d, _ = limpiar("frutales.csv")
    P = "Produccion_toneladas"
    c4 = ["Aguacate", "Mango", "Papaya", "Lulo"]
    g = d.groupby("Municipio")
    f = pd.DataFrame({
        "subregion": g["Subregion"].first(), "total": g[P].sum(),
        "ini": d[d.anio <= 2004].groupby("Municipio")[P].sum(),
        "fin": d[d.anio >= 2018].groupby("Municipio")[P].sum(),
        "c4": d[d.Cultivo.isin(c4)].groupby("Municipio")[P].sum()}).fillna(0)
    f["crecimiento_%"] = (f.fin / f.ini.replace(0, np.nan) - 1) * 100
    f["peso_4_cultivos_%"] = f.c4 / f.total * 100
    f = f.dropna()
    Z = StandardScaler().fit_transform(np.c_[np.log1p(f.total), f["crecimiento_%"].clip(-100, 500),
                                             f["peso_4_cultivos_%"]])
    f["grupo"] = KMeans(k, n_init=20, random_state=SEMILLA).fit_predict(Z)
    # nombrar grupos por su perfil
    perfil = f.groupby("grupo").agg(crec=("crecimiento_%", "median"), peso=("peso_4_cultivos_%", "median"))
    nombres = {}
    nombres[perfil.peso.idxmax()] = "Ya apuestan a los 4 cultivos"
    resto = perfil.drop(perfil.peso.idxmax())
    nombres[resto.crec.idxmax()] = "Crecen rápido, poca diversificación"
    nombres[resto.crec.idxmin()] = "Estancados"
    f["perfil"] = f.grupo.map(nombres)
    return f[["subregion", "total", "crecimiento_%", "peso_4_cultivos_%", "perfil"]].round(1)


if __name__ == "__main__":
    r, _, _ = entrenar()
    json.dump(r, open("resultados_modelo.json", "w"), indent=2, ensure_ascii=False)
    g = agrupar_municipios()
    print(g.groupby("perfil").agg(municipios=("total", "size"), crecimiento=("crecimiento_%", "median"),
                                  peso_4=("peso_4_cultivos_%", "median")))
    for p, s in g.groupby("perfil"):
        print(p, ":", ", ".join(s.index))


def pronosticar(anio_objetivo=2025, cultivos=("Aguacate", "Mango", "Papaya", "Lulo")):
    """Pronóstico a 3 años con datos hasta 2022 (modelo entrenado con toda la historia)."""
    d, _ = limpiar("frutales.csv")
    t = construir_tabla(d)
    r, _, _ = entrenar(verbose=False)
    m = nuevo_modelo(r["profundidad"]).fit(t[COLS], t["Produccion_toneladas"])
    # filas futuras: municipio-cultivo con dato en el año base
    base = anio_objetivo - HORIZONTE
    fut = d[(d.anio == base) & (d.Cultivo.isin(cultivos))][["Municipio", "Subregion", "Cultivo"]].drop_duplicates().copy()
    fut["anio"] = anio_objetivo
    fut["Produccion_toneladas"] = np.nan
    ext = pd.concat([d.groupby(["Municipio", "Subregion", "Cultivo", "anio"], as_index=False)["Produccion_toneladas"].sum(), fut])
    tt = construir_tabla(ext.assign(Produccion_toneladas=ext.Produccion_toneladas.fillna(0)))
    tt = tt[tt.anio == anio_objetivo].copy()
    for c in ["Municipio", "Cultivo", "Subregion"]:
        tt[c] = pd.Categorical(tt[c], categories=t[c].cat.categories)
    tt["pronostico"] = m.predict(tt[COLS]).clip(0)
    return tt[["Municipio", "Subregion", "Cultivo", "prod_hace_3", "pronostico"]]


if __name__ == "__main__" and True:
    p = pronosticar()
    print(p.groupby("Cultivo")[["prod_hace_3", "pronostico"]].sum().round(0))
    s = p[p.Subregion == "Sur"]
    print("Sur:", s[["prod_hace_3", "pronostico"]].sum().round(0).to_dict())
    print("Valle:", p[["prod_hace_3", "pronostico"]].sum().round(0).to_dict())
