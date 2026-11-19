"""Train on consecutive monthly Santander snapshots and export the app inputs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder

from rec_sys import label_encoder_products

PRODUCTS = (
    "ind_ahor_fin_ult1 ind_aval_fin_ult1 ind_cco_fin_ult1 ind_cder_fin_ult1 "
    "ind_cno_fin_ult1 ind_ctju_fin_ult1 ind_ctma_fin_ult1 ind_ctop_fin_ult1 "
    "ind_ctpp_fin_ult1 ind_deco_fin_ult1 ind_deme_fin_ult1 ind_dela_fin_ult1 "
    "ind_ecue_fin_ult1 ind_fond_fin_ult1 ind_hip_fin_ult1 ind_plan_fin_ult1 "
    "ind_pres_fin_ult1 ind_reca_fin_ult1 ind_tjcr_fin_ult1 ind_valo_fin_ult1 "
    "ind_viv_fin_ult1 ind_nomina_ult1 ind_nom_pens_ult1 ind_recibo_ult1"
).split()
NUMERIC = ["age", "antiguedad", "renta", "ind_nuevo", "indrel", "ind_actividad_cliente"]
CATEGORIES = ["sexo", "indrel_1mes", "pais_residencia", "ind_empleado", "segmento",
              "tiprel_1mes", "indresi", "indext", "conyuemp", "canal_entrada", "indfall", "nomprov"]


def snapshot(directory, month):
    path = directory / f"train_{month.year}_{month.month:02d}_28.csv"
    data = pd.read_csv(path, usecols=["ncodpers", *NUMERIC, *CATEGORIES, *PRODUCTS],
                       dtype={col: "string" for col in CATEGORIES}, skipinitialspace=True)
    data = data.set_index("ncodpers", verify_integrity=True)
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in PRODUCTS:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not data[col].dropna().isin([0, 1]).all():
            raise ValueError(f"Indicador inválido em {path.name}: {col}")
    return data


def acquisitions(previous, current, month):
    customers = previous.index.intersection(current.index)
    before = previous.loc[customers, PRODUCTS]
    after = current.loc[customers, PRODUCTS]
    rows, cols = np.where((before.eq(0) & after.eq(1)).to_numpy())
    features = previous.loc[customers.take(rows)].assign(month=month)
    return features, np.asarray(PRODUCTS)[cols]


def train(directory, output, first_month, target_month, rounds=400):
    months = pd.period_range(first_month, target_month, freq="M")
    if len(months) < 3:
        raise ValueError("Informe um histórico com ao menos uma transição de treino.")
    previous = snapshot(directory, months[0])
    batches, labels = [], []
    for month in months[1:-1]:
        current = snapshot(directory, month)
        features, products = acquisitions(previous, current, month.month)
        batches.append(features)
        labels.extend(products)
        previous = current

    training = pd.concat(batches, ignore_index=True)
    catalog, encoder = label_encoder_products(pd.DataFrame({"added_product": labels}))
    if len(encoder.classes_) < 2:
        raise ValueError("O treino multiclasse requer aquisições de pelo menos dois produtos.")
    candidates = previous.assign(month=months[-1].month).loc[:, training.columns].copy()
    categories = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    categories.fit(training[CATEGORIES].fillna("UNKNOWN"))
    for frame in (training, candidates):
        frame[CATEGORIES] = categories.transform(frame[CATEGORIES].fillna("UNKNOWN"))

    matrix = xgb.DMatrix(training, label=catalog["encoded_products"])
    model = xgb.train(dict(objective="multi:softprob", num_class=len(encoder.classes_),
                           eta=0.05, max_depth=6, min_child_weight=2, subsample=0.9,
                           colsample_bytree=0.9, eval_metric="mlogloss", seed=1428),
                      matrix, num_boost_round=rounds)
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output / "xgb_model.json")
    pd.DataFrame({"added_product": encoder.classes_}).to_csv(output / "added_products.csv", index=False)
    candidates.to_csv(output / "processed_test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--first-month", default="2015-05")
    parser.add_argument("--target-month", default="2016-05")
    args = parser.parse_args()
    train(args.data, args.output, args.first_month, args.target_month)
