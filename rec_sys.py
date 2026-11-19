"""Load the TCC artifacts and rank products for a single customer."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

PROJECT_DIR = Path(__file__).resolve().parent


def label_encoder_products(df):
    labels = df["added_product"]
    if labels.empty or labels.isna().any():
        raise ValueError("O catálogo contém rótulos ausentes ou está vazio.")
    encoder = LabelEncoder().fit(labels)
    return df.assign(encoded_products=encoder.transform(labels)), encoder


def load_test_dataset(directory=PROJECT_DIR):
    directory = Path(directory)
    customers = pd.read_csv(directory / "processed_test.csv")
    if customers.empty:
        raise ValueError("O arquivo de clientes está vazio.")
    products, encoder = label_encoder_products(pd.read_csv(directory / "added_products.csv"))
    return customers, products, encoder


def load_model(directory=PROJECT_DIR):
    path = Path(directory) / "xgb_model.json"
    if path.exists():
        model = xgb.Booster()
        model.load_model(path)
        return model
    # Compatibility with locally generated artifacts from the original notebooks.
    with (Path(directory) / "xgb_model.pkl").open("rb") as source:
        return pickle.load(source)


def recommend(model, customer, encoder):
    if len(customer) != 1:
        raise ValueError("Selecione um cliente por previsão.")
    if model.feature_names:
        missing = set(model.feature_names).difference(customer.columns)
        if missing:
            raise ValueError("Atributos ausentes no CSV: " + ", ".join(sorted(missing)))
        features = customer[model.feature_names]
    else:
        features = customer
    scores = model.predict(xgb.DMatrix(features))
    if scores.shape != (1, len(encoder.classes_)) or not np.isfinite(scores).all():
        raise ValueError("As previsões são incompatíveis com o catálogo de treinamento.")
    ranked = encoder.inverse_transform(np.argsort(-scores[0], kind="stable"))
    portfolio = customer.iloc[0]
    return [name for name in ranked if portfolio.get(name, 0) != 1][:7]
