import random

import streamlit as st
import xgboost as xgb

from rec_sys import PROJECT_DIR, load_model, load_test_dataset, recommend


@st.cache_resource
def load_artifacts():
    customers, _, encoder = load_test_dataset()
    return customers, encoder, load_model()


def main():
    st.set_page_config(page_title="Recomendação Santander", page_icon="🏦")
    st.image(str(PROJECT_DIR / "santander-banner.png"))
    st.title("Recomendação de produtos Santander")
    st.write("O XGBoost usa o perfil do cliente e sua carteira anterior para ordenar até sete produtos.")
    st.image(str(PROJECT_DIR / "diagrama_xgboost.png"), caption="Fluxo do modelo XGBoost")

    try:
        customers, encoder, model = load_artifacts()
    except FileNotFoundError as error:
        st.info(f"Para habilitar a previsão, adicione {error.filename} conforme o README.")
        return
    except (ValueError, KeyError, xgb.core.XGBoostError) as error:
        st.error(f"Falha ao carregar os artefatos: {error}")
        return

    if "customer_row" not in st.session_state:
        st.session_state.customer_row = random.randrange(len(customers))
    if st.button("Novo usuário") and len(customers) > 1:
        offset = random.randrange(1, len(customers))
        st.session_state.customer_row = (st.session_state.customer_row + offset) % len(customers)
    customer = customers.iloc[[st.session_state.customer_row]]
    st.dataframe(customer.T)

    if st.button("Realizar previsão"):
        try:
            products = recommend(model, customer, encoder)
        except (ValueError, xgb.core.XGBoostError) as error:
            st.error(f"Falha na previsão: {error}")
            return
        if products:
            st.subheader("Produtos recomendados")
            for position, product in enumerate(products, start=1):
                st.text(f"{position}. {product}")
        else:
            st.info("O cliente já possui todos os produtos do catálogo.")


if __name__ == "__main__":
    main()
