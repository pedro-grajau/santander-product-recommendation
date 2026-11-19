import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from rec_sys import PROJECT_DIR, load_model, load_test_dataset, recommend
from train import CATEGORIES, NUMERIC, PRODUCTS, train


class RecommendationWorkflowTest(unittest.TestCase):
    def test_train_select_and_predict(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rows = []
            for customer in range(8):
                row = dict.fromkeys(PRODUCTS, 0)
                row.update(dict.fromkeys(NUMERIC, customer + 20))
                row.update(dict.fromkeys(CATEGORIES, "A"))
                row["ncodpers"] = customer
                rows.append(row)
            april = pd.DataFrame(rows)
            april.to_csv(directory / "train_2016_04_28.csv", index=False)
            may = april.copy()
            for i in range(len(may)):
                may.loc[i, PRODUCTS[i % 2]] = 1
            may.to_csv(directory / "train_2016_05_28.csv", index=False)

            train(directory, directory, "2016-04", "2016-06", rounds=2)
            customers, catalog, encoder = load_test_dataset(directory)
            model = load_model(directory)
            self.assertEqual(recommend(model, customers.iloc[[0]], encoder), [PRODUCTS[1]])

            with patch("rec_sys.load_test_dataset", return_value=(customers, catalog, encoder)), \
                 patch("rec_sys.load_model", return_value=model):
                app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run()
                first = app.session_state["customer_row"]
                app.button[0].click().run()
                selected = app.session_state["customer_row"]
                self.assertNotEqual(selected, first)
                app.button[1].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["customer_row"], selected)
                self.assertEqual(app.text[0].value, "1. " + PRODUCTS[1 - selected % 2])


if __name__ == "__main__":
    unittest.main()
