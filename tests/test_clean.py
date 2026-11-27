import pandas as pd

from autofare.clean import clean


def test_clean_fixes_problems():
    df = pd.DataFrame({
        "distance_km": [5000.0, 7.0],
        "wait_min": [9999.0, 10.0],
        "booking": [" Stand", "app."],
        "zone": ["Station Area", "Market"],
        "fare_paid_receipt_rs": [100, 200],
        "fare_rs": [101, 199],
    })
    out = clean(df)
    assert out.loc[0, "distance_km"] == 5.0
    assert (out["wait_min"] != 9999).all()
    assert set(out["booking"]) == {"stand", "app"}
    assert "fare_paid_receipt_rs" not in out.columns
