"""Recipe 4: Custom Model Integration - End-to-end verification."""

import pandas as pd
import numpy as np
from tsagentkit import TaskSpec, rolling_backtest, run_forecast
from tsagentkit.contracts import ModelArtifact
from tsagentkit.series import TSDataset


class NaiveModel:
    """Simple naive forecast model."""

    def __init__(self, season_length: int = 1):
        self.season_length = season_length
        self.last_values = {}

    def fit(self, df: pd.DataFrame) -> "NaiveModel":
        """Fit by storing last value per series."""
        for uid in df["unique_id"].unique():
            series = df[df["unique_id"] == uid].sort_values("ds")
            self.last_values[uid] = series["y"].iloc[-self.season_length :].values
        return self

    def predict(
        self,
        unique_ids: list[str],
        horizon: int,
        last_dates: dict[str, pd.Timestamp],
        freq: str = "D",
    ) -> pd.DataFrame:
        """Generate naive forecast."""
        predictions = []
        for uid in unique_ids:
            if uid not in self.last_values:
                continue
            values = self.last_values[uid]
            last_date = last_dates[uid]
            for h in range(1, horizon + 1):
                idx = (h - 1) % len(values)
                predictions.append(
                    {
                        "unique_id": uid,
                        "ds": last_date + pd.Timedelta(days=h)
                        if freq == "D"
                        else last_date + pd.DateOffset(**{freq: h}),
                        "yhat": values[idx],
                    }
                )
        return pd.DataFrame(predictions)


def custom_fit(model_name: str, dataset: TSDataset, config: dict) -> ModelArtifact:
    """Fit custom model."""
    season_length = config.get("season_length", 1)
    model = NaiveModel(season_length=season_length)
    model.fit(dataset.df)

    return ModelArtifact(
        model=model,
        model_name=model_name,
        config=config,
    )


def custom_predict(artifact: ModelArtifact, dataset: TSDataset, horizon: int) -> pd.DataFrame:
    """Generate predictions."""
    naive_model = artifact.model

    # Get unique ids and their last dates
    df = dataset.df
    unique_ids = df["unique_id"].unique().tolist()
    last_dates = df.groupby("unique_id")["ds"].max().to_dict()

    return naive_model.predict(unique_ids, horizon, last_dates, freq="D")


def test_recipe4_custom_model():
    """Test Recipe 4: Custom Model Integration."""
    print("=" * 60)
    print("RECIPE 4: Custom Model Integration")
    print("=" * 60)

    # Generate data
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * 30 + ["B"] * 30,
            "ds": list(pd.date_range("2024-01-01", periods=30, freq="D")) * 2,
            "y": list(range(30)) * 2,
        }
    )
    print(f"Data shape: {df.shape}")
    print(f"Series: {df['unique_id'].unique().tolist()}")

    # Test custom model directly
    print("\n=== Testing Custom Model Directly ===")
    spec = TaskSpec(horizon=7, freq="D")
    dataset = TSDataset.from_dataframe(df, spec)

    artifact = custom_fit("CustomNaive", dataset, {"season_length": 7})
    print(f"Fitted model: {artifact.model_name}")

    predictions = custom_predict(artifact, dataset, 7)
    print(f"Predictions shape: {predictions.shape}")
    print(predictions.head(10))

    # Run with custom model through run_forecast
    print("\n=== Running via run_forecast ===")
    result = run_forecast(
        df,
        spec,
        mode="quick",  # Use quick mode to skip backtest (simpler for custom model)
        fit_func=custom_fit,
        predict_func=custom_predict,
    )

    print("\n=== Custom Model Results ===")
    print(result.summary())

    print("\n=== Forecast ===")
    print(result.forecast.df)

    # Assertions
    assert result is not None, "Result should not be None"
    assert result.forecast is not None, "Forecast should not be None"
    assert len(result.forecast.df) > 0, "Forecast should have rows"

    # Verify our custom predictions are in the forecast
    expected_rows = 7 * 2  # 7 days * 2 series
    assert len(result.forecast.df) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(result.forecast.df)}"
    )

    print("\n✅ Recipe 4 PASSED")
    return True


if __name__ == "__main__":
    test_recipe4_custom_model()
