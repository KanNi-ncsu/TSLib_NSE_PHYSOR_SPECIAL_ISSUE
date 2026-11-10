"""
Unit and Integration Tests for Standalone ARIMA Baseline

Tests verify correctness before running on real nuclear accident data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from arima_standalone import ARIMAForecaster


class TestARIMABasic:
    """Basic functionality tests for ARIMA forecaster"""

    def test_1_synthetic_linear_trend(self):
        """
        Test 1: Synthetic Linear Trend

        Data: y = 2*t + 10 (simple linear trend from t=0 to t=49)
        ARIMA(1,1,1) should predict continuation of trend

        Expected: Predictions follow linear trend approximately
        """
        # Generate synthetic linear trend
        t = np.arange(50)
        series = 2 * t + 10  # y = 2t + 10, values from 10 to 108

        # Use first 30 points as history, predict next 20
        history = series[:30]  # 10 to 68
        true_future = series[30:50]  # 70 to 108

        # Create ARIMA forecaster
        forecaster = ARIMAForecaster(order=(1, 1, 1))

        # Forecast 20 steps ahead
        predictions = forecaster.fit_and_forecast(history, steps=20)

        # Verify predictions exist and have correct shape
        assert predictions is not None, "Predictions should not be None"
        assert len(predictions) == 20, f"Expected 20 predictions, got {len(predictions)}"
        assert np.all(np.isfinite(predictions)), "All predictions should be finite"

        # Verify predictions follow trend (should be increasing)
        assert np.all(np.diff(predictions) > 0), "Linear trend predictions should be increasing"

        # Verify predictions are in reasonable range
        # True values: 70-108, ARIMA might not be perfect but should be close
        mae = np.mean(np.abs(predictions - true_future))
        print(f"Test 1 - Linear Trend MAE: {mae:.2f}")
        assert mae < 20, f"MAE should be < 20 for linear trend, got {mae:.2f}"

        print("Test 1 PASS: Synthetic linear trend")

    def test_2_synthetic_stationary(self):
        """
        Test 2: Synthetic Stationary Series

        Data: AR(1) process with phi=0.7
        ARIMA should handle stationary series well

        Expected: Predictions converge to mean
        """
        # Generate AR(1) process: y_t = 0.7 * y_{t-1} + noise
        np.random.seed(42)
        n = 100
        series = np.zeros(n)
        series[0] = 10.0

        for t in range(1, n):
            series[t] = 0.7 * series[t-1] + np.random.normal(0, 0.5)

        # Use first 70 for history, predict next 30
        history = series[:70]

        # Create ARIMA forecaster
        forecaster = ARIMAForecaster(order=(1, 1, 1))

        # Forecast 30 steps ahead
        predictions = forecaster.fit_and_forecast(history, steps=30)

        # Verify predictions exist and have correct shape
        assert predictions is not None
        assert len(predictions) == 30
        assert np.all(np.isfinite(predictions))

        # For AR(1) with phi < 1, predictions should be bounded
        assert np.max(np.abs(predictions)) < 50, "Stationary predictions should be bounded"

        print("Test 2 PASS: Synthetic stationary series")

    def test_3_small_real_data_shape(self):
        """
        Test 3: Small Data Test

        Verify output shapes and no NaN/Inf values
        """
        # Simulate 5 test samples with seq_len=30, pred_len=30
        np.random.seed(123)
        n_samples = 5
        seq_len = 30
        pred_len = 30

        # Generate random walk data (realistic for nuclear parameters)
        test_data = []
        for i in range(n_samples):
            start = 500 + i * 10
            walk = np.cumsum(np.random.normal(0, 5, seq_len + pred_len)) + start
            test_data.append(walk)

        test_data = np.array(test_data)  # [5, 60]

        # Create forecaster
        forecaster = ARIMAForecaster(order=(1, 1, 1))

        # Predict for each sample
        predictions = []
        for i in range(n_samples):
            history = test_data[i, :seq_len]  # First 30 points
            pred = forecaster.fit_and_forecast(history, steps=pred_len)
            predictions.append(pred)

        predictions = np.array(predictions)  # [5, 30]

        # Verify shape
        assert predictions.shape == (n_samples, pred_len), \
            f"Expected shape ({n_samples}, {pred_len}), got {predictions.shape}"

        # Verify no NaN or Inf
        assert np.all(np.isfinite(predictions)), "All predictions should be finite"

        # Calculate MAE for each sample
        for i in range(n_samples):
            true_future = test_data[i, seq_len:]
            mae = np.mean(np.abs(predictions[i] - true_future))
            print(f"  Sample {i+1} MAE: {mae:.2f}")

        print("Test 3 PASS: Small data shape verification")

    def test_4_fallback_mechanism(self):
        """
        Test 4: Fallback Mechanism

        When ARIMA fails (too short series, all zeros, etc.),
        should fallback to naive persistence (repeat last value)
        """
        # Test with very short series (should trigger fallback)
        short_series = np.array([10.0, 11.0, 12.0])  # Only 3 points

        forecaster = ARIMAForecaster(order=(1, 1, 1))
        predictions = forecaster.fit_and_forecast(short_series, steps=5)

        # Should fallback to naive (repeat last value)
        expected = np.full(5, 12.0)  # Last value repeated

        assert np.allclose(predictions, expected), \
            f"Fallback should repeat last value. Expected {expected}, got {predictions}"

        print("Test 4 PASS: Fallback mechanism works")

    def test_5_batch_predict(self):
        """
        Test 5: Batch Prediction

        Verify batch processing works correctly
        """
        # Create 3 samples
        np.random.seed(456)
        samples = []
        for i in range(3):
            t = np.arange(30)
            series = 2 * t + 100 + i * 50  # Different offsets
            samples.append(series)

        samples = np.array(samples)  # [3, 30]

        # Create forecaster
        forecaster = ARIMAForecaster(order=(1, 1, 1))

        # Batch predict
        predictions = forecaster.batch_predict(samples, steps=10)

        # Verify shape
        assert predictions.shape == (3, 10), \
            f"Expected shape (3, 10), got {predictions.shape}"

        # Verify all finite
        assert np.all(np.isfinite(predictions))

        # Verify each sample has increasing predictions (linear trend)
        for i in range(3):
            assert np.all(np.diff(predictions[i]) > 0), \
                f"Sample {i} should have increasing predictions"

        print("Test 5 PASS: Batch prediction works")


class TestARIMAIntegration:
    """Integration tests with real data structure"""

    def test_integration_realistic_scenario(self):
        """
        Integration Test: Realistic Nuclear Accident Scenario

        Simulates LOCA test data structure:
        - 15 test samples
        - seq_len = 30 (300 seconds history)
        - pred_len = 30 (300 seconds prediction)
        - Values in range 500-1500K (realistic TPCT range)
        """
        np.random.seed(789)

        n_samples = 15
        seq_len = 30
        pred_len = 30

        # Generate realistic LOCA temperature profiles
        # LOCA: rapid temperature rise, then plateau or slow decay
        test_samples = []
        ground_truth = []

        for i in range(n_samples):
            # Initial rise: 600K -> 1000K over 30 steps
            t = np.arange(seq_len + pred_len)

            # Sigmoid-like rise with noise
            base = 600 + 400 / (1 + np.exp(-(t - 15) / 3))
            noise = np.random.normal(0, 10, len(t))
            series = base + noise

            test_samples.append(series[:seq_len])
            ground_truth.append(series[seq_len:])

        test_samples = np.array(test_samples)  # [15, 30]
        ground_truth = np.array(ground_truth)  # [15, 30]

        # Create forecaster
        forecaster = ARIMAForecaster(order=(1, 1, 1))

        # Batch predict
        predictions = forecaster.batch_predict(test_samples, steps=pred_len)

        # Verify shape
        assert predictions.shape == (n_samples, pred_len)

        # Calculate overall MAE
        mae = np.mean(np.abs(predictions - ground_truth))

        print(f"\nIntegration Test Results:")
        print(f"  Samples: {n_samples}")
        print(f"  Prediction horizon: {pred_len}")
        print(f"  Overall MAE: {mae:.2f}")

        # MAE should be reasonable (not catastrophically high)
        assert mae < 100, f"Integration MAE should be < 100, got {mae:.2f}"

        # Verify no NaN/Inf
        assert np.all(np.isfinite(predictions))

        print("Integration Test PASS: Realistic scenario")


if __name__ == "__main__":
    print("="*80)
    print("ARIMA Standalone - Unit and Integration Tests")
    print("="*80)
    print()

    # Run basic tests
    print("Running Basic Tests...")
    print("-" * 80)

    basic_tests = TestARIMABasic()

    try:
        basic_tests.test_1_synthetic_linear_trend()
    except Exception as e:
        print(f"Test 1 FAILED: {e}")

    try:
        basic_tests.test_2_synthetic_stationary()
    except Exception as e:
        print(f"Test 2 FAILED: {e}")

    try:
        basic_tests.test_3_small_real_data_shape()
    except Exception as e:
        print(f"Test 3 FAILED: {e}")

    try:
        basic_tests.test_4_fallback_mechanism()
    except Exception as e:
        print(f"Test 4 FAILED: {e}")

    try:
        basic_tests.test_5_batch_predict()
    except Exception as e:
        print(f"Test 5 FAILED: {e}")

    # Run integration test
    print()
    print("Running Integration Test...")
    print("-" * 80)

    integration_tests = TestARIMAIntegration()

    try:
        integration_tests.test_integration_realistic_scenario()
    except Exception as e:
        print(f"Integration Test FAILED: {e}")

    print()
    print("="*80)
    print("All Tests Completed")
    print("="*80)
