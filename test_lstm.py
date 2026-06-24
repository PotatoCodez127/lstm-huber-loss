# test_lstm.py
import pytest
import torch

from data_engine_lstm import generate_macro_sequence
from model_lstm import MacroForecaster


@pytest.fixture
def initialized_model():
    """Provides a deterministic instance of the forecasting architecture."""
    torch.manual_seed(42)
    model = MacroForecaster(input_size=1, hidden_size=16, num_layers=1)
    model.eval()
    return model


def test_data_engine_shapes():
    """Validates that the data generator outputs compliant 3D and 2D Tensors."""
    samples = 10
    seq_length = 50

    X, y = generate_macro_sequence(samples=samples, seq_length=seq_length, seed=42)

    # Assert tensor structural dimensions match network expectations
    assert X.shape == (samples, seq_length, 1), f"Expected 3D input tensor shape, got {X.shape}"
    assert y.shape == (samples, 1), f"Expected 2D target tensor shape, got {y.shape}"
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_model_inference_flow(initialized_model):
    """Ensures the forward pass correctly condenses temporal sequences into a continuous scalar scalar."""
    samples = 4
    seq_length = 50

    # Generate mock evaluation batch
    X, _ = generate_macro_sequence(samples=samples, seq_length=seq_length, seed=100)
    x_scaled = X / X.max()

    with torch.no_grad():
        predictions = initialized_model(x_scaled)

    # Verify mapping boundary integrity
    assert predictions.shape == (
        samples,
        1,
    ), f"Inference output dimension mismatch: {predictions.shape}"
    assert not torch.isnan(predictions).any(), "Model forward pass generated NaN values."


def test_reproducibility_seeding():
    """Verifies that the pseudo-random sequence path generator respects seed control flags."""
    x1, y1 = generate_macro_sequence(5, seq_length=50, seed=42)
    x2, y2 = generate_macro_sequence(5, seq_length=50, seed=42)
    x3, y3 = generate_macro_sequence(5, seq_length=50, seed=99)

    # Matching seeds must produce identical tensor space layouts
    assert torch.equal(x1, x2), "Identical seeds produced divergent input sequences."
    assert torch.equal(y1, y2), "Identical seeds produced divergent target values."

    # Deviating seeds must generate isolated path movements
    assert not torch.equal(x1, x3), "Divergent seeds produced identical input matrices."
