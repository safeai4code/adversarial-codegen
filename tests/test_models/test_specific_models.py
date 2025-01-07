import pytest

from adversarial_codegen.models import Models
from adversarial_codegen.models.model_implementations import (
    CodeLLaMAModel,
    StarCoderModel,
)


@pytest.fixture
def sample_prompts():
    return [
        "def fibonacci(n):",
        "def quicksort(arr):",
        "def binary_search(arr, target):"
    ]


class TestCodeLLaMA:
    @pytest.mark.slow  # Mark as slow test due to model loading
    def test_model_initialization(self):
        model = CodeLLaMAModel()
        assert model.model is not None
        assert model.tokenizer is not None

    @pytest.mark.slow
    def test_generation(self):
        model = CodeLLaMAModel()
        prompt = "def add(a, b):"
        
        # Test basic generation
        output = model.generate(prompt)
        assert isinstance(output, str)
        assert len(output) > len(prompt)
        assert "def" in output  # Should contain function definition
        
        # Test generation with different parameters
        output_cold = model.generate(prompt, temperature=0.1)
        output_hot = model.generate(prompt, temperature=0.9)
        assert output_cold != output_hot  # Should be different due to temperature
        
        # Test with different max_lengths
        short = model.generate(prompt, max_length=50)
        long = model.generate(prompt, max_length=200)
        assert len(short) <= len(long)

    @pytest.mark.slow
    def test_batch_generation(self, sample_prompts):
        model = CodeLLaMAModel()
        outputs = model.batch_generate(sample_prompts)
        
        assert len(outputs) == len(sample_prompts)
        assert all(isinstance(o, str) for o in outputs)
        assert all(len(o) > 0 for o in outputs)

    def test_error_handling(self):
        model = CodeLLaMAModel()
        
        # Test with invalid prompt type
        with pytest.raises(TypeError):
            model.generate(123)
        
        # Test with empty prompt
        with pytest.raises(ValueError):
            model.generate("")
        
        # Test with None prompt
        with pytest.raises(TypeError):
            model.generate(None)


class TestStarCoder:
    @pytest.mark.slow
    def test_model_initialization(self):
        model = StarCoderModel()
        assert model.model is not None
        assert model.tokenizer is not None

    @pytest.mark.slow
    def test_generation(self):
        model = StarCoderModel()
        prompt = "def calculate_average(numbers):"
        
        output = model.generate(prompt)
        assert isinstance(output, str)
        assert "def" in output
        
        # Test parameter influence
        output1 = model.generate(prompt, 
                               temperature=0.7, 
                               top_p=0.95,
                               max_length=100)
        output2 = model.generate(prompt, 
                               temperature=0.7, 
                               top_p=None,
                               max_length=100)
        assert isinstance(output1, str)
        assert isinstance(output2, str)


# Add similar test classes for other models (CodeGen, DeepSeek, etc.)

def test_model_registry():
    # Test that all models are properly registered
    models = ["codellama", "starcoder", "codegen", 
             "deepseek", "incoder", "magicoder"]
    
    for model_name in models:
        model = Models.load(model_name)
        assert model is not None
        
    # Test invalid model name
    with pytest.raises(ValueError):
        Models.load("nonexistent_model")


@pytest.mark.slow
def test_model_consistency():
    """Test that models produce consistent output with same parameters"""
    models = ["codellama", "starcoder"]
    prompt = "def fibonacci(n):"
    
    for model_name in models:
        model = Models.load(model_name)
        
        # Test with fixed parameters and seed
        output1 = model.generate(prompt, temperature=0.0)
        output2 = model.generate(prompt, temperature=0.0)
        assert output1 == output2  # Should be deterministic with temp=0
