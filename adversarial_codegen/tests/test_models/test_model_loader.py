import pytest

from adversarial_codegen import Models


def test_model_loading():
    # Test loading default models
    model = Models.load("codellama")
    assert model is not None
    
    # Test loading with custom parameters
    model = Models.load("codellama", temperature=0.5, top_p=None)
    assert model is not None
    
    # Test loading with invalid model name
    with pytest.raises(ValueError):
        Models.load("nonexistent_model")


def test_model_generation():
    model = Models.load("codellama")
    prompt = "def fibonacci(n):"
    
    # Test basic generation
    output = model.generate(prompt)
    assert isinstance(output, str)
    assert len(output) > 0
    
    # Test generation with different parameters
    output = model.generate(prompt, max_length=200, temperature=0.5)
    assert isinstance(output, str)
    
    # Test batch generation
    prompts = ["def sum(a, b):", "def factorial(n):"]
    outputs = model.batch_generate(prompts)
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)
