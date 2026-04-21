import os

def test_models_exist():
    files = [f for f in os.listdir() if f.endswith(".pkl")]
    assert len(files) > 0, "No model files found!"

    print(" Test Passed: Models are generated")

if __name__ == "__main__":
    test_models_exist()