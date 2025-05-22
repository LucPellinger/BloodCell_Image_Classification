import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.base_model import BaseModel

if __name__ == "__main__":
    print("📚 Listing available pretrained models from keras.applications:\n")
    BaseModel.list_available_pretrained_models()
