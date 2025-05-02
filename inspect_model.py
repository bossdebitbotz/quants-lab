import torch
import collections

# Load the model data
model_path = 'models/tft_10sec_bias_corrected_20250501_230054.pt'
loaded_data = torch.load(model_path)

print(f"Type of loaded data: {type(loaded_data)}")
print(f"Keys in loaded data: {list(loaded_data.keys())}")

if 'model' in loaded_data:
    model_data = loaded_data['model']
    print(f"Type of model data: {type(model_data)}")
    
    # Print final layer shapes
    print("\nFinal layer parameters:")
    for key, value in model_data.items():
        if 'final_layer' in key:
            print(f"{key}: {value.shape}")
    
    # Print first few keys to understand structure
    print("\nFirst 10 keys in model data:")
    for i, key in enumerate(list(model_data.keys())[:10]):
        print(f"{i}. {key}: {model_data[key].shape}")

# Print bias correction value
if 'bias_correction' in loaded_data:
    print(f"\nBias correction value: {loaded_data['bias_correction']}")

# Look for hidden_size or other hyperparameters
print("\nSearching for hyperparameters...")
for key, value in loaded_data.items():
    if key not in ['model', 'bias_correction']:
        print(f"{key}: {value}") 