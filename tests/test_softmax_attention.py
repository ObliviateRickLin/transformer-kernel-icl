import torch
import sys
import os
import math

print("DEBUG: Script execution started.") # ADDED FOR DEBUGGING

# Add the parent directory (newreproduce) to sys.path
# so that we can import model
current_script_path = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(tests_dir) # Should be /path/to/newreproduce
sys.path.insert(0, parent_dir)

from model import GeneralizedAttentionLayer, TransformerModel

def run_softmax_test():
    print("--- Running Softmax Attention Test ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test parameters
    d_cov = 5  # d_covariates
    d_model = d_cov + 1
    n_ctx = 10
    N_total_tokens = n_ctx + 1
    num_model_layers = 1 # Keep it simple for a focused test
    batch_size_test = 4
    h_type_test = GeneralizedAttentionLayer.H_TYPE_SOFTMAX

    print(f"Testing h_type: {h_type_test}")
    print(f"Parameters: d_cov={d_cov}, N_total_tokens={N_total_tokens}, num_layers={num_model_layers}, batch_size={batch_size_test}")

    # Instantiate model
    # Pass d_covariates to TransformerModel
    model = TransformerModel(num_layers=num_model_layers, d=d_cov, h_type=h_type_test, n_plus_1=N_total_tokens).to(device)

    # Create a dummy Z_0 input (batched)
    dummy_X_batch = torch.randn(batch_size_test, d_cov, N_total_tokens, device=device)
    dummy_Y_context_batch = torch.randn(batch_size_test, 1, n_ctx, device=device)
    dummy_Y_query_batch = torch.zeros(batch_size_test, 1, 1, device=device) # Query Y is often initially zero
    dummy_Y_batch = torch.cat((dummy_Y_context_batch, dummy_Y_query_batch), dim=2)  # (B, 1, N)
    dummy_Z_0_batch = torch.cat((dummy_X_batch, dummy_Y_batch), dim=1)  # (B, d_model, N)
    
    # Transpose X to match expected input for Z_0: (B, d_covariates, N) -> (B, d_model, N) after cat with Y
    # Actually, figure1.py does Z_0_input[:, :DIM_D, :] = X_train_seq.transpose(1, 2)
    # So, X_train_seq is (B, N, d_covariates).
    # Let's adjust dummy_X_batch to be (B, N, d_covariates) then transpose it when creating Z_0.
    
    dummy_X_for_Z0 = torch.randn(batch_size_test, N_total_tokens, d_cov, device=device) # (B, N, d_cov)
    
    # Construct Z_0 input as in figure1.py
    Z_0_input = torch.zeros(batch_size_test, d_model, N_total_tokens, device=device)
    Z_0_input[:, :d_cov, :] = dummy_X_for_Z0.transpose(1,2) # (B, d_cov, N)
    Z_0_input[:, d_cov, :n_ctx] = dummy_Y_context_batch.squeeze(1) # Corrected: Squeeze dim 1

    print(f"Input Z_0_batch shape: {Z_0_input.shape}")

    # --- Forward pass ---
    model.train() # Ensure model is in training mode for gradients
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    Z_final_output_batch = model(Z_0_input)
    print(f"Output Z_final_batch shape: {Z_final_output_batch.shape}")

    # --- Prediction & Loss ---
    # Use the model's predict method
    # The predict method in model.py: Z_final[:, self.d_covariates, self.n_plus_1 - 1]
    # This extracts the Y component of the query token.
    prediction_batch = model.predict(Z_final_output_batch)
    print(f"Prediction_batch shape: {prediction_batch.shape}")

    # Create a dummy target for the predictions
    # Target should be (B,) for batch_size predictions
    target_batch = torch.randn(batch_size_test, device=device) 
    if prediction_batch.shape != target_batch.shape:
        # Handle case where batch_size might be 1 and one is (1,) and other is (1)
        if prediction_batch.numel() == target_batch.numel():
             target_batch = target_batch.reshape(prediction_batch.shape)
        else:
            print(f"Shape mismatch: Prediction {prediction_batch.shape}, Target {target_batch.shape}")
            # Fallback if shapes are truly incompatible for a simple MSE
            # This might happen if N_total_tokens or batch_size is 1 leading to squeezed dims
            if prediction_batch.numel() == batch_size_test : # if prediction is (B)
                 pass # target is already (B)
            else: # If prediction is scalar or other shape, make target match
                 target_batch = torch.randn_like(prediction_batch)


    loss = ((prediction_batch - target_batch) ** 2).mean()
    print(f"Calculated Loss: {loss.item()}")

    # --- Backward pass & Gradient Check ---
    try:
        loss.backward()
        print("Backward pass successful for Softmax attention.")
        
        # Check for gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                # print(f"Gradient found for: {name}")
            # else:
                # print(f"No gradient for: {name}") # Can be verbose
        
        if has_grad:
            print("Gradients were successfully computed for model parameters.")
        else:
            print("WARNING: No gradients found in model parameters after backward pass!")
        optimizer.step() # Try an optimizer step
        print("Optimizer step successful.")

    except RuntimeError as e:
        print(f"ERROR during backward pass for Softmax attention: {e}")
        if "inplace operation" in str(e):
            print("This is the 'inplace operation' error we were trying to fix.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False
        
    print("--- Softmax Attention Test Passed ---")
    return True

if __name__ == '__main__':
    print("DEBUG: Entering main block.") # ADDED FOR DEBUGGING
    test_passed = run_softmax_test()
    if test_passed:
        print("\\nTest Result: Softmax attention appears to be working correctly with gradient computation.")
    else:
        print("\\nTest Result: Softmax attention test FAILED.") 