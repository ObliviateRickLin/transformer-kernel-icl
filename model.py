import torch
import torch.nn as nn
import math

class GeneralizedAttentionLayer(nn.Module):
    # Define string constants for h_type for clarity and to avoid typos
    H_TYPE_RELU = "relu"
    H_TYPE_LINEAR = "linear"
    H_TYPE_EXP = "exp"
    H_TYPE_SOFTMAX = "softmax"

    def __init__(self, d: int, h_type: str, n_plus_1: int):
        super().__init__()
        self.d = d # This is d_covariates (e.g., DIM_D from figure1.py)
        self.h_type = h_type
        self.n_plus_1 = n_plus_1 # N = n+1 (total number of tokens)

        # Learnable weights
        # In paper's notation, Z is (d+1, N). X is (d,N), Y is (1,N)
        # Here, self.d corresponds to d_model (e.g., d_covariates + 1)
        # A_l, B_l, C_l are (d_model, d_model) if they operate on Z directly.
        # Or, if they operate on X part, they are (d_covariates, d_covariates)
        # Based on current Z_in[:, :-1, :], self.d is d_covariates.
        # Let's assume self.d in __init__ is d_covariates for now, consistent with A_l, B_l, C_l operating on X.
        # If d passed to __init__ is d_model (d_covariates+1), then A_l, B_l, C_l should be (d_model, d_model)
        # And X_in splitting needs to be Z_in directly or just X part.
        # From figure1.py, d=DIM_D (d_covariates) is passed to TransformerModel,
        # which then passes it to GeneralizedAttentionLayer.
        # So, self.d in this class IS d_covariates.
        
        # A_l, B_l, C_l operate on X (d_covariates, N)
        self.A_l = nn.Parameter(torch.randn(self.d, self.d) / math.sqrt(self.d))
        self.r_l = nn.Parameter(torch.randn(1))
        self.B_l = nn.Parameter(torch.randn(self.d, self.d) / math.sqrt(self.d))
        self.C_l = nn.Parameter(torch.randn(self.d, self.d) / math.sqrt(self.d))

    def compute_h_tilde(self, U: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Computes the core attention matrix ~h(U, W) in a batch-aware manner.
        U, W are of shape (B, d_covariates, N)
        Output is of shape (B, N, N)
        """
        # Batch matrix multiplication: (B, N, d_covariates) @ (B, d_covariates, N) -> (B, N, N)
        scores = U.transpose(-2, -1) @ W 

        if self.h_type == GeneralizedAttentionLayer.H_TYPE_LINEAR:
            return scores
        elif self.h_type == GeneralizedAttentionLayer.H_TYPE_RELU:
            return torch.relu(scores)
        elif self.h_type == GeneralizedAttentionLayer.H_TYPE_EXP:
            # Apply scaling alpha = 1/d_covariates as per Appendix D.2 (assuming self.d is d_covariates)
            # The paper mentions "alpha = 1/d" for exp(alpha * <u,v>).
            # If scores = U^T W where U,W are B_l X, C_l X, then <u,v> involves X^T B_l^T C_l X.
            # The scaling in exp should be applied to the argument of exp.
            # Here scores is already U^T W. If U and W implicitly contain B_l, C_l, this is fine.
            # Original paper: h(U_i, W_j) = exp(<U_i, W_j> / d). Assuming <U_i, W_j> is a scalar dot product.
            # Here U and W are matrices, U.transpose @ W gives matrix of products.
            # self.d is d_covariates
            if self.d == 0: # Should not happen with d_covariates
                return torch.exp(scores)
            return torch.exp(scores / self.d) # self.d is d_covariates
        elif self.h_type == GeneralizedAttentionLayer.H_TYPE_SOFTMAX:
            # scores shape: (B, N, N)
            # Paper: "the final row of the matrix ~h(U,W) is 0 and the other rows sum to 1 over columns."
            # This means for rows i in [0, N-2], sum_j h_tilde[i,j] = 1.
            # This is standard softmax applied along the last dimension for these rows.
            
            # Apply softmax to all rows first along the last dimension (dim=-1 or dim=2 for (B,N,N))
            softmax_result = torch.nn.functional.softmax(scores, dim=-1) # (B, N, N)
            
            # Then, zero out the last row (N-th row, index N-1, corresponding to self.n_plus_1 - 1)
            # Clone the tensor before applying the in-place modification to avoid autograd error.
            if self.n_plus_1 > 0: # Ensure N (self.n_plus_1) is at least 1
                cloned_softmax_result = softmax_result.clone()
                cloned_softmax_result[:, self.n_plus_1 - 1, :] = 0.0
                return cloned_softmax_result
            else:
                # If n_plus_1 <= 0 (e.g. no query token, which shouldn't happen in typical use cases here),
                # return the softmax result unmodified.
                return softmax_result
        else:
            raise ValueError(f"Unknown h_type: {self.h_type}")

    def forward(self, Z_in: torch.Tensor) -> torch.Tensor:
        """
        Z_in: shape (B, d_model, N) where d_model = d_covariates + 1
        Output: Z_out: shape (B, d_model, N)
        """
        original_ndim = Z_in.ndim
        if original_ndim == 2: # If a single sample (d_model, N) is passed
            Z_in = Z_in.unsqueeze(0) # Add batch dimension: (1, d_model, N)

        # X_in: (B, d_covariates, N), Y_in: (B, 1, N)
        # self.d is d_covariates. d_model is self.d + 1.
        X_in = Z_in[:, :self.d, :] 
        Y_in = Z_in[:, self.d:, :] # This ensures Y_in is (B, 1, N) 

        # U = B_l X_in, W = C_l X_in
        # self.B_l, self.C_l are (d_covariates, d_covariates). X_in is (B, d_covariates, N).
        # Result of @ is (B, d_covariates, N)
        U_matrix = self.B_l @ X_in 
        W_matrix = self.C_l @ X_in 

        # Call the batch-aware compute_h_tilde
        # U_matrix, W_matrix are (B, d_covariates, N)
        # att_core will be (B, N, N)
        att_core = self.compute_h_tilde(U_matrix, W_matrix)

        att_masked = att_core 
        if self.h_type in [GeneralizedAttentionLayer.H_TYPE_LINEAR, GeneralizedAttentionLayer.H_TYPE_RELU, GeneralizedAttentionLayer.H_TYPE_EXP]:
            att_masked = att_core.clone() # Clone to modify
            if self.n_plus_1 > 0: 
                 # Zeros out only the (N-1, N-1) entry (0-indexed) for each batch item
                 att_masked[:, self.n_plus_1 - 1, self.n_plus_1 - 1] = 0.0 
        elif self.h_type == GeneralizedAttentionLayer.H_TYPE_SOFTMAX:
            # Softmax's compute_h_tilde already ensures its last row is zero.
            pass
        
        # delta_X = self.A_l @ (X_in @ att_masked)
        # X_in is (B, d_covariates, N), att_masked is (B, N, N)
        # torch.bmm(X_in, att_masked) -> (B, d_covariates, N)
        # self.A_l (d_covariates, d_covariates) @ (B, d_covariates, N) -> (B, d_covariates, N)
        delta_X = self.A_l @ torch.bmm(X_in, att_masked) 
        
        # delta_Y = self.r_l * (Y_in @ att_masked)
        # Y_in is (B, 1, N), att_masked is (B, N, N)
        # torch.bmm(Y_in, att_masked) -> (B, 1, N)
        delta_Y = self.r_l * torch.bmm(Y_in, att_masked)

        X_out = X_in + delta_X
        Y_out = Y_in + delta_Y
        
        Z_out = torch.cat((X_out, Y_out), dim=1) # Concatenate along the d_model dimension
        
        if original_ndim == 2:
            Z_out = Z_out.squeeze(0) # Remove batch dimension if it was added
            
        return Z_out

class TransformerModel(nn.Module):
    def __init__(self, num_layers: int, d: int, h_type: str, n_plus_1: int):
        super().__init__()
        # d passed here is d_covariates from figure1.py (DIM_D)
        self.d_covariates = d 
        # The GeneralizedAttentionLayer will operate on Z which has d_model = d_covariates + 1 channels.
        # However, the A_l, B_l, C_l matrices in GeneralizedAttentionLayer are (d_covariates, d_covariates)
        # as they are defined using self.d which is d_covariates.
        # The r_l scalar operates on the Y part.
        self.n_plus_1 = n_plus_1
        self.layers = nn.ModuleList(
            [GeneralizedAttentionLayer(self.d_covariates, h_type, self.n_plus_1) for _ in range(num_layers)]
        )

    def forward(self, Z_0: torch.Tensor) -> torch.Tensor:
        """
        Z_0: shape (B, d_model, N) or (d_model, N) - Initial input
             where d_model = d_covariates + 1
        Output: Z_L: shape (B, d_model, N) or (d_model, N) - Output after all layers
        """
        Z_current = Z_0
        for layer in self.layers:
            Z_current = layer(Z_current)
        return Z_current

    def predict(self, Z_final: torch.Tensor) -> torch.Tensor:
        """
        Extracts the prediction Y_query = Z_final[d_covariates, N-1] (0-indexed for Y row, query token column)
        Z_final: shape (B, d_model, N) or (d_model, N)
        Output: (B,) or scalar tensor
        """
        if Z_final.ndim == 3: # Batched input
            return Z_final[:, self.d_covariates, self.n_plus_1 - 1]
        elif Z_final.ndim == 2: # Single sample
            return Z_final[self.d_covariates, self.n_plus_1 - 1]
        else:
            raise ValueError(f"Unsupported Z_final ndim: {Z_final.ndim}")

if __name__ == '__main__':
    # Basic test
    d_cov = 5 # d_covariates
    d_model = d_cov + 1
    n_ctx = 10 
    N_total_tokens = n_ctx + 1 
    num_model_layers = 3
    batch_size_test = 4
    
    # Test each h_type
    for h_type_test in [GeneralizedAttentionLayer.H_TYPE_LINEAR, 
                        GeneralizedAttentionLayer.H_TYPE_RELU, 
                        GeneralizedAttentionLayer.H_TYPE_EXP, 
                        GeneralizedAttentionLayer.H_TYPE_SOFTMAX]:
        print(f"Testing h_type: {h_type_test}")
        # Pass d_covariates to TransformerModel
        model = TransformerModel(num_layers=num_model_layers, d=d_cov, h_type=h_type_test, n_plus_1=N_total_tokens)
        
        # Create a dummy Z_0 input (batched)
        dummy_X_batch = torch.randn(batch_size_test, d_cov, N_total_tokens)
        dummy_Y_context_batch = torch.randn(batch_size_test, 1, n_ctx)
        dummy_Y_query_batch = torch.zeros(batch_size_test, 1, 1)
        dummy_Y_batch = torch.cat((dummy_Y_context_batch, dummy_Y_query_batch), dim=2) # (B, 1, N)
        dummy_Z_0_batch = torch.cat((dummy_X_batch, dummy_Y_batch), dim=1) # (B, d_model, N)
        
        print(f"Input Z_0_batch shape: {dummy_Z_0_batch.shape}")

        # Forward pass
        Z_final_output_batch = model(dummy_Z_0_batch)
        print(f"Output Z_final_batch shape: {Z_final_output_batch.shape}")
        
        # Prediction
        prediction_batch = model.predict(Z_final_output_batch) # predict now handles batch
        print(f"Prediction_batch shape: {prediction_batch.shape}, example value: {prediction_batch[0].item()}")
        
        # Check gradients (simple backprop)
        try:
            target_batch = torch.randn(batch_size_test) # Target for each item in batch
            loss = ((prediction_batch - target_batch)**2).mean() # Mean squared error over batch
            loss.backward()
            print("Backward pass successful for batch.")
            has_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    has_grad = True
            if has_grad:
                print("Gradients found in model parameters.")
            else:
                print("No gradients found in model parameters.")
        except Exception as e:
            print(f"Error during backward pass for batch: {e}")
        print("-" * 30)

    # Test for N=1 (e.g. only query point, n_ctx = 0) - single sample
    print("Testing h_type: linear with N=1 (n_ctx=0) - single sample")
    n_ctx_single = 0
    N_single = n_ctx_single + 1
    model_single = TransformerModel(num_layers=1, d=d_cov, h_type=GeneralizedAttentionLayer.H_TYPE_LINEAR, n_plus_1=N_single)
    dummy_X_s = torch.randn(d_cov, N_single)
    dummy_Y_s = torch.zeros(1, N_single) 
    dummy_Z_0_s = torch.cat((dummy_X_s, dummy_Y_s), dim=0) # (d_model, N_single)
    print(f"Input Z_0 shape (N=1, single): {dummy_Z_0_s.shape}")
    Z_final_s = model_single(dummy_Z_0_s) # Model forward handles unbatched then re-batches
    print(f"Output Z_final shape (N=1, single): {Z_final_s.shape}")
    prediction_s = model_single.predict(Z_final_s)
    print(f"Prediction TF shape (N=1, single): {prediction_s.shape}, value: {prediction_s.item()}")
    target_s = torch.randn_like(prediction_s) # Ensure target matches prediction shape
    loss_s = ((prediction_s - target_s)**2).mean()
    loss_s.backward()
    print("Backward pass successful for N=1 (single).")
    print("-" * 30) 