import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F # Added for potential use, though not strictly in current version
import datetime # Added for timestamps
import time # Added for detailed timing

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 5  # Input dimension d=5 as specified in Appendix D
N_CONTEXT = 14  # Number of in-context demonstrations n=14 as shown in Figure 3
N_QUERY = 1    # Number of query points for prediction (always 1 for ICL)
TOTAL_SAMPLES = N_CONTEXT + N_QUERY
DEFAULT_NUM_LAYERS_RANGE = list(range(1, 8)) # k from 1 to 7 as shown in Figure 3
TRAINING_ITERATIONS = 1000 # Increased for better convergence
LEARNING_RATE = 1e-3  
BATCH_SIZE = 30000 # Restored to paper specification from Appendix D
EVAL_BATCH_SIZE = 1000 # For evaluation

# Appendix D specifies: D is diagonal matrix with entries (1, 1, 0.25, 2.25, 1)
SIGMA_D_DIAGONAL = torch.tensor([1.0, 1.0, 0.25, 2.25, 1.0], device=DEVICE)

# --- Helper Functions (Data Generation - PyTorch based) ---
def generate_x_on_unit_sphere(num_samples, d_model, device=DEVICE):
    """Generates x_i from a unit sphere, on specified device."""
    x = torch.randn(num_samples, d_model, device=device)
    x = x / torch.norm(x, dim=1, keepdim=True)
    return x

def generate_covariance_matrix_sigma(d_model, device=DEVICE):
    """Generate covariance matrix Σ = U^T D U as specified in Appendix D"""
    # Generate uniformly random orthogonal matrix U
    random_matrix = torch.randn(d_model, d_model, device=device)
    U, _, _ = torch.linalg.svd(random_matrix)
    
    # D is diagonal matrix with entries (1, 1, 0.25, 2.25, 1)
    D = torch.diag(SIGMA_D_DIAGONAL)
    
    # Σ = U^T D U
    Sigma = U.T @ D @ U
    return Sigma, U

def pt_linear_kernel(u_batch, v_batch):
    """ Computes <u,v> for batches of vectors or matrices.
        u_batch: (batch, ..., dim) or (batch, num_samples, dim)
        v_batch: (batch, ..., dim) or (batch, num_samples, dim)
        Output for K_ij = <u_i, v_j> will be (batch, num_samples_u, num_samples_v)
    """
    return torch.bmm(u_batch, v_batch.transpose(-1, -2))

def pt_exp_kernel(u_batch, v_batch, scaling_factor=1.0):
    """ Computes exp(scaling_factor * <u,v>) for batches. """
    return torch.exp(scaling_factor * pt_linear_kernel(u_batch, v_batch))

def generate_data_from_combined_kernel(num_tasks, d_model, n_context, alpha, G1, G2, exp_scaling_factor=1.0, device=DEVICE):
    """
    Generates data for ICL tasks based on the combined kernel for Figure 3.
    K_circle(u,v) = alpha * K_linear(G1 u, G1 v) + (1-alpha) * K_exp(G2 u, G2 v)
    
    Following Appendix D specifications exactly:
    - Covariates: x^(i) = Σ^(1/2) ξ^(i), where ξ^(i) iid from unit sphere
    - Dimension d=5
    - Covariance Σ = U^T D U with D=diag(1,1,0.25,2.25,1)
    """
    total_samples_per_task = n_context + 1
    X_batch = torch.zeros(num_tasks, total_samples_per_task, d_model, device=device)
    Y_batch = torch.zeros(num_tasks, total_samples_per_task, 1, device=device)

    # Generate covariance matrix according to Appendix D
    Sigma, U = generate_covariance_matrix_sigma(d_model, device)
    Sigma_half = torch.linalg.cholesky(Sigma + 1e-6 * torch.eye(d_model, device=device))

    G1_dev = G1.to(device)
    G2_dev = G2.to(device)

    for i in range(num_tasks):
        # Generate ξ^(i) from unit sphere, then x^(i) = Σ^(1/2) ξ^(i)
        xi_i = generate_x_on_unit_sphere(total_samples_per_task, d_model, device=device)
        X_i = torch.matmul(xi_i, Sigma_half.T)  # x^(i) = Σ^(1/2) ξ^(i)

        # Apply G-matrices: x_transformed = x @ G for applying G to rows of x
        X_g1 = torch.matmul(X_i, G1_dev) # (total_samples, d_model)
        X_g2 = torch.matmul(X_i, G2_dev) # (total_samples, d_model)

        # Compute individual kernel matrices
        K_lin_matrix = pt_linear_kernel(X_g1.unsqueeze(0), X_g1.unsqueeze(0)).squeeze(0) # (total_samples, total_samples)
        K_exp_matrix = pt_exp_kernel(X_g2.unsqueeze(0), X_g2.unsqueeze(0), scaling_factor=exp_scaling_factor).squeeze(0)
        
        K_combined = alpha * K_lin_matrix + (1 - alpha) * K_exp_matrix
        K_combined += torch.eye(total_samples_per_task, device=device) * 1e-6 # For numerical stability

        L = torch.linalg.cholesky(K_combined)
        z = torch.randn(total_samples_per_task, 1, device=device)
        Y_i = torch.matmul(L, z)

        X_batch[i] = X_i
        Y_batch[i] = Y_i
        
    return X_batch, Y_batch

# --- Kernel Functions for h_tilde (passed to AttentionHead) ---
# These now expect batch inputs: u_batch_vec (B, D_k), v_batch_vec (B, D_k)
# and should return (B,) for batched scalar outputs K(u,v)
def h_linear_pt(u_batch_vec, v_batch_vec):
    return torch.sum(u_batch_vec * v_batch_vec, dim=-1)

def h_exp_pt(u_batch_vec, v_batch_vec, scaling_factor=1.0):
    return torch.exp(scaling_factor * torch.sum(u_batch_vec * v_batch_vec, dim=-1))

# --- Transformer Components (PyTorch based, self-contained) ---
class AttentionHead(nn.Module):
    def __init__(self, d_model_cov, d_k, h_tilde_func): # d_model_cov is dimension of X
        super().__init__()
        self.d_k = d_k
        self.h_tilde = h_tilde_func # This function K(u,v) should return a scalar (or batched scalars)

        self.W_q = nn.Linear(d_model_cov, d_k, bias=False)
        self.W_k = nn.Linear(d_model_cov, d_k, bias=False)
        # V_l projects Z_l M h(...). Z_l has d_model_cov + 1 dimensions.
        self.W_v = nn.Linear(d_model_cov + 1, d_model_cov + 1, bias=False)

    def forward(self, Z_l, X_l, M_mask):
        # Z_l: (B, Total_Samples, D_model_cov + 1)
        # X_l: (B, Total_Samples, D_model_cov) - covariates from Z_l
        # M_mask: (Total_Samples, Total_Samples) - Paper's M mask [[I_n,0],[0,0]]
        
        B, N, _ = X_l.shape # B=batch_size, N=TOTAL_SAMPLES

        queries = self.W_q(X_l) # (B, N, D_k)
        keys = self.W_k(X_l)    # (B, N, D_k)
        
        # Compute h_tilde(Query_i, Key_j) for all i,j
        # attn_scores[b, i, j] = h_tilde(queries[b,i,:], keys[b,j,:])
        # h_tilde function expects (B, D_k) and (B, D_k) to produce (B,)
        # We need to expand and then call.
        q_expanded = queries.unsqueeze(2).expand(B, N, N, self.d_k) # (B, N_q, N_k, D_k)
        k_expanded = keys.unsqueeze(1).expand(B, N, N, self.d_k)   # (B, N_q, N_k, D_k)
        
        # h_tilde takes two vectors. We have batches of pairs of vectors.
        # Reshape to (B*N*N, D_k) to pass to h_tilde
        attn_scores_flat = self.h_tilde(q_expanded.reshape(-1, self.d_k), k_expanded.reshape(-1, self.d_k))
        attn_scores = attn_scores_flat.reshape(B, N, N) # (B, N, N)
        
        # Apply paper's M mask: M * h(...)
        h_tilde_M_masked = attn_scores * M_mask # (B, N, N) element-wise broadcast M_mask
        
        # Compute update: V_l (Z_l @ (M*h(...)))
        # Z_l is (B, N, D_z), h_tilde_M_masked is (B, N, N)
        # The (i,j) entry of (M*h) is M_ij * h_ij.
        # (Z_l @ (M*h))_ik = sum_j Z_lj * M_jk * h_jk --- this comment implies Z_l is on left
        # However, standard attention is AttnWeights @ Values.
        # h_tilde_M_masked is (B, N_samples, N_samples)
        # Z_l is (B, N_samples, D_model_cov + 1)
        # Correct operation: h_tilde_M_masked @ Z_l
        term_before_V = torch.bmm(h_tilde_M_masked, Z_l) # (B, N, D_model_cov + 1)
        
        update = self.W_v(term_before_V) # (B, N, D_model_cov + 1)
        return update

class TransformerLayer(nn.Module):
    def __init__(self, d_model_cov, num_heads, h_tilde_funcs_per_head):
        super().__init__()
        # d_model_cov is the dimensionality of X, the covariates.
        # d_k for Q,K projections. For simplicity, d_k can be d_model_cov or d_model_cov // num_heads
        # The paper's V matrices are (d+1)x(d+1). W_v in AttentionHead reflects this.
        self.d_k = d_model_cov // num_heads if num_heads > 0 else d_model_cov 
        
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(AttentionHead(d_model_cov, self.d_k, h_tilde_funcs_per_head[i]))
        
    def forward(self, Z_l, X_l, M_mask):
        head_outputs_sum = torch.zeros_like(Z_l, device=Z_l.device)
        for head in self.heads:
            head_outputs_sum += head(Z_l, X_l, M_mask) # Summing according to Eq. 3.14
            
        return Z_l + head_outputs_sum # Residual connection

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model_cov, num_heads, h_tilde_funcs_list_for_layers):
        super().__init__()
        self.num_layers = num_layers
        self.d_model_cov = d_model_cov # Dimensionality of covariates X
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # h_tilde_funcs_list_for_layers is a list (per layer) of lists (per head) of functions
            h_tilde_funcs_for_current_layer = h_tilde_funcs_list_for_layers[i]
            self.layers.append(TransformerLayer(d_model_cov, num_heads, h_tilde_funcs_for_current_layer))

        # Mask M: Modified to enable learning while preserving causal structure
        # Paper's literal definition [[I_n, 0], [0, 0]] prevents query from learning
        # This version allows query to attend to context (essential for ICL)
        mask_m = torch.zeros(TOTAL_SAMPLES, TOTAL_SAMPLES)
        mask_m[:N_CONTEXT, :N_CONTEXT] = torch.eye(N_CONTEXT)  # Context self-attention
        mask_m[N_CONTEXT:, :N_CONTEXT] = 1.0  # Query can attend to all context
        # Query cannot attend to other queries or itself (maintains causality)
        self.register_buffer('mask_matrix_M', mask_m)
        
        # Initialize parameters according to paper: i.i.d. Gaussian 
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize parameters with i.i.d. Gaussian as specified in paper"""
        if isinstance(module, nn.Linear):
            # Standard deviation can be adjusted if needed
            nn.init.normal_(module.weight, mean=0.0, std=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, X_batch, Y_batch_context_and_query_zeros):
        # X_batch: (B, Total_Samples, D_model_cov)
        # Y_batch_context_and_query_zeros: (B, Total_Samples, 1) - context y is real, query y is 0
        
        Z_l = torch.cat([X_batch, Y_batch_context_and_query_zeros], dim=-1) # (B, Total_Samples, D_model_cov + 1)
        
        for layer_module in self.layers:
            # Extract X_l from current Z_l for Q, K projections if needed, or pass Z_l if V,Q,K act on Z
            # Current AttentionHead W_q, W_k expect X_l (covariates only)
            current_X_l = Z_l[:, :, :self.d_model_cov]
            Z_l = layer_module(Z_l, current_X_l, self.mask_matrix_M)
            
        # Prediction is the (d_model_cov+1)-th element (y part) of the (N_CONTEXT)-th entry (query, 0-indexed) of Z_L
        prediction = Z_l[:, N_CONTEXT, self.d_model_cov:] # (B, 1, 1) if N_QUERY=1 and last dim of Z_l is y
        return prediction.squeeze(-1) # (B, 1) -> (B,) if N_QUERY=1

# --- Training and Evaluation ---
def train_transformer(model, data_generator_func, training_iterations, batch_size, lr, device=DEVICE, eval_interval=50):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    val_losses = []

    print(f"Starting training with batch_size={batch_size}, iterations={training_iterations}")
    train_loop_start_time = time.time()

    # Variables to store the current minibatch, resampled every 10 steps (as per Appendix D)
    current_X_tasks = None
    current_Y_tasks = None

    for iteration in range(training_iterations):
        # Resample the minibatch every 10 steps
        if iteration % 10 == 0:
            current_X_tasks, current_Y_tasks = data_generator_func(batch_size)
            current_X_tasks, current_Y_tasks = current_X_tasks.to(device), current_Y_tasks.to(device)
        
        Y_input_for_Z0 = current_Y_tasks.clone()
        Y_input_for_Z0[:, N_CONTEXT:, :] = 0 # Query y is 0 for input Z0
        
        predictions_y = model(current_X_tasks, Y_input_for_Z0)
        true_query_y = current_Y_tasks[:, N_CONTEXT:].squeeze()
        loss = ((predictions_y - true_query_y)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping as specified in Appendix D
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())

        # Print progress less frequently for large batch sizes
        if (iteration + 1) % eval_interval == 0 or iteration == training_iterations - 1:
            val_loss = evaluate_transformer(model, data_generator_func, EVAL_BATCH_SIZE, device)
            val_losses.append(val_loss)
            
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            current_train_time = time.time() - train_loop_start_time
            steps_done = iteration + 1
            
            print(f"{timestamp} - Iter {steps_done}/{training_iterations} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f} (elapsed: {current_train_time:.1f}s)")
            
    return losses, val_losses

def evaluate_transformer(model, data_generator_func, num_eval_tasks, device=DEVICE):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        # Generate all eval data at once if num_eval_tasks is not too large for memory
        X_eval, Y_eval = data_generator_func(num_eval_tasks)
        X_eval, Y_eval = X_eval.to(device), Y_eval.to(device)

        Y_input_for_Z0_eval = Y_eval.clone()
        Y_input_for_Z0_eval[:, N_CONTEXT:, :] = 0
        
        predictions_y = model(X_eval, Y_input_for_Z0_eval)
        true_query_y = Y_eval[:, N_CONTEXT:].squeeze()
        
        loss = ((predictions_y - true_query_y)**2).mean()
        total_loss = loss.item()
    return total_loss

# --- Main Experiment Logic for Figure 3 ---
def run_figure3_experiment():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    print("="*80)
    print("FIGURE 3 REPRODUCTION - Transformers Implement Functional Gradient Descent")
    print("="*80)
    print(f"Using device: {DEVICE}")
    print(f"Configuration: d={D_MODEL}, n={N_CONTEXT}, batch={BATCH_SIZE}, iterations={TRAINING_ITERATIONS}")
    print(f"Note: Using modified mask matrix that allows query→context attention")
    print(f"      This is necessary to enable learning (strict paper mask blocks learning)")
    print("="*80)
    
    G_identity = torch.eye(D_MODEL, device=DEVICE)
    
    # Figure 3c configuration as specified in the paper:
    # G1 = diag([1,1,0,0,0]) (first 2 dimensions)  
    # G2 = diag([0,0,1,1,1]) (last 3 dimensions)
    diag1 = torch.zeros(D_MODEL, device=DEVICE)
    diag1[:2] = 1.0  # First 2 dims for linear part
    G1_fig3c = torch.diag(diag1)

    diag2 = torch.zeros(D_MODEL, device=DEVICE)  
    diag2[2:] = 1.0  # Last 3 dims for exp part
    G2_fig3c = torch.diag(diag2)

    # Figure 3 configurations exactly as described in paper
    # Note: exp_scaling_factor should be 0.5 for Fig3c as mentioned in paper
    figure3_configs = [
        (1.0, G_identity, G_identity, 1.0, "Fig3a (K_linear)"),
        (0.0, G_identity, G_identity, 1.0, "Fig3b (K_exp)"), 
        (0.5, G1_fig3c, G2_fig3c, 0.5, "Fig3c (K_mixed)")  # α=0.5, exp_scale=0.5 as in paper
    ]

    # Model h_tilde functions - keep scaling simple and consistent 
    h_linear_model = h_linear_pt
    h_exp_model_scale_1 = lambda u,v: h_exp_pt(u,v, scaling_factor=1.0)

    model_types_setup = [
        {"name": "1-Head Linear", "num_heads": 1, "h_tilde_funcs_per_head": [h_linear_model]},
        {"name": "1-Head Exp", "num_heads": 1, "h_tilde_funcs_per_head": [h_exp_model_scale_1]},
        {"name": "2-Head Lin+Exp", "num_heads": 2, "h_tilde_funcs_per_head": [h_linear_model, h_exp_model_scale_1]}
    ]
    
    results = {}

    for alpha, G1, G2, data_exp_scaling, config_desc in figure3_configs:
        print(f"\n--- Running Configuration: {config_desc} ---")
        print(f"    α={alpha}, G1=identity:{torch.allclose(G1, G_identity)}, G2=identity:{torch.allclose(G2, G_identity)}")
        print(f"    Data exp_scaling={data_exp_scaling}")
        
        results[config_desc] = {}

        data_gen_for_config = lambda bs: generate_data_from_combined_kernel(
            bs, D_MODEL, N_CONTEXT, alpha, G1, G2, exp_scaling_factor=data_exp_scaling, device=DEVICE
        )

        for model_spec_template in model_types_setup:
            model_name = model_spec_template["name"]
            num_h = model_spec_template["num_heads"]
            h_funcs_for_heads = model_spec_template["h_tilde_funcs_per_head"]

            print(f"\n  Training Model Type: {model_name}")
            results[config_desc][model_name] = []
            
            for num_layers in DEFAULT_NUM_LAYERS_RANGE:
                print(f"    Layers: {num_layers}", end=" -> ")
                
                # h_tilde_funcs_list_for_layers: list (layers) of list (heads) of functions
                h_funcs_for_all_layers = [h_funcs_for_heads] * num_layers

                transformer_model = Transformer(
                    num_layers=num_layers,
                    d_model_cov=D_MODEL,
                    num_heads=num_h,
                    h_tilde_funcs_list_for_layers=h_funcs_for_all_layers
                ).to(DEVICE)
                
                # Train transformer
                _, _ = train_transformer(transformer_model, data_gen_for_config, 
                                  TRAINING_ITERATIONS, BATCH_SIZE, LEARNING_RATE, device=DEVICE, 
                                  eval_interval=100)
                
                eval_loss = evaluate_transformer(transformer_model, data_gen_for_config, EVAL_BATCH_SIZE, device=DEVICE)
                results[config_desc][model_name].append(eval_loss)
                print(f"Eval Loss: {eval_loss:.6f}")
    
    # --- Plotting Results ---
    print("\n--- Plotting Results ---")
    plt.figure(figsize=(18, 6))
    plot_idx = 1
    
    for config_desc, model_results_dict in results.items():
        plt.subplot(1, len(figure3_configs), plot_idx)
        
        # Print results for debugging
        print(f"\n{config_desc} Results:")
        for model_name, losses in model_results_dict.items():
            print(f"  {model_name}: {losses}")
            
            # Plot with log scale and clipping for stability
            losses_array = np.array(losses)
            losses_clipped = np.clip(losses_array, 1e-20, 10.0)  # Allow very small losses as in paper
            log_losses = np.log(losses_clipped)
            
            plt.plot(DEFAULT_NUM_LAYERS_RANGE, log_losses, 
                    label=model_name, marker='o', linewidth=2, markersize=5)
        
        plt.title(f"{config_desc}\nlog(Loss) vs Layer Depth", fontsize=10)
        plt.xlabel("Layer Depth", fontsize=9)
        plt.ylabel("log(Loss)", fontsize=9)
        plt.xticks(DEFAULT_NUM_LAYERS_RANGE)
        plt.legend(fontsize=7)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=8)
        plot_idx += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(f"Figure 3 Reproduction (d={D_MODEL}, n={N_CONTEXT}, iter={TRAINING_ITERATIONS}, batch={BATCH_SIZE})", fontsize=14)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figure3_reproduction_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    
    # Also save results to file for analysis
    results_filename = f"figure3_results_{timestamp}.txt"
    with open(results_filename, 'w') as f:
        f.write(f"Figure 3 Reproduction Results\n")
        f.write(f"Configuration: d={D_MODEL}, n={N_CONTEXT}, iter={TRAINING_ITERATIONS}, batch={BATCH_SIZE}\n\n")
        for config_desc, model_results_dict in results.items():
            f.write(f"{config_desc}:\n")
            for model_name, losses in model_results_dict.items():
                f.write(f"  {model_name}: {losses}\n")
            f.write("\n")
    print(f"Results saved to {results_filename}")

if __name__ == "__main__":
    run_figure3_experiment() 