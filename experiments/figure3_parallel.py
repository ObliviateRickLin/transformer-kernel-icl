import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datetime
import time
from collections import defaultdict

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL = 5  # Input dimension d=5 as specified in Appendix D
N_CONTEXT = 14  # Number of in-context demonstrations n=14 as shown in Figure 3
N_QUERY = 1    # Number of query points for prediction (always 1 for ICL)
TOTAL_SAMPLES = N_CONTEXT + N_QUERY
DEFAULT_NUM_LAYERS_RANGE = list(range(1, 8)) # k from 1 to 7 as shown in Figure 3
TRAINING_ITERATIONS = 2000 # 20000 in original paper
LEARNING_RATE = 1e-4
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
    """ Computes <u,v> for batches of vectors or matrices. """
    return torch.bmm(u_batch, v_batch.transpose(-1, -2))

def pt_exp_kernel(u_batch, v_batch, scaling_factor=1.0):
    """ Computes exp(scaling_factor * <u,v>) for batches. """
    linear_term = scaling_factor * pt_linear_kernel(u_batch, v_batch)
    # Clamp to prevent overflow in exp function
    linear_term = torch.clamp(linear_term, max=10.0)  # exp(10) ≈ 22026, still reasonable
    return torch.exp(linear_term)

def generate_data_from_combined_kernel(num_tasks, d_model, n_context, alpha, G1, G2, exp_scaling_factor=1.0, device=DEVICE):
    """Generate data for ICL tasks based on the combined kernel for Figure 3."""
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
        K_lin_matrix = pt_linear_kernel(X_g1.unsqueeze(0), X_g1.unsqueeze(0)).squeeze(0)
        # Data generation: Use exp_scaling_factor as passed (for Fig3c: 0.5 inside exp as per equation)
        K_exp_matrix = pt_exp_kernel(X_g2.unsqueeze(0), X_g2.unsqueeze(0), scaling_factor=exp_scaling_factor).squeeze(0)
        
        K_combined = alpha * K_lin_matrix + (1 - alpha) * K_exp_matrix
        K_combined += torch.eye(total_samples_per_task, device=device) * 1e-6

        L = torch.linalg.cholesky(K_combined)
        z = torch.randn(total_samples_per_task, 1, device=device)
        Y_i = torch.matmul(L, z)

        X_batch[i] = X_i
        Y_batch[i] = Y_i
        
    return X_batch, Y_batch

# --- Kernel Functions for h_tilde ---
def h_linear_pt(u_batch_vec, v_batch_vec):
    return torch.sum(u_batch_vec * v_batch_vec, dim=-1)

def h_exp_pt(u_batch_vec, v_batch_vec, scaling_factor=1.0):
    linear_term = scaling_factor * torch.sum(u_batch_vec * v_batch_vec, dim=-1)
    # Clamp to prevent overflow in exp function
    linear_term = torch.clamp(linear_term, max=10.0)
    return torch.exp(linear_term)

# --- Transformer Components ---
class AttentionHead(nn.Module):
    def __init__(self, d_model_cov, d_k, h_tilde_func):
        super().__init__()
        self.d_k = d_k
        self.h_tilde = h_tilde_func

        self.W_q = nn.Linear(d_model_cov, d_k, bias=False)
        self.W_k = nn.Linear(d_model_cov, d_k, bias=False)
        self.W_v = nn.Linear(d_model_cov + 1, d_model_cov + 1, bias=False)

    def forward(self, Z_l, X_l, M_mask):
        B, N, _ = X_l.shape

        queries = self.W_q(X_l)
        keys = self.W_k(X_l)
        
        q_expanded = queries.unsqueeze(2).expand(B, N, N, self.d_k)
        k_expanded = keys.unsqueeze(1).expand(B, N, N, self.d_k)
        
        attn_scores_flat = self.h_tilde(q_expanded.reshape(-1, self.d_k), k_expanded.reshape(-1, self.d_k))
        attn_scores = attn_scores_flat.reshape(B, N, N)
        
        h_tilde_M_masked = attn_scores * M_mask
        term_before_V = torch.bmm(h_tilde_M_masked, Z_l)
        update = self.W_v(term_before_V)
        return update

class TransformerLayer(nn.Module):
    def __init__(self, d_model_cov, num_heads, h_tilde_funcs_per_head):
        super().__init__()
        self.d_k = d_model_cov // num_heads if num_heads > 0 else d_model_cov 
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(AttentionHead(d_model_cov, self.d_k, h_tilde_funcs_per_head[i]))
        
    def forward(self, Z_l, X_l, M_mask):
        head_outputs_sum = torch.zeros_like(Z_l, device=Z_l.device)
        for head in self.heads:
            head_outputs_sum += head(Z_l, X_l, M_mask)
        return Z_l + head_outputs_sum

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model_cov, num_heads, h_tilde_funcs_list_for_layers):
        super().__init__()
        self.num_layers = num_layers
        self.d_model_cov = d_model_cov
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            h_tilde_funcs_for_current_layer = h_tilde_funcs_list_for_layers[i]
            self.layers.append(TransformerLayer(d_model_cov, num_heads, h_tilde_funcs_for_current_layer))

        # Modified mask matrix to enable learning
        mask_m = torch.zeros(TOTAL_SAMPLES, TOTAL_SAMPLES)
        mask_m[:N_CONTEXT, :N_CONTEXT] = torch.eye(N_CONTEXT)
        mask_m[N_CONTEXT:, :N_CONTEXT] = 1.0
        self.register_buffer('mask_matrix_M', mask_m)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize parameters with i.i.d. Gaussian as specified in paper"""
        if isinstance(module, nn.Linear):
            # Use more reasonable initialization - paper's std=1.0 causes numerical instability
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, X_batch, Y_batch_context_and_query_zeros):
        Z_l = torch.cat([X_batch, Y_batch_context_and_query_zeros], dim=-1)
        
        for layer_module in self.layers:
            current_X_l = Z_l[:, :, :self.d_model_cov]
            Z_l = layer_module(Z_l, current_X_l, self.mask_matrix_M)
            
        prediction = Z_l[:, N_CONTEXT, self.d_model_cov:]
        return prediction.squeeze(-1)

# --- Parallel Training System ---
class ParallelTrainer:
    def __init__(self):
        self.models = {}  # {model_id: (model, optimizer, config_info)}
        self.data_generators = {}  # {config_id: data_generator_func}
        self.results = defaultdict(lambda: defaultdict(list))
        
    def add_model(self, model_id, model, config_id, config_desc, model_name, num_layers):
        """Add a model to the parallel training system"""
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.models[model_id] = {
            'model': model,
            'optimizer': optimizer, 
            'config_id': config_id,
            'config_desc': config_desc,
            'model_name': model_name,
            'num_layers': num_layers,
            'losses': [],
            'val_losses': []
        }
        
    def add_data_generator(self, config_id, data_gen_func):
        """Add a data generator for a specific configuration"""
        self.data_generators[config_id] = data_gen_func
        
    def train_all_parallel(self, training_iterations, batch_size, eval_interval=100):
        """Train all models in parallel"""
        print(f"Starting parallel training of {len(self.models)} models...")
        print(f"Total GPU memory usage estimation: {len(self.models)} models")
        
        start_time = time.time()
        
        # Pre-generate data for each configuration
        config_data = {}
        
        for iteration in range(training_iterations):
            # Generate new data every 10 steps (as per Appendix D)
            if iteration % 10 == 0:
                for config_id, data_gen in self.data_generators.items():
                    X_tasks, Y_tasks = data_gen(batch_size)
                    config_data[config_id] = (X_tasks.to(DEVICE), Y_tasks.to(DEVICE))
            
            # Parallel forward/backward pass for all models
            total_loss = 0
            for model_id, model_info in self.models.items():
                model = model_info['model']
                optimizer = model_info['optimizer']
                config_id = model_info['config_id']
                
                # Get data for this model's configuration
                X_tasks, Y_tasks = config_data[config_id]
                
                Y_input_for_Z0 = Y_tasks.clone()
                Y_input_for_Z0[:, N_CONTEXT:, :] = 0
                
                predictions_y = model(X_tasks, Y_input_for_Z0)
                true_query_y = Y_tasks[:, N_CONTEXT:].squeeze()
                loss = ((predictions_y - true_query_y)**2).mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                model_info['losses'].append(loss.item())
                total_loss += loss.item()
            
            # Progress reporting
            if (iteration + 1) % eval_interval == 0 or iteration == training_iterations - 1:
                elapsed = time.time() - start_time
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                
                # Print detailed progress
                print(f"\n{timestamp} - Iteration {iteration+1}/{training_iterations} | Elapsed: {elapsed:.1f}s")
                print("=" * 70)
                
                # Print individual model results (much more useful than averages)
                print("Individual Model Results:")
                for model_id, model_info in self.models.items():
                    current_loss = model_info['losses'][-1]
                    config_id = model_info['config_id']
                    model_name = model_info['model_name'].replace(' ', '').replace('-', '')
                    num_layers = model_info['num_layers']
                    print(f"  {config_id}_{model_name}_{num_layers}: {current_loss:.6f}")
                
                overall_avg = total_loss / len(self.models)
                print(f"Overall Average: {overall_avg:.6f}")
                print("=" * 70)
                
                # Evaluate all models
                self.evaluate_all_models(config_data)
                
        print(f"Parallel training completed in {time.time() - start_time:.1f} seconds")
        
    def evaluate_all_models(self, config_data):
        """Evaluate all models in parallel"""
        print("Validation Results:")
        
        for model_id, model_info in self.models.items():
            model = model_info['model']
            config_id = model_info['config_id']
            model_name = model_info['model_name'].replace(' ', '').replace('-', '')
            num_layers = model_info['num_layers']
            
            model.eval()
            with torch.no_grad():
                X_eval, Y_eval = config_data[config_id]
                Y_input_for_Z0_eval = Y_eval.clone()
                Y_input_for_Z0_eval[:, N_CONTEXT:, :] = 0
                
                predictions_y = model(X_eval, Y_input_for_Z0_eval)
                true_query_y = Y_eval[:, N_CONTEXT:].squeeze()
                val_loss = ((predictions_y - true_query_y)**2).mean().item()
                
                model_info['val_losses'].append(val_loss)
                print(f"  {config_id}_{model_name}_{num_layers}_val: {val_loss:.6f}")
            model.train()
        print()
            
    def get_final_results(self):
        """Extract final evaluation results organized by configuration and model type"""
        for model_id, model_info in self.models.items():
            config_desc = model_info['config_desc']
            model_name = model_info['model_name']
            num_layers = model_info['num_layers']
            final_val_loss = model_info['val_losses'][-1] if model_info['val_losses'] else float('inf')
            
            # Ensure the results are stored in the right order (by num_layers)
            if len(self.results[config_desc][model_name]) < num_layers:
                self.results[config_desc][model_name].extend([0.0] * (num_layers - len(self.results[config_desc][model_name])))
            
            if len(self.results[config_desc][model_name]) == num_layers:
                self.results[config_desc][model_name].append(final_val_loss)
            else:
                self.results[config_desc][model_name][num_layers-1] = final_val_loss
                
        return dict(self.results)

# --- Main Experiment Logic ---
def run_figure3_parallel_experiment():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    print("="*80)
    print("FIGURE 3 PARALLEL REPRODUCTION - Maximum A100 Utilization")
    print("="*80)
    print(f"Using device: {DEVICE}")
    print(f"Configuration: d={D_MODEL}, n={N_CONTEXT}, batch={BATCH_SIZE}, iterations={TRAINING_ITERATIONS}")
    print(f"Parallel training: ALL models simultaneously")
    print("="*80)
    
    # Setup configurations
    G_identity = torch.eye(D_MODEL, device=DEVICE)
    
    diag1 = torch.zeros(D_MODEL, device=DEVICE)
    diag1[:2] = 1.0
    G1_fig3c = torch.diag(diag1)

    diag2 = torch.zeros(D_MODEL, device=DEVICE)  
    diag2[2:] = 1.0
    G2_fig3c = torch.diag(diag2)

    # Figure 3 configurations EXACTLY as specified in paper
    figure3_configs = [
        (1.0, G_identity, G_identity, 1.0, "Fig3a (K_linear)", "fig3a"),     # K^diamond = K^linear  
        (0.0, G_identity, G_identity, 1.0, "Fig3b (K_exp)", "fig3b"),        # K^diamond = K^exp
        (0.5, G1_fig3c, G2_fig3c, 0.5, "Fig3c (K_mixed)", "fig3c")          # K^diamond = 0.5*K^lin + 0.5*K^exp with 0.5 scaling in data's exp
    ]

    # Create model types setup.
    # The MODEL'S h_tilde functions should be standard and fixed across all data configurations.
    # The model is expected to learn to adapt to data-specific scalings (like the 0.5 in Fig3c data's exp part)
    # through its trainable weights, not by changing its own activation function's definition.
    model_types_setup = {}
    
    h_linear_model = h_linear_pt
    # Standard h_exp for the model, always with scaling_factor=1.0
    h_exp_model_standard = lambda u,v: h_exp_pt(u,v, scaling_factor=1.0) 
    
    # Standard model setup for ALL configurations (Fig3a, Fig3b, Fig3c)
    standard_models_definitions = [
        {"name": "1-Head Linear", "num_heads": 1, "h_tilde_funcs_per_head": [h_linear_model]},
        {"name": "1-Head Exp", "num_heads": 1, "h_tilde_funcs_per_head": [h_exp_model_standard]},
        {"name": "2-Head Lin+Exp", "num_heads": 2, "h_tilde_funcs_per_head": [h_linear_model, h_exp_model_standard]}
    ]
    
    model_types_setup["fig3a"] = standard_models_definitions
    model_types_setup["fig3b"] = standard_models_definitions
    model_types_setup["fig3c"] = standard_models_definitions # Fig3c now uses the SAME standard model definitions
    
    # Initialize parallel trainer
    trainer = ParallelTrainer()
    
    # Add data generators for each configuration  
    # Use correct exp_scaling_factor for each config (Fig3c needs 0.5 as per the mathematical definition)
    for alpha, G1, G2, data_exp_scaling, config_desc, config_id in figure3_configs:
        data_gen_for_config = lambda bs, a=alpha, g1=G1, g2=G2, exp_scale=data_exp_scaling: generate_data_from_combined_kernel(
            bs, D_MODEL, N_CONTEXT, a, g1, g2, exp_scaling_factor=exp_scale, device=DEVICE
        )
        trainer.add_data_generator(config_id, data_gen_for_config)
    
    # Create and add all models
    model_count = 0
    for alpha, G1, G2, data_exp_scaling, config_desc, config_id in figure3_configs:
        # Get the appropriate model setup for this configuration
        current_model_types = model_types_setup[config_id]
        
        for model_spec in current_model_types:
            model_name = model_spec["name"]
            num_h = model_spec["num_heads"]
            h_funcs_for_heads = model_spec["h_tilde_funcs_per_head"]
            
            for num_layers in DEFAULT_NUM_LAYERS_RANGE:
                h_funcs_for_all_layers = [h_funcs_for_heads] * num_layers
                
                model = Transformer(
                    num_layers=num_layers,
                    d_model_cov=D_MODEL,
                    num_heads=num_h,
                    h_tilde_funcs_list_for_layers=h_funcs_for_all_layers
                ).to(DEVICE)
                
                model_id = f"{config_id}_{model_name.replace(' ', '_').replace('-', '_')}_{num_layers}"
                trainer.add_model(model_id, model, config_id, config_desc, model_name, num_layers)
                model_count += 1
    
    print(f"Created {model_count} models for parallel training")
    print(f"Estimated memory usage: ~{model_count * 50}MB (rough estimate)")
    
    # Train all models in parallel
    trainer.train_all_parallel(TRAINING_ITERATIONS, BATCH_SIZE)
    
    # Get results and plot
    results = trainer.get_final_results()
    plot_results(results)

def plot_results(results):
    """Plot the results matching paper format"""
    print("\n--- Plotting Results ---")
    plt.figure(figsize=(18, 6))
    
    config_names = ["Fig3a (K_linear)", "Fig3b (K_exp)", "Fig3c (K_mixed)"]
    plot_idx = 1
    
    for config_desc in config_names:
        if config_desc in results:
            plt.subplot(1, 3, plot_idx)
            model_results_dict = results[config_desc]
            
            print(f"\n{config_desc} Results:")
            for model_name, losses in model_results_dict.items():
                print(f"  {model_name}: {losses}")
                
                losses_array = np.array(losses)
                losses_clipped = np.clip(losses_array, 1e-20, 10.0)
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
    plt.suptitle(f"Figure 3 Parallel Reproduction (d={D_MODEL}, n={N_CONTEXT}, iter={TRAINING_ITERATIONS}, batch={BATCH_SIZE})", fontsize=14)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figure3_parallel_reproduction_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    
    # Save results to file
    results_filename = f"figure3_parallel_results_{timestamp}.txt"
    with open(results_filename, 'w') as f:
        f.write(f"Figure 3 Parallel Reproduction Results\n")
        f.write(f"Configuration: d={D_MODEL}, n={N_CONTEXT}, iter={TRAINING_ITERATIONS}, batch={BATCH_SIZE}\n\n")
        for config_desc, model_results_dict in results.items():
            f.write(f"{config_desc}:\n")
            for model_name, losses in model_results_dict.items():
                f.write(f"  {model_name}: {losses}\n")
            f.write("\n")
    print(f"Results saved to {results_filename}")

if __name__ == "__main__":
    run_figure3_parallel_experiment() 