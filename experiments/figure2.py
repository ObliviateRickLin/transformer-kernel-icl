import sys
import os

# Add the parent directory (newreproduce) to sys.path
# so that we can import data_utils and model
current_script_path = os.path.abspath(__file__)
experiments_dir = os.path.dirname(current_script_path) # Should be /path/to/newreproduce/experiments
parent_dir = os.path.dirname(experiments_dir) # Should be /path/to/newreproduce
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time # For tracking experiment time

from data_utils import (
    generate_covariance_sigma,
    generate_covariates_X,
    generate_gaussian_process_labels,
    KAPPA_TYPE_LINEAR, KAPPA_TYPE_RELU, KAPPA_TYPE_EXP,
    build_covariance_matrix, 
    get_k_plus_matrix,
    KAPPA_FUNCTIONS
)
from model import TransformerModel, GeneralizedAttentionLayer

# --- Experiment Configuration ---
# Common settings from Appendix D
DIM_D = 5  # d: Dimensionality of covariates (Changed for Fig 2 based on Appendix D)
# NUM_LAYERS will be iterated as k_val in run_experiment for Figure 2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32 # Reduced for quick test
NUM_TRAINING_STEPS = 10 # Reduced for quick test
N_TEST_QUERY_POINTS = 10 # Reduced for quick test
SIGMA_D_DIAG_VALUES_LIST = [1.0, 1.0, 0.25, 2.25, 1.0] # From Appendix D for d=5
NUM_AVERAGE_RUNS = 1 # Reduced for quick test

# Figure 2 specific settings
NUM_LAYERS_K_SWEEP = np.arange(1, 3, 1) # k: Number of transformer layers (Reduced for quick test: 1, 2)

CONTEXT_LENGTH_N_FIG2_AB = 14 # n for Fig 2a, 2b (top row)
CONTEXT_LENGTH_N_FIG2_CD = 6  # n for Fig 2c, 2d (bottom row)

DATA_KAPPA_FIG2_AC = KAPPA_TYPE_RELU # Data generating kernel for Fig 2a, 2c
DATA_KAPPA_FIG2_BD = KAPPA_TYPE_EXP   # Data generating kernel for Fig 2b, 2d

MODEL_H_TYPES_FIG2_AC = [
    GeneralizedAttentionLayer.H_TYPE_LINEAR,
    GeneralizedAttentionLayer.H_TYPE_RELU,
    GeneralizedAttentionLayer.H_TYPE_EXP
] # Model attention types for ReLU data (Fig 2a, 2c)

MODEL_H_TYPES_FIG2_BD = [
    GeneralizedAttentionLayer.H_TYPE_LINEAR,
    GeneralizedAttentionLayer.H_TYPE_RELU,
    GeneralizedAttentionLayer.H_TYPE_EXP,
    GeneralizedAttentionLayer.H_TYPE_SOFTMAX
] # Model attention types for Exp data (Fig 2b, 2d)

# These were from Figure 1, keep them commented or remove if not cross-used
# CONTEXT_LENGTHS_N = np.arange(2, 13, 2)
# KAPPA_TYPES_FIG1 = [KAPPA_TYPE_LINEAR, KAPPA_TYPE_RELU, KAPPA_TYPE_EXP]
# H_TYPES_FIG1 = [GeneralizedAttentionLayer.H_TYPE_RELU,
#                 GeneralizedAttentionLayer.H_TYPE_LINEAR,
#                 GeneralizedAttentionLayer.H_TYPE_EXP,
#                 GeneralizedAttentionLayer.H_TYPE_SOFTMAX]

# Plotting settings
COLORS = {
    GeneralizedAttentionLayer.H_TYPE_LINEAR: 'green',
    GeneralizedAttentionLayer.H_TYPE_RELU: 'blue',
    GeneralizedAttentionLayer.H_TYPE_EXP: 'red',
    GeneralizedAttentionLayer.H_TYPE_SOFTMAX: 'orange',
    'bayes': 'black'
}
LINESTYLES = {
    GeneralizedAttentionLayer.H_TYPE_LINEAR: '-',
    GeneralizedAttentionLayer.H_TYPE_RELU: '-',
    GeneralizedAttentionLayer.H_TYPE_EXP: '-',
    GeneralizedAttentionLayer.H_TYPE_SOFTMAX: '-',
    'bayes': '--'
}

# --- Helper Functions ---

def get_sigma_matrix(dim_d, diag_values_list, device):
    sigma_matrix_np = generate_covariance_sigma(dim=dim_d, D_diag_values=diag_values_list)
    sigma_matrix_torch = torch.from_numpy(sigma_matrix_np).float().to(device)
    return sigma_matrix_torch

def generate_batch_data(n_context, dim_d, sigma_matrix_torch, kappa_type, batch_size, device):
    sigma_matrix_np = sigma_matrix_torch.cpu().numpy()

    N_total_points = n_context + 1
    
    X_sequence_list_np = []
    for _ in range(batch_size):
        X_np = generate_covariates_X(num_samples=N_total_points, dim_d=dim_d, sigma_matrix=sigma_matrix_np)
        X_sequence_list_np.append(X_np)
    
    X_sequence_batch_np = np.stack(X_sequence_list_np)
    X_sequence_batch_torch = torch.from_numpy(X_sequence_batch_np).float().to(device)

    Y_sequence_list_np = []
    for i in range(batch_size):
        X_item_np = X_sequence_batch_np[i]
        Y_item_np = generate_gaussian_process_labels(
            X=X_item_np, 
            kappa_type=kappa_type, 
            dim_d=dim_d,
            noise_std=1e-6 
        ) 
        Y_sequence_list_np.append(Y_item_np)

    Y_sequence_batch_np = np.stack(Y_sequence_list_np)
    Y_sequence_batch_torch = torch.from_numpy(Y_sequence_batch_np).float().unsqueeze(-1).to(device)
    
    Z_batch = X_sequence_batch_torch

    Z_batch_as_X = X_sequence_batch_torch

    Y_context_batch = Y_sequence_batch_torch[:, :n_context, :]
    Y_query_batch = Y_sequence_batch_torch[:, n_context:, :]
    
    return Z_batch_as_X, Y_context_batch, Y_query_batch

def calculate_bayes_loss(X_context_batch_torch, Y_context_batch_torch, X_query_batch_torch, kappa_type, dim_d, device, sigma_diag_noise=1e-5):
    # X_context_batch_torch: (batch_size, n_context, dim_d)
    # Y_context_batch_torch: (batch_size, n_context, 1) - NOT USED by numpy bayes loss if only variance is needed.
    # X_query_batch_torch: (batch_size, 1, dim_d)
    # kappa_type, dim_d are for numpy functions
    # device is for torch.eye

    batch_size = X_context_batch_torch.shape[0]
    n_context = X_context_batch_torch.shape[1]
    bayes_losses_np = [] # Store individual numpy float variances

    # Convert batch inputs to NumPy for data_utils functions
    X_context_batch_np = X_context_batch_torch.cpu().numpy()
    X_query_batch_np = X_query_batch_torch.cpu().numpy()

    for i in range(batch_size):
        X_c_np = X_context_batch_np[i] # (n_context, dim_d)
        x_q_np = X_query_batch_np[i]   # (1, dim_d)

        # build_covariance_matrix and get_k_plus_matrix are numpy based
        # K_cc = build_covariance_matrix(X1=X_c, X2=None, kappa_type=kappa_type, d=dim_d, exp_alpha=exp_alpha_val, device=device)
        # The numpy version of build_covariance_matrix takes X (num_samples, dim_d), kappa_type, dim_d.
        # It does not take X2, exp_alpha, or device.
        K_cc_np = build_covariance_matrix(X=X_c_np, kappa_type=kappa_type, dim_d=dim_d)
        K_cc_plus_np = get_k_plus_matrix(K_matrix=K_cc_np, kappa_type=kappa_type)
        
        # Convert K_cc_plus_np to torch tensor for inversion with torch.linalg.inv
        K_cc_plus_torch = torch.from_numpy(K_cc_plus_np).float().to(device)
        
        try:
            # Inverse operation is on torch tensors
            K_cc_inv_torch = torch.linalg.inv(K_cc_plus_torch + torch.eye(n_context, device=device) * sigma_diag_noise)
        except torch.linalg.LinAlgError as e:
            print(f"LinAlgError inverting K_cc_plus_torch for Bayes loss (n={n_context}, bs_item={i}): {e}. Using pseudo-inverse.")
            K_cc_inv_torch = torch.linalg.pinv(K_cc_plus_torch + torch.eye(n_context, device=device) * sigma_diag_noise)
        
        # K_qc and K_qq also use numpy functions
        # K_qc = build_covariance_matrix(X1=x_q, X2=X_c, kappa_type=kappa_type, d=dim_d, exp_alpha=exp_alpha_val, device=device)
        # build_covariance_matrix needs to be called differently if X1 and X2 are specified.
        # The current numpy version in data_utils.py: build_covariance_matrix(X, kappa_type, dim_d)
        # This implies it only computes K(X,X). We need K(X_query, X_context) and K(X_query, X_query).
        # This is another MAJOR INCONSISTENCY with data_utils.py (numpy version).
        # The numpy version of build_covariance_matrix CANNOT compute K(X1, X2) if X1 != X2 directly.
        # It computes K_ij = kappa_func(X[i], X[j], dim_d).
        # For K_qc (K(x_q, X_c)), we need to compute row vector [kappa(x_q, X_c[0]), kappa(x_q, X_c[1]), ...]
        # For K_qq (K(x_q, x_q)), this is just a scalar kappa(x_q, x_q).

        # Assuming KAPPA_FUNCTIONS is accessible or we redefine mini-helpers here for K_qc, K_qq
        kappa_func_from_du = KAPPA_FUNCTIONS.get(kappa_type) # KAPPA_FUNCTIONS is in data_utils
        if kappa_func_from_du is None:
            raise ValueError(f"Unknown kappa_type for Bayes loss: {kappa_type}")

        # K_qc: shape (1, n_context)
        K_qc_np_row = np.array([kappa_func_from_du(x_q_np.squeeze(0), X_c_np[j], dim_d) for j in range(n_context)]).reshape(1, n_context)
        
        # K_qq: shape (1, 1)
        K_qq_scalar_np = kappa_func_from_du(x_q_np.squeeze(0), x_q_np.squeeze(0), dim_d)
        # get_k_plus_matrix expects a matrix. For a scalar, K_plus is just K if K is non-negative (which it should be for K(x,x)).
        # Or, if we strictly follow, we form a 1x1 matrix.
        K_qq_matrix_np = np.array([[K_qq_scalar_np]])
        K_qq_plus_matrix_np = get_k_plus_matrix(K_matrix=K_qq_matrix_np, kappa_type=kappa_type)

        # Convert to torch tensors for matrix multiplication
        K_qc_torch = torch.from_numpy(K_qc_np_row).float().to(device) # (1, n_context)
        K_qq_plus_torch = torch.from_numpy(K_qq_plus_matrix_np).float().to(device) # (1,1)
        
        # term = K_qc @ K_cc_inv @ K_qc.T
        term_torch = K_qc_torch @ K_cc_inv_torch @ K_qc_torch.T # (1,1)
        variance_torch = K_qq_plus_torch[0,0] - term_torch[0,0]
        bayes_losses_np.append(variance_torch.item()) # .item() converts scalar tensor to python float
        
    return np.mean(bayes_losses_np) # Return as numpy float

# --- Main Experiment Logic ---

def run_experiment():
    """
    Main function to run experiments for Figure 2.
    Iterates over configurations for Fig 2a, 2b, 2c, 2d.
    Results are averaged over NUM_AVERAGE_RUNS.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define configurations for each part of Figure 2
    figure_parts_config = {
        '2a': {'data_kappa': DATA_KAPPA_FIG2_AC, 'n_ctx': CONTEXT_LENGTH_N_FIG2_AB, 'h_types': MODEL_H_TYPES_FIG2_AC, 'desc': "Data: ReLU, n=14"},
        '2b': {'data_kappa': DATA_KAPPA_FIG2_BD, 'n_ctx': CONTEXT_LENGTH_N_FIG2_AB, 'h_types': MODEL_H_TYPES_FIG2_BD, 'desc': "Data: Exp, n=14"},
        '2c': {'data_kappa': DATA_KAPPA_FIG2_AC, 'n_ctx': CONTEXT_LENGTH_N_FIG2_CD, 'h_types': MODEL_H_TYPES_FIG2_AC, 'desc': "Data: ReLU, n=6"},
        '2d': {'data_kappa': DATA_KAPPA_FIG2_BD, 'n_ctx': CONTEXT_LENGTH_N_FIG2_CD, 'h_types': MODEL_H_TYPES_FIG2_BD, 'desc': "Data: Exp, n=6"}
    }

    all_results_fig2 = {}

    for part_key, config in figure_parts_config.items():
        print(f"\n--- Running Experiment for Figure Part: {part_key} ({config['desc']}) ---")
        data_kappa_type = config['data_kappa']
        n_ctx_fixed = config['n_ctx']
        model_h_types_to_run = config['h_types']
        
        part_results = {'k_values': NUM_LAYERS_K_SWEEP, 'bayes_log_loss': None, 'model_log_losses': {}}

        # 1. Calculate Bayes Loss (averaged over runs)
        # Bayes loss depends on data_kappa and n_ctx_fixed, but not on model's k or h_type.
        # It will be a scalar value, repeated across k_values for plotting.
        print(f"  Calculating Average Bayes Loss for K_data={data_kappa_type}, n={n_ctx_fixed}...")
        sum_bayes_loss_scalar_across_runs = 0.0
        for run_idx in range(NUM_AVERAGE_RUNS):
            print(f"    Bayes Loss Run {run_idx + 1}/{NUM_AVERAGE_RUNS}")
            # Sigma matrix changes per run as U is random
            current_sigma_matrix_torch = get_sigma_matrix(DIM_D, SIGMA_D_DIAG_VALUES_LIST, device)
            
            X_bayes_all, _, _ = generate_batch_data(
                n_context=n_ctx_fixed, dim_d=DIM_D, sigma_matrix_torch=current_sigma_matrix_torch,
                kappa_type=data_kappa_type, batch_size=N_TEST_QUERY_POINTS, device=device
            )
            X_context_bayes = X_bayes_all[:, :n_ctx_fixed, :]
            X_query_bayes = X_bayes_all[:, n_ctx_fixed:, :]
            dummy_Y_context_bayes = torch.zeros_like(X_context_bayes[:, :, :1])

            bayes_loss_val_this_run = calculate_bayes_loss(
                X_context_batch_torch=X_context_bayes,
                Y_context_batch_torch=dummy_Y_context_bayes, 
                X_query_batch_torch=X_query_bayes,
                kappa_type=data_kappa_type,
                dim_d=DIM_D,
                device=device
            )
            sum_bayes_loss_scalar_across_runs += bayes_loss_val_this_run
            print(f"      Run {run_idx+1} Bayes Loss: {bayes_loss_val_this_run:.4e}")
        
        avg_bayes_loss_scalar = sum_bayes_loss_scalar_across_runs / NUM_AVERAGE_RUNS
        part_results['bayes_log_loss'] = np.log(avg_bayes_loss_scalar) # Store single log value
        print(f"  Average Bayes Loss (K_data={data_kappa_type}, n={n_ctx_fixed}): {avg_bayes_loss_scalar:.4e}, Log: {part_results['bayes_log_loss']:.4e}")

        # 2. Train and Evaluate Models for different h_types and k_layers (averaged over runs)
        for h_type_model in model_h_types_to_run:
            print(f"  Training Model: Attention h_type={h_type_model} for K_data={data_kappa_type}, n={n_ctx_fixed}")
            # Accumulate losses for this h_type across all runs and all k_layers
            # Shape: (NUM_AVERAGE_RUNS, len(NUM_LAYERS_K_SWEEP))
            all_runs_model_test_losses_for_k_sweep = np.zeros((NUM_AVERAGE_RUNS, len(NUM_LAYERS_K_SWEEP)))

            for run_idx in range(NUM_AVERAGE_RUNS):
                print(f"    Run {run_idx + 1}/{NUM_AVERAGE_RUNS} for h_type={h_type_model}")
                # Sigma matrix changes per run
                current_sigma_matrix_torch_model = get_sigma_matrix(DIM_D, SIGMA_D_DIAG_VALUES_LIST, device)
                
                for k_idx, k_val in enumerate(NUM_LAYERS_K_SWEEP):
                    print(f"      Num Layers k={k_val}")
                    
                    model = TransformerModel(num_layers=k_val, d=DIM_D, h_type=h_type_model, n_plus_1=n_ctx_fixed + 1)
                    model.to(device)
                    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                    criterion = nn.MSELoss()

                    model.train()
                    for step in range(NUM_TRAINING_STEPS):
                        optimizer.zero_grad()
                        X_train_batch, Y_context_train, Y_query_train = generate_batch_data(
                            n_context=n_ctx_fixed, dim_d=DIM_D, sigma_matrix_torch=current_sigma_matrix_torch_model,
                            kappa_type=data_kappa_type, batch_size=BATCH_SIZE, device=device
                        )
                        X_train_batch_permuted = X_train_batch.permute(0, 2, 1)
                        y_query_zeros = torch.zeros(BATCH_SIZE, 1, 1, device=device)
                        Y_input_for_Z0 = torch.cat(
                            (Y_context_train.permute(0,2,1), y_query_zeros), dim=2
                        )
                        Z_0_input = torch.cat((X_train_batch_permuted, Y_input_for_Z0), dim=1)
                        Z_L_output = model(Z_0_input)
                        Y_pred = Z_L_output[:, DIM_D, -1]
                        Y_true = Y_query_train.squeeze()
                        if Y_pred.shape != Y_true.shape:
                            if Y_pred.ndim == 1 and Y_true.ndim == 0: Y_true = Y_true.unsqueeze(0)
                            elif Y_pred.ndim == 0 and Y_true.ndim == 1: Y_pred = Y_pred.unsqueeze(0)
                            if Y_pred.shape != Y_true.shape: raise ValueError(f"Shape mismatch for loss: Y_pred {Y_pred.shape}, Y_true {Y_true.shape}")
                        loss = criterion(Y_pred, Y_true)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        if (step + 1) % (NUM_TRAINING_STEPS // 5) == 0: # Print 5 times during training
                             print(f"        Step [{step+1}/{NUM_TRAINING_STEPS}], Loss: {loss.item():.4e}")
                    
                    model.eval()
                    with torch.no_grad():
                        X_test_batch, Y_context_test, Y_query_test = generate_batch_data(
                            n_context=n_ctx_fixed, dim_d=DIM_D, sigma_matrix_torch=current_sigma_matrix_torch_model,
                            kappa_type=data_kappa_type, batch_size=N_TEST_QUERY_POINTS, device=device
                        )
                        X_test_batch_permuted = X_test_batch.permute(0, 2, 1)
                        y_query_zeros_test = torch.zeros(N_TEST_QUERY_POINTS, 1, 1, device=device)
                        Y_input_for_Z0_test = torch.cat((Y_context_test.permute(0,2,1), y_query_zeros_test), dim=2)
                        Z_0_input_test = torch.cat((X_test_batch_permuted, Y_input_for_Z0_test), dim=1)
                        Z_L_output_test = model(Z_0_input_test)
                        Y_pred_test = Z_L_output_test[:, DIM_D, -1]
                        Y_true_test = Y_query_test.squeeze()
                        if Y_pred_test.shape != Y_true_test.shape:
                            if Y_pred_test.ndim == 1 and Y_true_test.ndim == 0: Y_true_test = Y_true_test.unsqueeze(0)
                            elif Y_pred_test.ndim == 0 and Y_true_test.ndim == 1: Y_pred_test = Y_pred_test.unsqueeze(0)
                            if Y_pred_test.shape != Y_true_test.shape: raise ValueError(f"Shape mismatch for test loss: Y_pred {Y_pred_test.shape}, Y_true {Y_true_test.shape}")
                        test_loss = criterion(Y_pred_test, Y_true_test)
                        all_runs_model_test_losses_for_k_sweep[run_idx, k_idx] = test_loss.item()
                        print(f"      k={k_val}, Run={run_idx+1}, Test ICL Loss: {test_loss.item():.4e}")
            
            # Average the test losses across runs for this h_type_model
            avg_model_test_losses_k_sweep = np.mean(all_runs_model_test_losses_for_k_sweep, axis=0)
            part_results['model_log_losses'][h_type_model] = np.log(avg_model_test_losses_k_sweep)
            print(f"  Avg Test Log Losses for h_type={h_type_model} vs k: {part_results['model_log_losses'][h_type_model]}")

        all_results_fig2[part_key] = part_results
    
    return all_results_fig2


def plot_results(all_results_fig2, filename_prefix="figure2_reproduction"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    plot_info = {
        '2a': {'ax_idx': 0, 'title': 'Fig 2a: Data ReLU, n=14'},
        '2b': {'ax_idx': 1, 'title': 'Fig 2b: Data Exp, n=14'},
        '2c': {'ax_idx': 2, 'title': 'Fig 2c: Data ReLU, n=6'},
        '2d': {'ax_idx': 3, 'title': 'Fig 2d: Data Exp, n=6'}
    }

    for part_key, results_for_part in all_results_fig2.items():
        if part_key not in plot_info: continue
        
        ax_idx = plot_info[part_key]['ax_idx']
        ax = axes_flat[ax_idx]
        k_values = results_for_part['k_values']

        # Plot Bayes Loss (same log value repeated across k_values)
        bayes_log_loss_scalar = results_for_part['bayes_log_loss']
        ax.plot(k_values, np.full_like(k_values, bayes_log_loss_scalar, dtype=float), 
                label='Bayes', color=COLORS['bayes'], linestyle=LINESTYLES['bayes'])

        # Plot Model Losses
        for h_type_model, model_log_losses_vs_k in results_for_part['model_log_losses'].items():
            ax.plot(k_values, model_log_losses_vs_k, label=f"H: {h_type_model}", 
                    color=COLORS.get(h_type_model, 'grey'), linestyle=LINESTYLES.get(h_type_model, '-'))
        
        ax.set_title(plot_info[part_key]['title'])
        if ax_idx % 2 == 0: # Left column
            ax.set_ylabel("log(Test ICL Loss)")
        if ax_idx // 2 == 1: # Bottom row
            ax.set_xlabel("Number of Layers (k)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    final_filename = f"{filename_prefix}_all_subplots.png"
    plt.savefig(final_filename)
    print(f"\nPlot saved to {final_filename}")


if __name__ == "__main__":
    start_time = time.time()
    
    experiment_results_fig2 = run_experiment()
    plot_results(experiment_results_fig2, filename_prefix="figure2_reproduction")
    
    end_time = time.time()
    print(f"Total experiment time: {end_time - start_time:.2f} seconds") 