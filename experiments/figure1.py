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
from torch.optim.lr_scheduler import LambdaLR # Added for LR scheduling

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
DIM_D = 20  # d: Dimensionality of covariates
NUM_LAYERS = 3  # L: Number of transformer layers
LEARNING_RATE = 1e-3
BATCH_SIZE = 500 # 30000 Batch size for Adam optimizer (Strictly from paper's Appendix D)
NUM_TRAINING_STEPS = 20000 # For Figure 1
# NUM_TRAINING_STEPS = 200 # Reduced for initial testing, original is 20000
N_TEST_QUERY_POINTS = 200 # n_test: Number of query points for evaluation
SIGMA_D_DIAG_VALUES_LIST = [1.0] + [0.1] * (DIM_D - 1) if DIM_D > 1 else [1.0]

# Figure 1 specific settings
# CONTEXT_LENGTHS_N = np.arange(2, 13, 2) # Old: Context lengths n (e.g., 2, 4, ..., 12)
# For K_exp, paper's x-axis goes up to 14. For linear/relu, up to 12.
CONTEXT_LENGTHS_N_LINEAR_RELU = np.arange(2, 13, 2)
CONTEXT_LENGTHS_N_EXP = np.arange(2, 15, 2) # For K_exp, n up to 14


KAPPA_TYPES_FIG1 = [KAPPA_TYPE_LINEAR, KAPPA_TYPE_RELU, KAPPA_TYPE_EXP]
H_TYPES_FIG1 = [GeneralizedAttentionLayer.H_TYPE_RELU, 
                GeneralizedAttentionLayer.H_TYPE_LINEAR, 
                GeneralizedAttentionLayer.H_TYPE_EXP, 
                GeneralizedAttentionLayer.H_TYPE_SOFTMAX]

# Plotting settings
COLORS = {'relu': 'blue', 'linear': 'green', 'exp': 'red', 'softmax': 'orange', 'bayes': 'black'}
LINESTYLES = {'relu': '-', 'linear': '-', 'exp': '-', 'softmax': '-', 'bayes': '--'}

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
    Main function to run experiments for Figure 1.
    Iterates over kappa_types, context_lengths, and h_types.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    overall_start_time = time.time()

    sigma_matrix_torch = get_sigma_matrix(DIM_D, SIGMA_D_DIAG_VALUES_LIST, device)
    
    results = {}

    for kappa_idx, kappa_type_data in enumerate(KAPPA_TYPES_FIG1):
        kappa_start_time = time.time()
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === Processing Data Generating Kappa: {kappa_type_data} ({kappa_idx+1}/{len(KAPPA_TYPES_FIG1)}) ===")
        results[kappa_type_data] = {}
        
        if kappa_type_data == KAPPA_TYPE_EXP:
            current_context_lengths = CONTEXT_LENGTHS_N_EXP
        else:
            current_context_lengths = CONTEXT_LENGTHS_N_LINEAR_RELU
        print(f"  Context lengths for this kappa: {current_context_lengths}")

        bayes_loss_for_n_log = []
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] Calculating Bayes Loss for K={kappa_type_data}...")
        bayes_calc_start_time = time.time()
        for n_ctx_idx, n_ctx in enumerate(current_context_lengths):
            # For Bayes loss calculation, X_context_batch_torch and X_query_batch_torch are X values.
            # Y_ctx_bayes is not actually used in the current calculate_bayes_loss for variance calculation.
            X_bayes_all, _, _ = generate_batch_data(
                n_context=n_ctx, dim_d=DIM_D, sigma_matrix_torch=sigma_matrix_torch,
                kappa_type=kappa_type_data, batch_size=N_TEST_QUERY_POINTS, device=device
            )
            X_c_bayes = X_bayes_all[:, :n_ctx, :] 
            X_q_bayes = X_bayes_all[:, n_ctx:, :]
            dummy_Y_c_bayes = torch.zeros_like(X_c_bayes[:, :, :1])

            avg_bayes_var = calculate_bayes_loss(X_c_bayes, dummy_Y_c_bayes, X_q_bayes, 
                                                 kappa_type_data, DIM_D, device)
            log_avg_bayes_var = np.log(avg_bayes_var) if avg_bayes_var > 0 else -float('inf')
            bayes_loss_for_n_log.append(log_avg_bayes_var)
            print(f"    n={n_ctx} (Context {n_ctx_idx+1}/{len(current_context_lengths)}), Bayes Loss (Avg Var): {avg_bayes_var:.4e}, log(Loss): {log_avg_bayes_var:.4e}")
        results[kappa_type_data]['bayes'] = bayes_loss_for_n_log
        print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] Bayes Loss calculation for K={kappa_type_data} took {time.time() - bayes_calc_start_time:.2f}s")
        
        for h_model_idx, h_type_model in enumerate(H_TYPES_FIG1):
            h_model_start_time = time.time()
            print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Training Model Attention h_tilde: {h_type_model} ({h_model_idx+1}/{len(H_TYPES_FIG1)}) for K_data={kappa_type_data} ---")
            results[kappa_type_data][h_type_model] = []
            
            for n_ctx_idx, n_ctx in enumerate(current_context_lengths):
                n_ctx_loop_start_time = time.time()
                print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Context Length n: {n_ctx} ({n_ctx_idx+1}/{len(current_context_lengths)}) for h_type={h_type_model}")
                N_total_points_for_model = n_ctx + 1

                model = TransformerModel(num_layers=NUM_LAYERS, d=DIM_D, h_type=h_type_model, 
                                         n_plus_1=N_total_points_for_model).to(device)
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                criterion = nn.MSELoss() # Use instance for consistency if needed elsewhere
                
                # Learning rate scheduler setup
                warmup_steps = int(NUM_TRAINING_STEPS * 0.1) # 10% of steps for warmup
                total_steps = NUM_TRAINING_STEPS

                def lr_lambda_func(current_step: int):
                    if current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    # Linear decay from warmup_steps to total_steps
                    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return max(0.0, 1.0 - progress) # Decay from 1.0 (at end of warmup) to 0.0

                scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_func)

                train_loop_start_time = time.time()
                # Data is resampled every 10 steps according to Appendix D
                # Current BATCH_SIZE is 512, not 30000 as mentioned for the resampled minibatch in paper.
                # This change makes data generation 10x less frequent.
                data_for_10_steps = None
                for step in range(NUM_TRAINING_STEPS):
                    if step % 10 == 0:
                        # print(f"      Resampling data for steps {step+1}-{step+10}...")
                        X_train_seq, Y_ctx_train, Y_query_train_target = generate_batch_data(
                            n_context=n_ctx, dim_d=DIM_D, sigma_matrix_torch=sigma_matrix_torch,
                            kappa_type=kappa_type_data, batch_size=BATCH_SIZE, device=device
                        )
                        data_for_10_steps = (X_train_seq, Y_ctx_train, Y_query_train_target)
                    else:
                        # Reuse data from the last sample point
                        X_train_seq, Y_ctx_train, Y_query_train_target = data_for_10_steps

                    model.train()
                    optimizer.zero_grad()
                    
                    # Z_0_input construction remains the same
                    Z_0_input = torch.zeros(BATCH_SIZE, DIM_D + 1, N_total_points_for_model, device=device)
                    Z_0_input[:, :DIM_D, :] = X_train_seq.transpose(1, 2) 
                    Z_0_input[:, DIM_D, :n_ctx] = Y_ctx_train.squeeze(-1) 

                    Z_L_output = model(Z_0_input) 
                    Y_pred = Z_L_output[:, DIM_D, -1] 
                    Y_query_train_target_squeezed = Y_query_train_target.squeeze() 
                    
                    if Y_pred.shape != Y_query_train_target_squeezed.shape:
                        if Y_pred.ndim == 1 and Y_query_train_target_squeezed.ndim == 0: Y_query_train_target_squeezed = Y_query_train_target_squeezed.unsqueeze(0)
                        elif Y_pred.ndim == 0 and Y_query_train_target_squeezed.ndim == 1: Y_pred = Y_pred.unsqueeze(0)
                        if Y_pred.shape != Y_query_train_target_squeezed.shape: 
                            raise ValueError(f"Shape mismatch for loss: Y_pred {Y_pred.shape}, Y_true {Y_query_train_target_squeezed.shape} n_ctx={n_ctx} step={step}")

                    loss = criterion(Y_pred, Y_query_train_target_squeezed)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step() # Step the scheduler
                    
                    if step == 0 or (step + 1) % (NUM_TRAINING_STEPS // 200 or 1) == 0 or (step + 1) == NUM_TRAINING_STEPS:
                        current_loop_time = time.time()-train_loop_start_time
                        steps_done = step + 1
                        avg_time_per_step = current_loop_time / steps_done if steps_done > 0 else 0
                        est_time_remaining = avg_time_per_step * (NUM_TRAINING_STEPS - steps_done)
                        print(f"      Epoch {steps_done}/{NUM_TRAINING_STEPS}, Train MSE Loss: {loss.item():.4e} (step_avg_time: {avg_time_per_step:.3f}s, total_elapsed: {current_loop_time:.2f}s, est_remain: {est_time_remaining:.2f}s)")
                
                print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] Training for n={n_ctx} (h_type={h_type_model}) took {time.time() - train_loop_start_time:.2f}s")

                eval_start_time = time.time()
                model.eval()
                total_test_loss_val = 0
                with torch.no_grad():
                    X_test_seq, Y_ctx_test, Y_query_test_target = generate_batch_data(
                        n_context=n_ctx, dim_d=DIM_D, sigma_matrix_torch=sigma_matrix_torch,
                        kappa_type=kappa_type_data, batch_size=N_TEST_QUERY_POINTS, device=device
                    )
                    
                    Z_0_test_input = torch.zeros(N_TEST_QUERY_POINTS, DIM_D + 1, N_total_points_for_model, device=device)
                    Z_0_test_input[:, :DIM_D, :] = X_test_seq.transpose(1, 2) 
                    Z_0_test_input[:, DIM_D, :n_ctx] = Y_ctx_test.squeeze(-1) 

                    Z_L_test_output = model(Z_0_test_input)
                    Y_pred_test = Z_L_test_output[:, DIM_D, -1]
                    Y_query_test_target_squeezed = Y_query_test_target.squeeze()
                    
                    if Y_pred_test.shape != Y_query_test_target_squeezed.shape:
                        if Y_pred_test.ndim == 1 and Y_query_test_target_squeezed.ndim == 0: Y_query_test_target_squeezed = Y_query_test_target_squeezed.unsqueeze(0)
                        elif Y_pred_test.ndim == 0 and Y_query_test_target_squeezed.ndim == 1: Y_pred_test = Y_pred_test.unsqueeze(0)
                        if Y_pred_test.shape != Y_query_test_target_squeezed.shape: 
                             raise ValueError(f"Shape mismatch for test loss: Y_pred {Y_pred_test.shape}, Y_true {Y_query_test_target_squeezed.shape} n_ctx={n_ctx}")

                    test_loss = criterion(Y_pred_test, Y_query_test_target_squeezed)
                    total_test_loss_val = test_loss.item()

                avg_test_loss = total_test_loss_val 
                log_avg_test_loss = np.log(avg_test_loss) if avg_test_loss > 0 else -float('inf')
                results[kappa_type_data][h_type_model].append(log_avg_test_loss)
                print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] n={n_ctx}, Avg Test ICL Loss: {avg_test_loss:.4e}, log(Loss): {log_avg_test_loss:.4e} (Eval took: {time.time()-eval_start_time:.2f}s)")
                print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed Context Length n: {n_ctx} for h_type={h_type_model}. Total time for this n_ctx: {time.time() - n_ctx_loop_start_time:.2f}s")
            print(f"  [{time.strftime('%Y-%m-%d %H:%M:%S')}] --- Finished Model Attention h_tilde: {h_type_model} for K_data={kappa_type_data}. Total time for this h_model: {time.time() - h_model_start_time:.2f}s ---")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] === Finished Processing Data Generating Kappa: {kappa_type_data}. Total time for this kappa: {time.time() - kappa_start_time:.2f}s ===")
    
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total experiment time: {time.time() - overall_start_time:.2f} seconds")
    return results # Removed current_context_lengths from return as it's handled by plot_results

# --- Plotting ---
def plot_results(results, filename="figure1_reproduction.png"):
    """
    Plots the results similar to Figure 1 from the paper.
    results: dict from run_experiment
    filename: output filename for the plot
    """
    num_kappa_types = len(KAPPA_TYPES_FIG1)
    fig, axes = plt.subplots(1, num_kappa_types, figsize=(6 * num_kappa_types, 5), sharey=False)
    if num_kappa_types == 1: axes = [axes] # Make sure axes is always iterable

    subplot_titles = {
        KAPPA_TYPE_LINEAR: f'(a) Data: $K^{{{KAPPA_TYPE_LINEAR}}}$',
        KAPPA_TYPE_RELU: f'(b) Data: $K^{{{KAPPA_TYPE_RELU}}}$',
        KAPPA_TYPE_EXP: f'(c) Data: $K^{{{KAPPA_TYPE_EXP}}}$'
    }

    for i, kappa_type_data in enumerate(KAPPA_TYPES_FIG1):
        ax = axes[i]
        ax.set_title(subplot_titles.get(kappa_type_data, f"Data: {kappa_type_data}"))
        ax.set_xlabel("Context Length (n)")
        if i == 0:
            ax.set_ylabel("log(Test ICL Loss)")
        
        if kappa_type_data == KAPPA_TYPE_EXP:
            current_n_values_for_plot = CONTEXT_LENGTHS_N_EXP
        else:
            current_n_values_for_plot = CONTEXT_LENGTHS_N_LINEAR_RELU

        for h_or_bayes_type, losses in results[kappa_type_data].items():
            label_name = h_or_bayes_type 
            if len(losses) != len(current_n_values_for_plot):
                print(f"Warning: Mismatch in lengths for plotting. Kappa: {kappa_type_data}, Type: {label_name}. Losses len: {len(losses)}, N_values len: {len(current_n_values_for_plot)}. Skipping this plot line.")
                continue
            ax.plot(current_n_values_for_plot, losses, 
                    label=label_name, 
                    color=COLORS.get(label_name, 'gray'), 
                    linestyle=LINESTYLES.get(label_name, '-'), marker='o', markersize=4)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nPlot saved to {filename}")

if __name__ == '__main__':
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Figure 1 reproduction script...")
    
    experiment_results = run_experiment()
    # The n_values are now determined inside plot_results based on kappa_type
    plot_results(experiment_results)
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Figure 1 reproduction script finished.") 