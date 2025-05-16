import numpy as np
from scipy.linalg import sqrtm, eigh

def generate_unit_sphere_samples(num_samples: int, dim: int) -> np.ndarray:
    """
    Generates num_samples points of dimension dim, sampled uniformly from the surface of a unit sphere.
    Each sample xi will have ||xi||_2 = 1.

    Args:
        num_samples: The number of samples to generate.
        dim: The dimensionality of each sample.

    Returns:
        A numpy array of shape (num_samples, dim) where each row is a sample.
    """
    samples = np.random.randn(num_samples, dim)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    unit_samples = samples / norms
    return unit_samples

def generate_random_orthogonal_matrix(dim: int) -> np.ndarray:
    """
    Generates a random orthogonal matrix of size dim x dim.

    Args:
        dim: The dimension of the square matrix.

    Returns:
        A numpy array of shape (dim, dim) representing an orthogonal matrix.
    """
    # Generate a random matrix from a standard normal distribution
    random_matrix = np.random.randn(dim, dim)
    # Perform QR decomposition
    q, _ = np.linalg.qr(random_matrix)
    return q 

def generate_covariance_sigma(dim: int, D_diag_values: tuple) -> np.ndarray:
    """
    Generates the covariance matrix Sigma = U^T D U.

    Args:
        dim: The dimension of the square matrix Sigma and D.
        D_diag_values: A tuple or list of diagonal entries for matrix D.
                       Its length must be equal to dim.

    Returns:
        A numpy array of shape (dim, dim) representing the covariance matrix Sigma.
    """
    if len(D_diag_values) != dim:
        raise ValueError("Length of D_diag_values must be equal to dim.")

    U = generate_random_orthogonal_matrix(dim)
    D_matrix = np.diag(D_diag_values)
    
    # Sigma = U^T D U
    # According to Appendix D: "Sigma = U^T D U, where U is a uniformly random orthogonal matrix"
    # However, in many conventions, if U is from QR of A (A=QR), U is orthogonal. 
    # If we want U to be "random orthogonal", then U itself is the random orthogonal matrix.
    # The paper uses U from "changes across seeds", implying U is random. Sigma = U D U^T or U^T D U is standard for changing basis.
    # Let's stick to U^T D U as per the text to be precise.
    sigma_matrix = np.dot(U.T, np.dot(D_matrix, U))
    
    return sigma_matrix 

def generate_covariates_X(num_samples: int, dim_d: int, sigma_matrix: np.ndarray) -> np.ndarray:
    """
    Generates covariates x^(i) = Sigma^(1/2) xi^(i),
    where xi^(i) are sampled iid from the unit sphere.

    Args:
        num_samples: The number of covariate samples to generate (n+1 in paper context).
        dim_d: The dimensionality of each covariate.
        sigma_matrix: The covariance matrix Sigma (dim_d x dim_d).

    Returns:
        A numpy array of shape (num_samples, dim_d) representing the covariates X.
    """
    if sigma_matrix.shape != (dim_d, dim_d):
        raise ValueError(f"Sigma matrix shape must be ({dim_d}, {dim_d}).")

    xi_samples = generate_unit_sphere_samples(num_samples, dim_d)
    
    # Calculate Sigma^(1/2)
    # sqrtm can return complex matrices if Sigma is not positive semi-definite.
    # However, Sigma is constructed as U^T D U with D having non-negative diags (usually positive for covariance),
    # so it should be positive semi-definite. We take the real part to handle potential numerical inaccuracies.
    sigma_sqrt = np.real(sqrtm(sigma_matrix))

    # x_samples = xi_samples @ sigma_sqrt.T (if xi_samples are rows)
    # or x_samples = sigma_sqrt @ xi_samples (if xi_samples are columns)
    # Paper: x^(i) = Sigma^(1/2) xi^(i). Assuming xi^(i) is a column vector.
    # Our generate_unit_sphere_samples returns rows. So, (xi^(i).T) is a column vector.
    # X = [Sigma^(1/2)xi^(1).T, ..., Sigma^(1/2)xi^(N).T].T
    # X_i = (Sigma^(1/2) xi_i^T)^T = xi_i (Sigma^(1/2))^T
    # Since Sigma is symmetric, Sigma^(1/2) is also symmetric. So (Sigma^(1/2))^T = Sigma^(1/2).
    # Thus X_i = xi_i Sigma^(1/2)
    
    X_covariates = np.dot(xi_samples, sigma_sqrt) # xi_samples (N, d), sigma_sqrt (d,d) -> (N,d)
    
    return X_covariates 

# Kernel functions for Gaussian Process data generation (kappa)
KAPPA_TYPE_LINEAR = "linear"
KAPPA_TYPE_RELU = "relu"
KAPPA_TYPE_EXP = "exp"

def kappa_linear(x_i: np.ndarray, x_j: np.ndarray, dim_d: int) -> float:
    return np.dot(x_i, x_j)

def kappa_relu(x_i: np.ndarray, x_j: np.ndarray, dim_d: int) -> float:
    return max(0, np.dot(x_i, x_j))

def kappa_exp(x_i: np.ndarray, x_j: np.ndarray, dim_d: int) -> float:
    if dim_d == 0: # Should not happen, d is for scaling
        return np.exp(np.dot(x_i, x_j))
    return np.exp(np.dot(x_i, x_j) / dim_d) # alpha = 1/d scaling

KAPPA_FUNCTIONS = {
    KAPPA_TYPE_LINEAR: kappa_linear,
    KAPPA_TYPE_RELU: kappa_relu,
    KAPPA_TYPE_EXP: kappa_exp,
}

def build_covariance_matrix(X: np.ndarray, kappa_type: str, dim_d: int) -> np.ndarray:
    """
    Builds the covariance matrix K(X) where K_ij = kappa(x_i, x_j).
    Args:
        X: Covariates, shape (num_samples, dim_d).
        kappa_type: Type of kappa kernel ('linear', 'relu', 'exp').
        dim_d: Dimensionality, used for exp kernel scaling.
    Returns:
        Covariance matrix K, shape (num_samples, num_samples).
    """
    num_samples = X.shape[0]
    K_matrix = np.zeros((num_samples, num_samples))
    kappa_func = KAPPA_FUNCTIONS.get(kappa_type)
    if kappa_func is None:
        raise ValueError(f"Unknown kappa_type: {kappa_type}")

    for i in range(num_samples):
        for j in range(num_samples):
            K_matrix[i, j] = kappa_func(X[i], X[j], dim_d)
    return K_matrix

def get_k_plus_matrix(K_matrix: np.ndarray, kappa_type: str) -> np.ndarray:
    """
    Computes K_+(X) by taking the absolute value of eigenvalues of K(X) if kappa is non-PSD (e.g., relu).
    For PSD kernels (linear, exp), K_+ = K.
    """
    # Based on paper, linear and exp kernels used are PSD. ReLU kernel is not always PSD.
    if kappa_type == KAPPA_TYPE_RELU:
        # Ensure K_matrix is symmetric for eigh
        K_symmetric = (K_matrix + K_matrix.T) / 2.0
        eigenvalues, eigenvectors = eigh(K_symmetric)
        # print(f"RELU KERNEL EIGENVALUES (min, max): {eigenvalues.min()}, {eigenvalues.max()}") # For debugging
        # Take absolute value of eigenvalues
        abs_eigenvalues = np.abs(eigenvalues)
        # Reconstruct K_plus
        K_plus_matrix = eigenvectors @ np.diag(abs_eigenvalues) @ eigenvectors.T
        # Ensure symmetry for the reconstructed matrix due to potential floating point inaccuracies
        return (K_plus_matrix + K_plus_matrix.T) / 2.0 
    else:
        # For linear and exp, assume they are PSD and K_plus = K
        return K_matrix

def generate_gaussian_process_labels(X: np.ndarray, kappa_type: str, dim_d: int, noise_std: float = 1e-6) -> np.ndarray:
    """
    Generates labels Y from a Gaussian Process N(0, K_+(X)).
    Args:
        X: Covariates, shape (num_samples, dim_d).
        kappa_type: Type of kappa kernel ('linear', 'relu', 'exp').
        dim_d: Dimensionality of covariates (d), used for exp kernel scaling.
        noise_std: Small standard deviation for diagonal noise to ensure K_plus is well-conditioned for sampling.
                   Paper doesn't explicitly mention this for label generation but it is common practice.
                   This is different from y_i = f(x_i) + eps_i type of noise.
                   This is for numerical stability of Cholesky decomposition if used by multivariate_normal.
    Returns:
        Labels Y, shape (num_samples,).
    """
    num_samples = X.shape[0]
    K_raw = build_covariance_matrix(X, kappa_type, dim_d)
    K_plus = get_k_plus_matrix(K_raw, kappa_type)

    # Add small diagonal noise for numerical stability before sampling
    # This helps ensure K_plus is positive definite for np.random.multivariate_normal
    # Particularly important if some abs_eigenvalues in K_plus were zero or very small.
    K_plus_stable = K_plus + np.eye(num_samples) * (noise_std**2)
    
    mean_vector = np.zeros(num_samples)
    try:
        Y = np.random.multivariate_normal(mean_vector, K_plus_stable, check_valid='warn', tol=1e-8)
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError during multivariate_normal sampling: {e}")
        print(f"K_plus_stable condition number: {np.linalg.cond(K_plus_stable)}")
        # Fallback or re-throw, for now, let's re-throw after printing info
        raise e
    except ValueError as e:
        # This can happen if covariance matrix is not PSD despite efforts.
        print(f"ValueError during multivariate_normal sampling (often PSD issue): {e}")
        print(f"Min eigenvalue of K_plus_stable before sampling: {np.min(np.linalg.eigvalsh(K_plus_stable))}")
        raise e
        
    return Y 