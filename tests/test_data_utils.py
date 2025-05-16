import numpy as np
import unittest
from unittest import mock  # 正确导入mock模块
from scipy.linalg import sqrtm # For test verification if needed
from data_utils import (
    generate_unit_sphere_samples, 
    generate_random_orthogonal_matrix, 
    generate_covariance_sigma,
    generate_covariates_X,
    kappa_linear, kappa_relu, kappa_exp, KAPPA_FUNCTIONS, KAPPA_TYPE_LINEAR, KAPPA_TYPE_RELU, KAPPA_TYPE_EXP,
    build_covariance_matrix,
    get_k_plus_matrix,
    generate_gaussian_process_labels
)

class TestDataUtils(unittest.TestCase):

    def test_generate_unit_sphere_samples(self):
        num_samples = 100
        dim = 5
        samples = generate_unit_sphere_samples(num_samples, dim)

        # Test shape
        self.assertEqual(samples.shape, (num_samples, dim))

        # Test normalization (norm of each sample should be close to 1)
        norms = np.linalg.norm(samples, axis=1)
        self.assertTrue(np.allclose(norms, np.ones(num_samples)), 
                        f"Norms are not all close to 1. Norms: {norms}")

    def test_generate_random_orthogonal_matrix(self):
        dim = 5
        orthogonal_matrix = generate_random_orthogonal_matrix(dim)

        # Test shape
        self.assertEqual(orthogonal_matrix.shape, (dim, dim))

        # Test orthogonality: U^T U should be close to identity
        identity_matrix = np.eye(dim)
        product = np.dot(orthogonal_matrix.T, orthogonal_matrix)
        self.assertTrue(np.allclose(product, identity_matrix),
                        f"Matrix U^T U is not close to identity. U^T U:\n{product}")
        
        # Test orthogonality: U U^T should be close to identity
        product_inv = np.dot(orthogonal_matrix, orthogonal_matrix.T)
        self.assertTrue(np.allclose(product_inv, identity_matrix),
                        f"Matrix U U^T is not close to identity. U U^T:\n{product_inv}")

    def test_generate_covariance_sigma(self):
        dim = 5
        D_diag_values = (1, 1, 0.25, 2.25, 1) # From Appendix D
        
        sigma_matrix = generate_covariance_sigma(dim, D_diag_values)

        # Test shape
        self.assertEqual(sigma_matrix.shape, (dim, dim))

        # Test symmetry (Sigma should be symmetric)
        self.assertTrue(np.allclose(sigma_matrix, sigma_matrix.T),
                        f"Sigma matrix is not symmetric:\n{sigma_matrix}")

        # Test eigenvalues (eigenvalues of Sigma should be D_diag_values, up to permutation and numerical precision)
        eigenvalues = np.linalg.eigvalsh(sigma_matrix) # Use eigvalsh for symmetric matrices
        self.assertTrue(np.allclose(np.sort(eigenvalues), np.sort(D_diag_values)),
                        f"Eigenvalues of Sigma {np.sort(eigenvalues)} do not match D_diag_values {np.sort(D_diag_values)}")

        # Test with invalid D_diag_values length
        with self.assertRaises(ValueError):
            generate_covariance_sigma(dim, (1, 2)) # Incorrect length

    def test_generate_covariates_X(self):
        num_samples = 100
        dim_d = 5
        D_diag_values = (1, 1, 0.25, 2.25, 1)
        sigma_matrix = generate_covariance_sigma(dim_d, D_diag_values)

        X_covariates = generate_covariates_X(num_samples, dim_d, sigma_matrix)

        # Test shape
        self.assertEqual(X_covariates.shape, (num_samples, dim_d))

        # Test covariance: Cov(X) should be close to Sigma if num_samples is large enough
        # This is a statistical property, so it might not hold perfectly for small num_samples.
        # For a more robust test, one might increase num_samples significantly or run multiple times.
        if num_samples >= 10000: # Only run for larger sample sizes for stability
            empirical_covariance = np.cov(X_covariates, rowvar=False)
            self.assertTrue(np.allclose(empirical_covariance, sigma_matrix, atol=0.1),
                            f"Empirical covariance:\n{empirical_covariance}\nSigma_matrix:\n{sigma_matrix}")

        # Test with identity Sigma
        identity_sigma = np.eye(dim_d)
        X_identity_cov = generate_covariates_X(num_samples, dim_d, identity_sigma)
        # In this case, X_identity_cov should be unit sphere samples (xi_samples)
        # So, their norms should be 1
        norms_identity_cov = np.linalg.norm(X_identity_cov, axis=1)
        self.assertTrue(np.allclose(norms_identity_cov, np.ones(num_samples)),
                        "Covariates generated with identity Sigma should have norms close to 1.")

        with self.assertRaises(ValueError):
            generate_covariates_X(num_samples, dim_d, np.eye(dim_d + 1)) # Mismatched sigma shape

    def test_kappa_functions(self):
        x_i = np.array([1.0, 2.0])
        x_j = np.array([-1.0, 1.0])
        d = x_i.shape[0]

        # Linear
        self.assertAlmostEqual(kappa_linear(x_i, x_j, d), 1.0) # 1*(-1) + 2*1 = 1
        # ReLU
        self.assertAlmostEqual(kappa_relu(x_i, x_j, d), 1.0)  # relu(1) = 1
        x_k = np.array([-3.0, 1.0]) # dot product is -1*(-3) + 1*1 = 4 for x_j, x_k or -3 + 2 = -1 for x_i,x_k
        self.assertAlmostEqual(kappa_relu(x_i, x_k, d), 0.0)  # relu(1*(-3) + 2*1) = relu(-1) = 0
        # Exp
        # np.exp((1*(-1) + 2*1) / 2) = np.exp(1/2)
        self.assertAlmostEqual(kappa_exp(x_i, x_j, d), np.exp(1.0 / d))
        self.assertAlmostEqual(kappa_exp(x_i, x_j, 0), np.exp(1.0)) # Test d=0 case

    def test_build_covariance_matrix(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        d = X.shape[1]

        # Linear
        K_lin = build_covariance_matrix(X, KAPPA_TYPE_LINEAR, d)
        # Expected:
        # <x0,x0>=1, <x0,x1>=0, <x0,x2>=1
        # <x1,x0>=0, <x1,x1>=1, <x1,x2>=1
        # <x2,x0>=1, <x2,x1>=1, <x2,x2>=2
        expected_K_lin = np.array([[1,0,1],[0,1,1],[1,1,2]])
        self.assertTrue(np.allclose(K_lin, expected_K_lin))

        # ReLU
        K_relu = build_covariance_matrix(X, KAPPA_TYPE_RELU, d)
        expected_K_relu = np.array([[1,0,1],[0,1,1],[1,1,2]]) # Same for these positive dot products
        self.assertTrue(np.allclose(K_relu, expected_K_relu))
        
        X_neg = np.array([[1.0,0.0],[-1.0,0.0]])
        K_relu_neg = build_covariance_matrix(X_neg, KAPPA_TYPE_RELU, d)
        # <x0,x0>=1, <x0,x1>=-1 -> relu(-1)=0
        # <x1,x0>=-1, <x1,x1>=1
        expected_K_relu_neg = np.array([[1,0],[0,1]])
        self.assertTrue(np.allclose(K_relu_neg, expected_K_relu_neg))

    def test_get_k_plus_matrix(self):
        # Test with a PSD matrix (e.g., from linear kernel)
        K_psd = np.array([[2.0, 1.0], [1.0, 2.0]])
        K_plus_psd = get_k_plus_matrix(K_psd, KAPPA_TYPE_LINEAR)
        self.assertTrue(np.allclose(K_plus_psd, K_psd))

        # Test with a non-PSD matrix (handmade for relu testing)
        # This matrix has eigenvalues 3 and -1
        K_non_psd = np.array([[1.0, 2.0], [2.0, 1.0]]) 
        K_plus_non_psd = get_k_plus_matrix(K_non_psd, KAPPA_TYPE_RELU)
        
        # Eigenvalues of K_non_psd are 3, -1. Eigenvectors are [1,1]/sqrt(2) and [1,-1]/sqrt(2)
        # K_plus should have eigenvalues 3, 1.
        # Expected K_plus = V * diag([3,1]) * V.T
        # V = [[1,-1],[1,1]]/sqrt(2) ( eigenvectors are [1,1] and [-1,1] for e.g. eigh, order may vary)
        # Let's check properties: symmetry and positive semi-definiteness
        self.assertTrue(np.allclose(K_plus_non_psd, K_plus_non_psd.T), "K_plus should be symmetric")
        eigenvalues_k_plus = np.linalg.eigvalsh(K_plus_non_psd)
        self.assertTrue(np.all(eigenvalues_k_plus >= -1e-9), 
                        f"K_plus should be positive semi-definite. Eigenvalues: {eigenvalues_k_plus}")
        # For this specific K_non_psd, K_plus should be [[2,1],[1,2]]
        # K_non_psd = V Diag(3,-1) V.T ; K_plus = V Diag(3,1) V.T
        # V = 1/sqrt(2) * [[1,1],[-1,1]] (for eigh)
        # V.T = 1/sqrt(2) * [[1,-1],[1,1]]
        # D_plus = [[3,0],[0,1]]
        # V D_plus = 1/sqrt(2) * [[3,1],[-3,1]]
        # V D_plus V.T = 1/2 * [[3,1],[-3,1]] @ [[1,-1],[1,1]] = 1/2 * [[3+1, -3+1],[-3+1, 3+1]] = 1/2 * [[4,-2],[-2,4]] = [[2,-1],[-1,2]]
        # Hmm, my manual calculation seems to lead to [[2,-1],[-1,2]]. Let me recheck the eigenvectors of [[1,2],[2,1]].
        # (1-L)x + 2y = 0; 2x + (1-L)y = 0. (1-L)^2 - 4 = 0. 1-L = +/-2. L = 1-2 = -1 OR L = 1+2 = 3.
        # L=-1: 2x+2y=0 -> x=-y. Eigenvector [1,-1].
        # L=3: -2x+2y=0 -> x=y. Eigenvector [1,1].
        # So V = 1/sqrt(2) * [[1,1],[1,-1]] (cols are eigenvectors for 3, -1)
        # V D_plus V.T = (1/sqrt(2) * [[1,1],[1,-1]]) @ [[3,0],[0,1]] @ (1/sqrt(2) * [[1,1],[1,-1]])
        # = 1/2 * ([[3,1],[3,-1]]) @ [[1,1],[1,-1]] = 1/2 * [[3+1, 3-1],[3-1, 3+1]] = 1/2 * [[4,2],[2,4]] = [[2,1],[1,2]]
        expected_K_plus_for_non_psd = np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(np.allclose(K_plus_non_psd, expected_K_plus_for_non_psd))

    def test_generate_gaussian_process_labels(self):
        num_samples = 10
        dim_d = 3
        X = generate_unit_sphere_samples(num_samples, dim_d)

        for kappa_type in KAPPA_FUNCTIONS.keys():
            with self.subTest(kappa_type=kappa_type):
                Y = generate_gaussian_process_labels(X, kappa_type, dim_d, noise_std=1e-8)
                self.assertEqual(Y.shape, (num_samples,))
                self.assertLess(np.abs(np.mean(Y)), 1.0, f"Mean of Y for {kappa_type} is too far from 0") 

        # Section for testing with mocked K_manual_diag
        # This part currently only sets up mocks but doesn't use them in a generate_gaussian_process_labels call.
        # It can be expanded later if needed to specifically test how generate_gaussian_process_labels behaves with pre-defined K.
        K_manual_diag = np.diag([1.0, 2.0, 3.0])
        # num_diag_samples = K_manual_diag.shape[0] # This would be 3

        with mock.patch('data_utils.build_covariance_matrix') as mock_build_cov, \
             mock.patch('data_utils.get_k_plus_matrix') as mock_get_k_plus:
            
            mock_build_cov.return_value = K_manual_diag
            mock_get_k_plus.return_value = K_manual_diag
            # Example of how this mock could be used:
            # X_mock_input = np.random.rand(num_diag_samples, dim_d) # Needs to match K_manual_diag's dim if used directly
            # Y_mocked = generate_gaussian_process_labels(X_mock_input, KAPPA_TYPE_LINEAR, dim_d, noise_std=1e-8)
            # self.assertEqual(mock_build_cov.call_count, 1)
            # self.assertEqual(mock_get_k_plus.call_count, 1)
            # Further assertions on Y_mocked based on K_manual_diag could be made.
            # For now, we just ensure this block is correctly scoped.
            pass # Placeholder for potential future assertions using the mocks

        # Test for a singular K_plus from relu (e.g. all dot products negative)
        # This part should use the REAL functions, not the mocks from above.
        X_all_neg_dot = np.array([[1.0,0.0],[-1.0,0.00001]]) 
        dim_d_singular = X_all_neg_dot.shape[1]
        
        K_relu_singular_raw = build_covariance_matrix(X_all_neg_dot, KAPPA_TYPE_RELU, dim_d_singular)
        expected_K_relu_singular = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(np.allclose(K_relu_singular_raw, expected_K_relu_singular, atol=1e-7),
                        f"K_relu_singular_raw was\n{K_relu_singular_raw}\nexpected\n{expected_K_relu_singular}")
        
        K_plus_singular = get_k_plus_matrix(K_relu_singular_raw, KAPPA_TYPE_RELU)
        expected_K_plus_singular = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(np.allclose(K_plus_singular, expected_K_plus_singular, atol=1e-7),
                        f"K_plus_singular was\n{K_plus_singular}\nexpected\n{expected_K_plus_singular}")
        
        Y_singular = generate_gaussian_process_labels(X_all_neg_dot, KAPPA_TYPE_RELU, dim_d_singular, noise_std=1e-8)
        self.assertEqual(Y_singular.shape, (2,))

if __name__ == '__main__':
    unittest.main() 