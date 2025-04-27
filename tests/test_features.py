# tests/test_features.py

import numpy as np
from rclass.features import preprocess_data, create_polynomial_features

def test_preprocess_and_pca():
    # small random data
    X_train = np.random.rand(50, 784)
    X_test  = np.random.rand(10, 784)
    n_comp  = 5

    X_tr_pca, X_te_pca, scaler, pca = preprocess_data(X_train, X_test, n_components=n_comp)
    # Shapes should match and PCA dims
    assert X_tr_pca.shape == (50, n_comp)
    assert X_te_pca.shape == (10, n_comp)

    # PCA components stored
    assert pca.n_components_ == n_comp

def test_polynomial_features():
    # 2D inputs, degree=2 gives 6 features: [1, x1, x2, x1^2, x1*x2, x2^2]
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_poly, poly = create_polynomial_features(X, degree=2)
    # Expect shape (2 samples, 6 features)
    assert X_poly.shape == (2, 6)
    # Check first row manually
    # [1, 1,2, 1^2,1*2,2^2] => [1,1,2,1,2,4]
    np.testing.assert_array_almost_equal(X_poly[0], [1,1,2,1,2,4])
