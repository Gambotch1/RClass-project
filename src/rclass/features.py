import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA


def preprocess_data(X_train, X_test, n_components=15):
    """
    Standardize data (zero mean, unit variance) then apply PCA.
    Returns: X_train_pca, X_test_pca, scaler, pca
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, scaler, pca


def create_polynomial_features(X, degree):
    """
    Generate polynomial features (with bias) up to given degree.
    Returns: X_poly, poly_transformer
    """
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    return poly.fit_transform(X), poly
