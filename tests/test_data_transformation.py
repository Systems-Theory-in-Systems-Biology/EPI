import numpy as np

from eulerpi.core.data_transformations import (
    AffineTransformation,
    DataIdentity,
    DataNormalization,
    DataPCA,
)


def test_DataIdentity():
    """Test whether the DataIdentity transformation does not change the data."""
    data0dim = np.random.rand()
    data1dim = np.random.rand(100)
    data2dim = np.random.rand(100, 2)

    data_transformation = DataIdentity()

    assert np.allclose(data_transformation.transform(data0dim), data0dim)
    assert np.allclose(data_transformation.transform(data1dim), data1dim)
    assert np.allclose(data_transformation.transform(data2dim), data2dim)


def test_AffineTransformation():
    A1 = np.squeeze(np.array([[2]]))
    b1 = np.array([0.5])
    T1 = AffineTransformation(A1, b1)
    data1dim = np.random.rand(100)
    expected_result1 = np.inner(data1dim, A1) + b1
    assert np.allclose(T1.transform(data1dim), expected_result1)

    A2 = np.squeeze(np.array([[1, 2], [3, 4]]))
    b2 = np.array([0.5])
    T2 = AffineTransformation(A2, b2)
    data2dim = np.random.rand(100, 2)
    expected_result2 = np.inner(data2dim, A2) + b2
    assert np.allclose(T2.transform(data2dim), expected_result2)


def test_DataNormalization():
    """Test whether the DataNormalization transformation normalizes the data to zero mean and unit variance."""
    data1d = np.random.rand(100, 1)
    data2d = np.random.rand(100, 2)
    test_data = [(1, data1d), (2, data2d)]

    for dim, data in test_data:
        data_transformation = DataNormalization(data)
        transformed_data = data_transformation.transform(data)
        assert np.allclose(
            np.mean(transformed_data, axis=0),
            np.zeros_like(transformed_data[0]),
        )
        assert np.allclose(np.cov(transformed_data, rowvar=False), np.eye(dim))

        # Check if transform also works for single datapoints
        transformed_datapoint = data_transformation.transform(data[0])
        assert transformed_datapoint.shape == data[0].shape


def test_DataPCA():
    """Test whether the DataPCA transformation is able to run on data with different dimensions."""
    n_samples = 100
    data1 = np.random.rand(n_samples, 1)
    data2 = np.random.rand(n_samples, 2)
    data3 = np.random.rand(n_samples, 3)
    test_data = [(1, 1, data1), (2, 2, data2), (2, 1, data2), (3, 2, data3)]

    for data_dim, pca_dim, data in test_data:
        data_transformation = DataPCA(data=data, n_components=pca_dim)

        transformed_data = data_transformation.transform(data)
        assert transformed_data.shape == (n_samples, pca_dim)

        transformed_datapoint = data_transformation.transform(data[0])
        assert transformed_datapoint.shape == (pca_dim,)

        jacobian = data_transformation.jacobian(data[0])
        assert jacobian.shape == (pca_dim, data_dim)
