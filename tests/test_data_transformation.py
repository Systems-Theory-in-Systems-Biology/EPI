import numpy as np

from eulerpi.core.data_transformation import (
    DataIdentity,
    DataNormalizer,
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


def test_DataNormalizer():
    """Test whether the DataNormalizer transformation normalizes the data to zero mean and unit variance."""
    data1d = np.random.rand(100, 1)
    data2d = np.random.rand(100, 2)
    test_data = [(1, data1d), (2, data2d)]

    for dim, data in test_data:
        data_transformation = DataNormalizer.from_data(data)
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
    data1d = np.random.rand(100, 1)
    data2d = np.random.rand(100, 2)
    test_data = [(1, data1d), (2, data2d)]

    for dim, data in test_data:
        data_transformation = DataPCA.from_data(data)
        transformed_data = data_transformation.transform(data)
