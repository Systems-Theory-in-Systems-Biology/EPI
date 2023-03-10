from typing import Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yfinance as yf

from epi import logger
from epi.core.model import ArtificialModelInterface, JaxModel

# Ticker source: https://investexcel.net/all-yahoo-finance-stock-tickers/#google_vignette, Date:27.10.2022
TICKERS = [
    "ETF",
    "Index1",
    "Index2",
    "Mutual",
    "Stocks1",
    "Stocks2",
    "Stocks3",
    "ETF50",  # First 50 tickers from ETF. Just for testing
]


class Stock(JaxModel):
    """Model simulating stock data."""

    data_dim = 19
    param_dim = 6

    PARAM_LIMITS = np.array([[-10.0, 10.0] for _ in range(param_dim)])
    CENTRAL_PARAM = np.array(
        [
            0.41406223,
            1.04680993,
            1.21173553,
            0.8078955,
            1.07772437,
            0.64869251,
        ]
    )

    def __init__(
        self,
        central_param: np.ndarray = CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(central_param, param_limits, name=name, **kwargs)

    def download_data(self, ticker_list_path: str):
        """Download stock data for a ticker list from yahoo finance.

        Args:
          ticker_list_path(str): path to the ticker list csv file

        Returns:
            stock_data, stock_ids and the name of the tickerList

        """
        logger.info("Downloading stock data...")
        start = "2022-01-31"
        end = "2022-03-01"

        stocks = np.loadtxt(ticker_list_path, dtype="str")

        try:
            df: pd.DataFrame = yf.download(
                stocks.tolist(), start, end, interval="1d", repair=True
            )
        except Exception as e:
            logger.warning("Download Failed!", exc_info=e)

        # drop all columns except for the open price
        df.drop(
            df.columns[df.columns.get_level_values(0) != "Open"],
            axis=1,
            inplace=True,
        )

        # remove columns with missing data
        missing = list(yf.shared._ERRORS.keys())
        df = df.loc[:, ~df.columns.get_level_values(1).isin(missing)]

        # subtract initial value of complete timeline, its simply subtracting the first row from the whole dataframe
        df = df.subtract(df.iloc[0], axis=1)

        # remove columns with extreme values
        df = df.loc[:, ~(df.abs() > 100).any()]

        # Drop the row of the first day
        df = df.iloc[1:, :]

        # get the remaining stock_ids and create a numpy array from the dataframe
        stock_ids = df.columns.get_level_values(1)  # .unique()
        stock_data = df.to_numpy()

        # get the name of the tickerList
        if type(ticker_list_path) != str:
            ticker_list_name = ticker_list_path.name.split(".")[0]
        else:
            ticker_list_name = ticker_list_path.split("/")[-1].split(".")[0]

        return stock_data.T, stock_ids, ticker_list_name

    @classmethod
    def forward(cls, param):
        def iteration(x, param):
            return jnp.array(
                [
                    param[0] / (1 + jnp.power(x[0], 2))
                    + param[1] * x[1]
                    + param[2],
                    param[3] * x[1] - param[4] * x[0] + param[5],
                ]
            )

        def repetition(x, param, num_repetitions):
            for i in range(num_repetitions):
                x = iteration(x, param)
            return x

        x = jnp.zeros(2)
        time_course = [
            repetition(x, param, n)
            for n in [1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 4, 1, 1, 1, 3]
        ]
        return jnp.array([x[0] for x in time_course])


class StockArtificial(Stock, ArtificialModelInterface):
    """ """

    PARAM_LIMITS = np.array([[-1.0, 3.0] for _ in range(Stock.param_dim)])

    def __init__(
        self,
        central_param: np.ndarray = Stock.CENTRAL_PARAM,
        param_limits: np.ndarray = PARAM_LIMITS,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super(Stock, self).__init__(
            central_param, param_limits, name=name, **kwargs
        )

    def generate_artificial_params(self, num_samples):
        mean = np.array(
            [
                0.41406223,
                1.04680993,
                1.21173553,
                0.8078955,
                1.07772437,
                0.64869251,
            ]
        )
        stdevs = np.array([0.005, 0.01, 0.05, 0.005, 0.01, 0.05])

        true_param_sample = np.random.randn(num_samples, self.param_dim)
        true_param_sample *= stdevs
        true_param_sample += mean

        return true_param_sample
