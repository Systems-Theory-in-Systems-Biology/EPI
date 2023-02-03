import importlib
import os

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yfinance as yf
from jax import vmap

from epi import logger
from epi.core.model import (
    ArtificialModelInterface,
    JaxModel,
    VisualizationModelInterface,
)

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


class Stock(JaxModel, VisualizationModelInterface):
    """Model simulating stock data."""

    DataDim = 19
    ParamDim = 6

    def __init__(
        self, delete: bool = False, create: bool = True, ticker="ETF50"
    ) -> None:
        """Initialize the model and set a ticker. Can be chosen from the list of available tickers TICKERS.
        Possibly outdated list: [ETF, Index1, Index2, Mutual, Stocks1, Stocks2, Stocks3]

        :param ticker: The ticker from which the data should be used, defaults to "ETF"
        :type ticker: str, optional
        """
        super().__init__(delete, create)
        self.dataPath = f"Data/{self.getModelName()}/{ticker}Data.csv"

        # Check if data for the given ticker exists
        if not os.path.isfile(self.dataPath):
            logger.warning("Ticker data not found. Downloading data...")
            tickerPath = importlib.resources.path(
                "epi.examples.stock", f"{ticker}.csv"
            )
            self.downloadData(tickerPath)

    def getDataBounds(self):
        return np.array([[-7.5, 7.5] * self.DataDim])

    def getParamBounds(self):
        return np.array([[-2.0, 2.0] * self.ParamDim])

    def getParamSamplingLimits(self):
        return np.array([[-10.0, 10.0] * self.ParamDim])

    def getCentralParam(self):
        return np.array(
            [
                0.41406223,
                1.04680993,
                1.21173553,
                0.8078955,
                1.07772437,
                0.64869251,
            ]
        )

    def downloadData(self, tickerListPath: str):
        """Download stock data for a ticker list from yahoo finance.

        :param tickerListPath: path to the ticker list csv file
        :type tickerListPath: str
        """
        logger.info("Downloading stock data...")
        start = "2022-01-31"
        end = "2022-03-01"
        dates = np.array(
            [
                "2022-01-31",
                "2022-02-01",
                "2022-02-02",
                "2022-02-03",
                "2022-02-04",
                "2022-02-07",
                "2022-02-08",
                "2022-02-09",
                "2022-02-10",
                "2022-02-11",
                "2022-02-14",
                "2022-02-15",
                "2022-02-16",
                "2022-02-17",
                "2022-02-18",
                "2022-02-22",
                "2022-02-23",
                "2022-02-24",
                "2022-02-25",
                "2022-02-28",
            ]
        )

        stocks = np.loadtxt(tickerListPath, dtype="str")

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

        # get the remaining stockIDs and create a numpy array from the dataframe
        stockIDs = df.columns.get_level_values(1)  # .unique()
        stockData = df.to_numpy()

        # get the name of the tickerList
        if type(tickerListPath) != str:
            tickerListName = tickerListPath.name.split(".")[0]
        else:
            tickerListName = tickerListPath.split("/")[-1].split(".")[0]

        # save all time points except for the first
        os.makedirs(f"Data/{self.getModelName()}", exist_ok=True)
        np.savetxt(
            f"Data/{self.getModelName()}/{tickerListName}Data.csv",
            stockData.T,
            delimiter=",",
        )
        np.savetxt(
            f"Data/{self.getModelName()}/{tickerListName}IDs.csv",
            stockIDs,
            delimiter=",",
            fmt="% s",
        )

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

        def repetition(x, param, numRepetitions):
            for i in range(numRepetitions):
                x = iteration(x, param)

            return x

        x0 = jnp.zeros(2)
        x1 = repetition(x0, param, 1)
        x2 = repetition(x1, param, 1)
        x3 = repetition(x2, param, 1)
        x4 = repetition(x3, param, 1)
        x5 = repetition(x4, param, 3)
        x6 = repetition(x5, param, 1)
        x7 = repetition(x6, param, 1)
        x8 = repetition(x7, param, 1)
        x9 = repetition(x8, param, 1)
        x10 = repetition(x9, param, 3)
        x11 = repetition(x10, param, 1)
        x12 = repetition(x11, param, 1)
        x13 = repetition(x12, param, 1)
        x14 = repetition(x13, param, 1)
        x15 = repetition(x14, param, 4)
        x16 = repetition(x15, param, 1)
        x17 = repetition(x16, param, 1)
        x18 = repetition(x17, param, 1)
        x19 = repetition(x18, param, 3)

        timeCourse = jnp.array(
            [
                x1,
                x2,
                x3,
                x4,
                x5,
                x6,
                x7,
                x8,
                x9,
                x10,
                x11,
                x12,
                x13,
                x14,
                x15,
                x16,
                x17,
                x18,
                x19,
            ]
        )

        return timeCourse[:, 0]


class StockArtificial(Stock, ArtificialModelInterface):
    def __init__(self, *args, **kwargs):
        super(Stock, self).__init__(*args, **kwargs)

    def generateArtificialData(
        self, numSamples=ArtificialModelInterface.NUM_ARTIFICIAL_SAMPLES
    ):
        logger.info(
            f"Generating {numSamples} data samples by evaluating the model. "
            "This might take a very long time!"
        )

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

        trueParamSample = np.random.randn(numSamples, self.ParamDim)
        trueParamSample *= stdevs
        trueParamSample += mean

        artificialData = vmap(self.forward, in_axes=0)(trueParamSample)

        np.savetxt(
            f"Data/{self.getModelName()}Data.csv",
            artificialData,
            delimiter=",",
        )

        np.savetxt(
            f"Data/{self.getModelName()}Params.csv",
            trueParamSample,
            delimiter=",",
        )

    def getParamSamplingLimits(self):
        return np.array([[-1.0, 3.0] * self.ParamDim])
