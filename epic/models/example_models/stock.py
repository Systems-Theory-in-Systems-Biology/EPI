import jax.numpy as jnp
import numpy as np
import yfinance as yf

from epic.models.model import (
    ArtificialModelInterface,
    Model,
    VisualizationModelInterface,
)


class Stock(Model, VisualizationModelInterface):
    def getDataBounds(self):
        return np.array([[-7.5, 7.5] * self.dataDim])

    def getParamBounds(self):
        return np.array([[-2.0, 2.0] * self.paramDim])

    def getParamSamplingLimits(self):
        return np.array([[-10.0, 10.0] * self.paramDim])

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

    def downloadData(self, tickerList, tickerListName):
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

        stocks = np.loadtxt(tickerList, dtype="str")

        stockData = np.zeros((stocks.shape[0], dates.shape[0]))
        stockIDs = []

        successCounter = 0

        for i in range(stocks.shape[0]):
            try:
                df = yf.download(stocks[i], start, end, interval="1d")

                try:
                    for j in range(dates.shape[0]):
                        # extract the opening value (indicated by "[0]") of stock i at day j
                        stockData[successCounter, j] = df.loc[str(dates[j])][0]

                    # subtract initial value of complete timeline
                    stockData[successCounter, :] = (
                        stockData[successCounter, :]
                        - stockData[successCounter, 0]
                    )

                    if np.all(np.abs(stockData[successCounter, :]) < 100.0):
                        print("Success for ", stocks[i])
                        stockIDs.append(stocks[i])
                        successCounter += 1
                    else:
                        print("Values too large for ", stocks[i])

                except Exception as e:
                    print("Fail for ", stocks[i])
                    print(repr(e))
                    pass

            except Exception as e:
                print("Download Failed!")
                print(repr(e))

        # save all time points except for the first
        np.savetxt(
            "Data/DataOrigins/Stock/" + tickerListName + "Data.csv",
            stockData[0:successCounter, 1:],
            delimiter=",",
        )
        np.savetxt(
            "Data/DataOrigins/Stock/" + tickerListName + "IDs.csv",
            stockIDs,
            delimiter=",",
            fmt="% s",
        )

    def dataLoader(self, downloadData=False):
        if downloadData:
            # TODO: What is/are the tickerLists and names?
            self.downloadData()
        return super().dataLoader()

    def forward(self, param):
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
    def generateArtificialData(self):
        numSamples = 100000

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

        trueParamSample = np.random.randn(numSamples, 6)

        for i in range(6):
            trueParamSample[:, i] *= stdevs[i]

        artificialData = np.zeros((numSamples, 19))

        for j in range(numSamples):
            trueParamSample[j, :] += mean
            artificialData[j, :] = self.forward(trueParamSample[j, :])

        np.savetxt(
            "Data/StockArtificialData.csv", artificialData, delimiter=","
        )
        np.savetxt(
            "Data/StockArtificialParams.csv", trueParamSample, delimiter=","
        )

    def getParamSamplingLimits(self):
        return np.array([[-1.0, 3.0] * self.paramDim])
