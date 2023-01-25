from epi.core.model import Model
from epi.core.sampling import inference
from epi.examples.corona import CoronaArtificial

model: Model = CoronaArtificial()

# generate artificial data
if model.isArtificial():
    model.generateArtificialData()

# run MCMC sampling for EPI
numWalkers = 12
inference(model=model, numWalkers=numWalkers)
