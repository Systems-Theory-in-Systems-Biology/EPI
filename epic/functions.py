from epic.models import *

### This file contains all computing functions for EPI

def evalKDECauchy(data, simRes, scales):
    """ Evaluates a Cauchy Kernel Density estimator in one simulation result.
        Assumes that each data point is a potentially high-dimensional sample
        from a joint data distribution.
        This is for example given for time-series data, where each evaluation
        time is one dimension of the data point.

    Input: data (data for the model: 2D array with shape (#Samples, #MeasurementDimensions))
           simRes (evaluation coordinates array with one entry for each data dimension)
           scales (one scale for each dimension)
           
    Output: densityEvaluation (estimated kernel density evaluated at the simulation result)
    """
    # This quantity will store the probability density.
    evaluation = 0
    
    # Loop over each measurement sample.
    for s in range(data.shape[0]):

        # Construct a Cauchy-ditribution centered around the data point and evaluate it in the simulation result.
        evaluation += np.prod(1/((np.power((simRes-data[s,:])/scales,2)+1)*scales*np.pi))
        
    # Return the average of all Cauchy distribution evaluations to eventually obtain a probability density again.
    return evaluation/data.shape[0]


def evalKDEGauss(data, simRes, stdevs):
    """ Evaluates a Gaussian Kernel Density estimator in one simulation result.
        Assumes that each data point is a potentially high-dimensional sample from a joint data distribution.
        This is for example given for time-series data, where each evaluation time is one dimension of the data point.
        While it is possible to define different standard deviations for different measurement dimensions, it is so far not possible to define covariances.

    Input: data (data for the model: 2D array with shape (#Samples, #MeasurementDimensions))
           simRes (evaluation coordinates array with one entry for each data dimension)
           stdevs (one standard deviation for each dimension)
           
    Output: densityEvaluation (estimated kernel density evaluated at the simulation result)
    """
    # This quantity will store the probability density
    evaluation = 0
    
    # Loop over each measurement sample
    for s in range(data.shape[0]):
        
        # Construct a Cauchy-ditribution centered around the data point and evaluate it in the simulation result.
        diff = simRes-data[s,:]
        mult = -np.sum(diff*diff/stdevs/stdevs)/2.0
        
        evaluation += np.exp(mult)/np.sqrt(np.power(2*np.pi,simRes.shape[0])*np.power(np.prod(stdevs),2))
    
    # Return the average of all Gauss distribution evaluations to eventually obtain a probability density again.
    return evaluation/data.shape[0]


def calcCorrection(modelJac, param):
    """ Evaluate the pseudo-determinant of the simulation jacobian (that serves as a correction term) in one specific parameter point.


    Input: modelJac (algorithmic differentiation object for Jacobian of the sim model)
           param (parameter at which the simulation model is evaluated)
    Output: correction (correction factor for density transformation)
    """
    
    # Evaluate the algorithmic differentiation object in the parameter
    jac = modelJac(param)
    jacT = np.transpose(jac)
    
    # The pseudo-determinant is calculated as the square root of the determinant of the matrix-product of the Jacobian and its transpose.
    # For numerical reasons, one can regularize the matrix product by adding a diagonal matrix of ones before calculating the determinant.
    #correction = np.sqrt(np.linalg.det(np.matmul(jacT,jac) + np.eye(param.shape[0])))
    correction = np.sqrt(np.linalg.det(np.matmul(jacT,jac)))
    
    # If the correction factor is not a number or infinite, return 0 instead to not affect the sampling.
    if (math.isnan(correction) or math.isinf(correction)):
        correction = 0.0
        print("invalid value encountered for correction factor")
    
    return correction
    

def evalLogTransformedDensity(param, modelName, data, dataStdevs):
    """ Given a simulation model, its derivative and corresponding data, evaluate the natural log of the parameter density that is the backtransformed data distribution.
        This function is intended to be used with the emcee sampler and can be implemented more efficiently at some points.
        
    Input: param (parameter for which the transformed density shall be evaluated)
           modelName (model ID)
           data (data for the model: 2D array with shape (#numDataPoints, #dataDim))
           dataStdevs (array of suitable kernel standard deviations for each data dimension)
    Output: logTransformedDensity (natural log of parameter density at the point param)
          : allRes (array concatenation of parameters, simulation results and evaluated density, stored as "blob" by the emcee sampler)
    """
    
    # Define model-specific lower...
    paramsLowerLimitsDict = {
    "Linear": np.array([-10.0,-10.0]),
    "LinearODE": np.array([-10.0,-10.0]),
    "Temperature": np.array([0]),
    "TemperatureArtificial": np.array([0]),
    "Corona": np.array([-4.5,-2.0,-2.0]),
    "CoronaArtificial": np.array([-2.5,-0.75,0.0]),
    "Stock": np.array([-10.0,-10.0,-10.0,-10.0,-10.0,-10.0]),
    "StockArtificial": np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])}
    
    # ... and upper borders for sampling to avoid parameter regions where the simulation can only be evaluated instably.
    paramsUpperLimitsDict = {
    "Linear": np.array([11.0, 11.0]),
    "LinearODE": np.array([23.0,23.0]),
    "Temperature": np.array([np.pi/2]),
    "TemperatureArtificial": np.array([np.pi/2]),
    "Corona": np.array([0.5,3.0,3.0]),
    "CoronaArtificial": np.array([-1.0,0.75,1.5]),
    "Stock": np.array([10.0,10.0,10.0,10.0,10.0,10.0]),
    "StockArtificial": np.array([3.0,3.0,3.0,3.0,3.0,3.0])}
    
    # Check if the tried parameter is within the just-defined bounds and return the lowest possible log density if not.
    if ((np.any(param < paramsLowerLimitsDict[modelName])) or (np.any(param > paramsUpperLimitsDict[modelName]))):
        print("parameters outside of predefines range")
        return -np.inf, np.zeros(param.shape[0]+data.shape[1]+1)
    
    # If the parameter is within the valid ranges...
    else:
        # ... load the model and its Jacobian. One could also hand them over as function arguements. However, emcee requires all arguements to be pickable and jax objects violate this condition.
        model, modelJac = modelLoader(modelName)

        # Evaluate the simulation result for the specified parameter.
        simRes = model(param)  
        
        # Evaluate the data density in the simulation result.
        densityEvaluation = evalKDEGauss(data, simRes, dataStdevs)    
        
        # Calculate the simulation model's pseudo-determinant in the parameter point (also called the correction factor).
        correction = calcCorrection(modelJac, param)    
        
        # Multiply data density and correction factor. 
        trafoDensityEvaluation = densityEvaluation*correction  
        
        # Use the log of the transformed density because emcee requires this.
        logTransformedDensity = np.log(trafoDensityEvaluation)

        # Store the current parameter, its simulation result as well as its density in a large vector that is stored separately by emcee.
        allRes = np.concatenate((param, simRes, np.array([trafoDensityEvaluation])))

        return logTransformedDensity, allRes

def countEmceeSubRuns(modelName):
    
    """ This data organization function counts how many sub runs are saved for the specified scenario.
    
    Input: modelName (model ID)
    Output: numExistingFiles (number of completed sub runs of the emcee particle swarm sampler)
    """
    # Initialize the number of existing files to be 0
    numExistingFiles = 0

    # Increase the just defined number until no corresponding file is found anymore ...
    while(path.isfile("Applications/" + modelName + "/DensityEvals/" + str(numExistingFiles) + ".csv")):
        numExistingFiles += 1
    
    return numExistingFiles


def runEmceeSampling(modelName, numRuns, numWalkers, numSteps, numProcesses):
    
    """ Create a representative sample from the transformed parameter density using the emcee particle swarm sampler.
        Inital values are not stored in the chain and each file contains <numSteps> blocks of size numWalkers.

    Input: modelName (model ID)
           numRuns (number of stored sub runs)
           numWalkers (number of particles in the particle swarm sampler)
           numSteps (number of samples each particle performs before storing the sub run)
           numProcesses (number of parallel threads)
    Output: <none except for stored files>
    """

    # Load data, data standard deviations and model characteristics for the specified model.
    paramDim, dataDim, numDataPoints, centralParam, data, dataStdevs = dataLoader(modelName)

    # Initialize each walker at a Gaussian-drawn random, slightly different parameter close to the central parameter.
    walkerInitParams = [centralParam + 0.002*(np.random.rand(paramDim)-0.5) for i in range(numWalkers)]

    # Count and print how many runs have already been performed for this model
    numExistingFiles = countEmceeSubRuns(modelName)
    print(numExistingFiles, " existing files found")

    # Loop over the remaining sub runs and contiune the counter where it ended.
    for run in range(numExistingFiles, numExistingFiles+numRuns):
        print("Run ", run)

        # If there are current walker positions defined by runs before this one, use them.
        if (path.isfile("Applications/" + modelName + "/currentPos.csv")):
            walkerInitParams = np.loadtxt("Applications/" + modelName + "/currentPos.csv", delimiter = ",")
            print("continue sampling")

        else:
            print("start sampling")

        # Create a pool of worker processes.
        pool = Pool(processes = numProcesses)
        
        # define a custom move policy
        #movePolicy = [(emcee.moves.WalkMove(), 0.8), (emcee.moves.StretchMove(), 0.2)]
        #movePolicy = [(emcee.moves.KDEMove(), 1.0)]
        movePolicy = [(emcee.moves.WalkMove(), 0.1), (emcee.moves.StretchMove(), 0.1), (emcee.moves.GaussianMove(0.00001, mode='sequential', factor=None), 0.8)]
        #movePolicy = [(emcee.moves.GaussianMove(0.00001, mode='sequential', factor=None), 1.0)]
        
        # Call the sampler for all parallel workers (possibly use arg moves = movePolicy)
        sampler = emcee.EnsembleSampler(numWalkers, paramDim, evalLogTransformedDensity, pool=pool, moves = movePolicy, args=[modelName, data, dataStdevs])
        
        # Extract the final walker position and close the pool of worker processes.
        finalPos, _, _, _ = sampler.run_mcmc(walkerInitParams, numSteps, tune=True, progress=True)
        pool.close()
        pool.join()

        # Save the current walker positions as initial values for the next run.
        np.savetxt("Applications/" + modelName + "/currentPos.csv", finalPos, delimiter = ",")

        # Create a large container for all sampling results (sampled parameters, corresponding simulation results and parameter densities) and fill it using the emcee blob option.
        allRes = np.zeros((numWalkers*numSteps, paramDim+dataDim+1))

        for i in range(numSteps):
            for j in range(numWalkers):
                allRes[i*numWalkers+j,:] = sampler.blobs[i][j]

        # Save all sampling results in .csv files.        
        np.savetxt("Applications/" + modelName + "/Params/" + str(run) + ".csv", allRes[:,0:paramDim], delimiter = ",")
        np.savetxt("Applications/" + modelName + "/SimResults/" + str(run) + ".csv", allRes[:,paramDim:paramDim+dataDim], delimiter = ",")
        np.savetxt("Applications/" + modelName + "/DensityEvals/" + str(run) + ".csv", allRes[:,-1], delimiter = ",")

        # Print the sampling acceptance ratio.
        print("acceptance fractions:")
        print(np.round(sampler.acceptance_fraction,2))
        
        # Print the autocorrelation time (produces a so-far untreated runtime error if chains are too short)
        #print("autocorrelation time:")
        #print(sampler.get_autocorr_time()[0])

    return 0


def concatenateEmceeSamplingResults(modelName):
    """ Concatenate many sub runs of the emcee sampler to create 3 large files for sampled parameters, corresponding simulation results and density evaluations. 
        These files are later used for result visualization.
        
    Input: modelName (model ID)
    Output: <none except for stored files>
    """  
    
    # Load data, data standard deviations and model characteristics for the specified model.
    paramDim, dataDim, numDataPoints, centralParam, data, dataStdevs = dataLoader(modelName)
    
    # Count and print how many sub runs are ready to be merged.
    numExistingFiles = countEmceeSubRuns(modelName)
    print(numExistingFiles, " existing files found")
    
    # Load one example file and use it to extract how many samples are stored per file.
    numSamplesPerFile = np.loadtxt("Applications/" + modelName + "/Params/0.csv", delimiter = ",").shape[0]   

    # The overall number of sampled is the number of sub runs multiplied with the number of samples per file.
    numSamples = numExistingFiles*numSamplesPerFile

    # Create containers large enough to store all sampling information.
    overallDensityEvals = np.zeros(numSamples)
    overallSimResults = np.zeros((numSamples, dataDim))
    overallParams = np.zeros((numSamples, paramDim))

    # Loop over all sub runs, load the respective sample files and store them at their respective places in the overall containers.
    for i in range(numExistingFiles):

        overallDensityEvals[i*numSamplesPerFile:(i+1)*numSamplesPerFile] = np.loadtxt("Applications/" + modelName + "/DensityEvals/" + str(i) + ".csv", delimiter = ",")
        overallSimResults[i*numSamplesPerFile:(i+1)*numSamplesPerFile,:] = np.loadtxt("Applications/" + modelName + "/SimResults/" + str(i) + ".csv", delimiter = ",")
        overallParams[i*numSamplesPerFile:(i+1)*numSamplesPerFile,:] = np.loadtxt("Applications/" + modelName + "/Params/" + str(i) + ".csv", delimiter = ",")

    # Save the three just-created files.
    np.savetxt("Applications/" + modelName + "/OverallDensityEvals.csv", overallDensityEvals, delimiter = ",")
    np.savetxt("Applications/" + modelName + "/OverallSimResults.csv", overallSimResults, delimiter = ",")
    np.savetxt("Applications/" + modelName + "/OverallParams.csv", overallParams, delimiter = ",")
    
    return 0

def calcDataMarginals(modelName, resolution):
    """ Evaluate the one-dimensional marginals of the original data over equi-distant grids.
        The stores evaluations can then be used for result visualization.
        
    Input: modelName (model ID)
           resolution (defines the number of grid points for each marginal evaluation is directly proportional to the runtime)
    Output: <none except for stored files>

    Standard parameters : resolution = 100
    """
    
    # Load data, data standard deviations and model characteristics for the specified model.
    paramDim, dataDim, numDataPoints, centralParam, data, dataStdevs = dataLoader(modelName)
    
    # Create containers for the data marginal evaluations.
    trueDataMarginals = np.zeros((resolution, dataDim))
    
    # Load the grid over which the data marginal will be evaluated
    dataGrid, _ = returnVisualizationGrid(modelName, resolution)

    # Loop over each simulation result dimension and marginalize over the rest.
    for l in range(dataDim):
    
        # The 1D-arrays of true data have to be casted to 2D-arrays, as this format is obligatory for kernel density estimation.
        marginalData = np.zeros((data.shape[0],1))
        marginalData[:,0] = data[:,l]

        # Loop over all grid points and evaluate the 1D kernel marginal density estimation of the data sample.
        for i in range(resolution):        
            trueDataMarginals[i,l] = evalKDEGauss(marginalData, np.array([dataGrid[i,l]]), np.array([dataStdevs[l]]))
        
    # Store the marginal KDE approximation of the data
    np.savetxt("Applications/" + modelName + "/Plots/trueDataMarginals.csv", trueDataMarginals, delimiter = ",")
    
    return 0
    

def calcEmceeSimResultsMarginals(modelName, numBurnSamples, occurrence, resolution):
    """ Evaluate the one-dimensional marginals of the emcee sampling simulation results over equi-distant grids.
        The stores evaluations can then be used for result visualization.
        
    Input: modelName (model ID)
           numBurnSamples (Number of ignored first samples of each chain)
           occurence (step of sampling from chains)
           resolution (defines the number of grid points for each marginal evaluation is directly proportional to the runtime)
    Output: <none except for stored files>

    Standard parameters : numBurnSamples = 20% of all samples 
                          occurence = numWalkers+1 (ensures that the chosen samples are nearly uncorrelated)
                          resolution = 100
    """
    
    # Load the emcee simulation results chain
    simResults = np.loadtxt("Applications/" + modelName + "/OverallSimResults.csv", delimiter = ",")[numBurnSamples::occurrence,:]
    
    # Load data, data standard deviations and model characteristics for the specified model.
    paramDim, dataDim, numDataPoints, centralParam, data, dataStdevs = dataLoader(modelName)
    
    # Create containers for the simulation results marginal evaluations.
    inferredDataMarginals = np.zeros((resolution, dataDim))
    
    # Load the grid over which the simulation results marginal will be evaluated
    dataGrid, _ = returnVisualizationGrid(modelName, resolution)
    
    # Loop over each data dimension and marginalize over the rest.
    for l in range(dataDim):
     
        # The 1D-arrays of simulation resultshave to be casted to 2D-arrays, as this format is obligatory for kernel density estimation.
        marginalSimResults = np.zeros((simResults.shape[0],1))
        marginalSimResults[:,0] = simResults[:,l]
           
        # Loop over all grid points and evaluate the 1D kernel marginal density estimation of the emcee simulation results.
        for i in range(resolution):        
            inferredDataMarginals[i,l] = evalKDEGauss(marginalSimResults, np.array([dataGrid[i,l]]), np.array([dataStdevs[l]]))
            
    # Store the marginal KDE approximation of the simulation results emcee sample
    np.savetxt("Applications/" + modelName + "/Plots/inferredDataMarginals.csv", inferredDataMarginals,  delimiter = ",")
    
    return 0

    
def calcParamMarginals(modelName, numBurnSamples, occurrence, resolution):
    """ Evaluate the one-dimensional marginals of the emcee sampling parameters (and potentially true parameters) over equi-distant grids.
        The stores evaluations can then be used for result visualization.
        
    Input: modelName (model ID)
           numBurnSamples (Number of ignored first samples of each chain)
           occurence (step of sampling from chains)
           resolution (defines the number of grid points for each marginal evaluation is directly proportional to the runtime)
    Output: <none except for stored files>

    Standard parameters : numBurnSamples = 20% of all samples 
                          occurence = numWalkers+1 (ensures that the chosen samples are nearly uncorrelated)
                          resolution = 100
    """
    
    # By default, we assume that no true parameter information is available
    artificialBool = 0
    
    # If the model name indicates an artificial setting, indicate that true parameter information is available
    if ((modelName == "TemperatureArtificial") or (modelName == "CoronaArtificial") or (modelName == "StockArtificial")):
        artificialBool = 1
    
    # Load the emcee parameter chain
    paramChain = np.loadtxt("Applications/" + modelName + "/OverallParams.csv", delimiter = ",")[numBurnSamples::occurrence,:]
    
    # Load data, data standard deviations and model characteristics for the specified model.
    paramDim, dataDim, numDataPoints, centralParam, data, dataStdevs = dataLoader(modelName)
    
    # Define the standard deviation for plotting the parameters based on the sampled parameters and not the true ones.
    paramStdevs = calcStdevs(paramChain)
    
    # Create containers for the parameter marginal evaluations and the underlying grid.
    _, paramGrid = returnVisualizationGrid(modelName, resolution)
    inferredParamMarginals = np.zeros((resolution, paramDim))
    
    # If there are true parameter values available, load them and allocate storage similar to the just-defined one.
    if (artificialBool == 1):
        trueParamSample, _ = paramLoader(modelName)
        trueParamMarginals = np.zeros((resolution, paramDim))   
    
    # Loop over each parameter dimension and marginalize over the rest.
    for l in range(paramDim):
        
        # As the kernel density estimators only work for 2D-arrays of data, we have to cast the column of parameter samples into a 1-column matrix (or 2D-array).
        marginalParamChain = np.zeros((paramChain.shape[0],1))
        marginalParamChain[:,0] = paramChain[:,l]
        
        # If there is true parameter information available, we have to do the same type cast for the true parameter samples.
        if (artificialBool == 1):
            trueMarginalParamSample = np.zeros((trueParamSample.shape[0],1))
            trueMarginalParamSample[:,0] = trueParamSample[:,l]
        
        # Loop over all grid points and evaluate the 1D kernel density estimation of the reconstructed marginal parameter distribution.
        for i in range(resolution):        
            inferredParamMarginals[i,l] = evalKDEGauss(marginalParamChain, np.array([paramGrid[i,l]]), np.array(paramStdevs[l]))

            # If true parameter information is available, evaluate a similat 1D marginal distribution based on the true parameter samples.
            if (artificialBool == 1):
                trueParamMarginals[i,l] = evalKDEGauss(trueMarginalParamSample, np.array([paramGrid[i,l]]), np.array(paramStdevs[l]))

                
    # Store the (potentially 2) marginal distribution(s) for later plotting
    np.savetxt("Applications/" + modelName + "/Plots/inferredParamMarginals.csv", inferredParamMarginals,  delimiter = ",")
    
    if (artificialBool == 1):
        np.savetxt("Applications/" + modelName + "/Plots/trueParamMarginals.csv", trueParamMarginals, delimiter = ",")
        
    return 0
    
def calcWalkerAcceptance(modelName, numBurnSamples, numWalkers):

    """ Calculate the acceptance ratio for each individual walker of the emcee chain.
        This is especially important to find "zombie" walkers, that are never movig.
        
    Input: modelName (model ID)
           numBurnSamples (integer number of ignored first samples of each chain)
           numWalkers (integer number of emcee walkers)
           
    Output: acceptanceRatios (np.array of size numWalkers)
    """
    
    # load the emcee parameter chain
    params = np.loadtxt("Applications/" + modelName + "/OverallParams.csv", delimiter = ",")[numBurnSamples:,:]
    
    # calculate the number of steps each walker walked
    # subtract 1 because we count the steps between the parameters
    numSteps = int(params.shape[0]/numWalkers) - 1
    print("Number of steps fo each walker = ", numSteps)
    
    # create storage to count the number of accepted steps for each counter
    numAcceptedSteps = np.zeros(numWalkers)
    
    for i in range(numWalkers):
        for j in range(numSteps):
            numAcceptedSteps[i] += 1 - np.all(params[i + j*numWalkers,:] == params[i + (j+1)*numWalkers,:])
            
    # calculate the acceptance ratio by dividing the number of accepted steps by the overall number of steps
    acceptanceRatios = numAcceptedSteps/numSteps
    
    return acceptanceRatios