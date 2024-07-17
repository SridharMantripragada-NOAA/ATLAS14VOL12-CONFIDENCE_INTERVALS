from pathlib import Path
import pandas as pd
import numpy as np
import math
import random
from scipy.stats import norm
import scipy.stats as stats
import itertools
import json
from joblib import Parallel, delayed
from lmoments3 import distr

def load_config(file_path):
    '''
    Load the configuration from the given file path.
    
    :param file_path: The path to the configuration file.
    :return: A dictionary containing the configuration.
    :raises FileNotFoundError: If the configuration file does not exist.
    '''
    if not file_path.is_file():
        raise FileNotFoundError(f'Configuration file {file_path} does not exist.\
        Please create it based on the provided template.')
    
    with open(file_path, 'r') as file:
        return json.load(file)


def extract_regions(region_file):
    """
    Reads regionalization station information from a file and returns reference 
    stations and its dictionary of stations in its region.

    Parameters:
    - region_file (str or Path): Path to the file with region data.

    Returns:
    - tuple: (list of reference station ids, 
    dictionary {reference station id: list of its stations in a region})
    """
    
    regions_table = {}
    with open(Path(region_file), 'r') as file:
        for line in file:
            parts = line.strip().split()
            key = parts[0]
            values = parts[1:]
            regions_table[key] = values
    sids = list(regions_table.keys())
    
    return sids, regions_table

    
def extract_coloc(coloc_file):
    """
    Extract data from a fixed-width formatted file into a pandas DataFrame.

    This function reads a file where columns are of fixed widths, specified by
    the 'widths' list. The extracted data is then returned as a pandas DataFrame.

    Args:
    coloc_file (str): The path to the fixed-width file to be read.

    Returns:
    DataFrame: A pandas DataFrame containing the data from the fixed-width file.
    """

    # Define the width of each column in the fixed-width file
    widths = [3, 31, 8, 9, 10, 6, 8, 8, 5, 4, 8, 4, 4, 8, 4, 8, 4, 8, 4, 2, 2, 6, 7, 5, 11]

    # Read the fixed-width file using the defined column widths
    coloc = pd.read_fwf(coloc_file, widths=widths)
    
    # Return the DataFrame containing the parsed data
    return coloc



def legendre_shift_poly(n):
    """
    Returns the coefficients of the shifted Legendre polynomial of degree n.

    The shifted Legendre polynomials are orthogonal on the interval [0, 1]. 
    This function computes the coefficients of the polynomial using a recursive 
    relation adapted to the shifted interval.

    Parameters:
    n (int): The degree of the polynomial.

    Returns:
    np.array: Coefficients of the polynomial. The m-th element is the coefficient
              of x^(n+1-m), arranged from the highest degree term to the lowest.
    """
    # Base case for n = 0
    if n == 0:
        pk = np.array([1])
    # Base case for n = 1
    elif n == 1:
        pk = np.array([2, -1])
    else:
        # Initialize polynomials for k-2 and k-1
        pkm2 = np.zeros(n + 1)
        pkm2[n] = 1  # P0(x)
        pkm1 = np.zeros(n + 1)
        pkm1[n] = -1  # Coefficients of P1(x)
        pkm1[n - 1] = 2
        
        # Recursive calculation for k = 2 to n
        for k in range(2, n + 1):
            pk = np.zeros(n + 1)  # Polynomial for current degree k
            # Update polynomial coefficients using the recurrence relation
            for i in range(n - k, n):
                pk[i] = ((4 * k - 2) * pkm1[i + 1] + (1 - 2 * k) * pkm1[i] +
                         (1 - k) * pkm2[i]) / k
            pk[n] = ((1 - 2 * k) * pkm1[n] + (1 - k) * pkm2[n]) / k
            
            # Update previous polynomials for the next iteration
            if k < n:
                pkm2, pkm1 = pkm1, pk

    return pk
    

def samlmom(data, nmom):
    """
    Calculates the sample L-moments and their ratios for a given dataset.

    L-moments are linear combinations of order statistics that provide robust
    measures of the probability distribution characteristics, similar to
    conventional moments but less influenced by outliers.

    Parameters:
    data (np.array): Input data from which L-moments are to be calculated.
    nmom (int): Number of moments to calculate.

    Returns:
    tuple: A tuple containing two numpy arrays:
           - The first array contains the calculated L-moments.
           - The second array contains the L-moment ratios.

    Raises:
    ValueError: If `nmom < 1` or if the number of data points is less than `nmom`.
    """

    # Validate input parameters
    if nmom < 1 or len(data) < nmom:
        raise ValueError("Invalid number of moments or insufficient data length.")
    
    n = len(data)  # Total number of data points
    data = np.sort(data, axis=0)  # Sort data for order statistic calculations
    b0 = np.mean(data)  # First L-moment (mean)
    
    # Calculate preliminary L-moment coefficients
    b = np.zeros(nmom - 1)
    for i in range(1, nmom):
        a1 = np.tile(np.arange(i + 1, n + 1), (i, 1))
        a2 = np.tile(np.arange(1, i + 1), (n - i, 1)).T
        num = np.prod(a1 - a2, axis=0)
        den = np.prod([n] * i - np.arange(1, i + 1))
        b[i - 1] = 1 / n * sum(num / den * data[i:])
    B = np.append(b0, b)[::-1]  # Reversed coefficients for L-moment calculations
    
    # Calculate L-moments using the coefficients
    lmom = np.zeros(nmom - 1)
    for i in range(1, nmom):
        temp = np.zeros(len(B) - (i + 1))
        coeff = np.append(temp, legendre_shift_poly(i))
        lmom[i - 1] = sum(coeff * B)
    lambda_ = np.append(b0, lmom)  # Final L-moments array
    
    # Calculate L-moment ratios
    tau = np.ones(nmom)
    tau[1] = lambda_[1] / lambda_[0] if lambda_[0] != 0 else np.inf
    if lambda_[1] != 0:
        tau[2:nmom] = lambda_[2:nmom] / lambda_[1]
    else:
        # Handle potential division by zero if lambda_[1] is 0
        raise ValueError("Error: lambda[1] = 0, invalid for ratios.")

    return lambda_, tau

    
def pargev(xmom):
    """
    Estimates parameters of the Generalized Extreme Value (GEV) distribution
    using L-moments.

    Adaptation of Hosking's Fortran code to Python.

    Parameters:
    xmom (list): L-moments [lambda1, lambda2, tau3] of the sample.

    Returns:
    list: GEV distribution parameters [xi (location), alpha (scale), k (shape)].

    Raises:
    ValueError: If L-moments are invalid for GEV parameter estimation.
    """

    # Constants for numerical stability and iteration thresholds
    small = 1e-5
    eps = 1e-6
    maxit = 40
    eu = 0.57721566  # Euler's constant, used in computations

    # Coefficients for rational-function approximations for GEV shape parameter
    a0, a1, a2, a3, a4 = 0.28377530, -1.21096399, -2.50728214, -1.13455566, -0.07138022
    b1, b2, b3 = 2.06189696, 1.31912239, 0.25077104
    c1, c2, c3 = 1.59921491, -0.48832213, 0.01573152
    d1, d2 = -0.64363929, 0.08985247

    t3 = xmom[2]  # L-skewness
    if xmom[1] <= 0 or abs(t3) >= 1:
        raise ValueError(f'L-moments invalid for GEV parameter estimation: ')

    par_GEV = [0, 0, 0]  # Initialize GEV parameters

    # Use rational-function approximations based on the value of t3
    if t3 > 0:
        z = 1 - t3
        g = (-1 + z * (c1 + z * (c2 + z * c3))) / (1 + z * (d1 + z * d2))
        if abs(g) >= small:
            par_GEV[2] = g
            gam = math.gamma(1 + g)
            par_GEV[1] = xmom[1] * g / (gam * (1 - 2 ** (-g)))
            par_GEV[0] = xmom[0] - par_GEV[1] * (1 - gam) / g
        else:
            par_GEV[2] = 0
            par_GEV[1] = xmom[1] / math.log(2)
            par_GEV[0] = xmom[0] - eu * par_GEV[1]

    elif t3 >= -0.8:
        g = (a0 + t3 * (a1 + t3 * (a2 + t3 * (a3 + t3 * a4)))) / \
            (1 + t3 * (b1 + t3 * (b2 + t3 * b3)))
        par_GEV[2] = g
        gam = math.gamma(1 + g)
        par_GEV[1] = xmom[1] * g / (gam * (1 - 2 ** (-g)))
        par_GEV[0] = xmom[0] - par_GEV[1] * (1 - gam) / g

    else:  # Newton-Raphson method for more negative skewness
        g = 1 - math.log(1 + t3) / math.log(2) if t3 <= -0.97 else \
            (a0 + t3 * (a1 + t3 * (a2 + t3 * (a3 + t3 * a4)))) / \
            (1 + t3 * (b1 + t3 * (b2 + t3 * b3)))
        for it in range(maxit):
            x2, xx2 = 2 ** (-g), 1 - 2 ** (-g)
            x3, xx3 = 3 ** (-g), 1 - 3 ** (-g)
            t, deriv = xx3 / xx2, \
                (xx2 * x3 * math.log(3) - xx3 * x2 * math.log(2)) / (xx2 ** 2)
            g -= (t - (t3 + 3) * 0.5) / deriv
            if abs(g - par_GEV[2]) <= eps:
                break
        par_GEV[2] = g
        gam = math.gamma(1 + g)
        par_GEV[1] = xmom[1] * g / (gam * (1 - 2 ** (-g)))
        par_GEV[0] = xmom[0] - par_GEV[1] * (1 - gam) / g
        if it == maxit - 1:
            raise ValueError('Iteration did not converge for GEV parameter estimation')

    return par_GEV


def quagev(f, par_gev):
    """
    Calculates the quantiles for given non-exceedance probabilities in the
    Generalized Extreme Value (GEV) distribution for an array of probabilities.
    
    Parameters:
    - f (array-like): Non-exceedance probabilities (0 < f < 1),
                      representing the probabilities of observing values
                      less than or equal to the quantile.
    - par_gev (tuple/list): Parameters of the GEV distribution as (xi, alpha, k),
                            corresponding to location, scale, and shape parameters
                            respectively.
    
    Returns:
    - np.ndarray: The quantile values for the given non-exceedance probabilities.
    
    Raises:
    - ValueError: If the scale parameter is not positive or if probabilities
                  are not in the (0,1) interval.
    """
    u, a, g = par_gev  # Unpack location, scale, and shape parameters

    # Validate input parameters
    if a <= 0:
        raise ValueError('Scale parameter must be positive.')
    if np.any(f <= 0) or np.any(f >= 1):
        raise ValueError('Non-exceedance probabilities must be between 0 and 1.')

    # Compute quantiles based on the GEV parameters and probabilities
    if g != 0:
        # Calculate quantiles for non-zero shape parameter
        # The formula transforms log probabilities into the GEV scale
        y = (1 - np.exp(-g * (-np.log(-np.log(f))))) / g
    else:
        # Calculate quantiles for zero shape parameter (Gumbel distribution)
        y = -np.log(-np.log(f))

    quantile_gev = u + a * y  # Calculate the quantiles based on the location and scale

    return quantile_gev


def corrmatrix(ams):
    """
    Calculates the average absolute correlation of precipitation data
    across multiple sites and the standard error using bootstrap resampling.

    Parameters:
    - ams (pd.DataFrame): A DataFrame with an index of sites and columns for
                          'year' and 'precip', where 'precip' stands for precipitation.

    Returns:
    - tuple: A tuple containing the original rho, a list of bootstrap rhos, and the standard error of rho.
    """


        
    amswy = ams[['year', 'precip']].copy()
    amswy = amswy.reset_index()

    # Pivot table to get years as rows and sites as columns
    amswy = amswy.pivot(index='year', columns='HDSC', values='precip')

    # Compute the correlation matrix
    corrMatrix = amswy.corr()

    # Mask to extract the lower triangle excluding the diagonal
    lower_tri_mask = np.tril_indices_from(corrMatrix, k=-1)

    # Extract the lower triangle values
    lower_tri_values = corrMatrix.values[lower_tri_mask]

    # Replace NaNs with 1 to prevent them from affecting the mean calculation
    lower_tri_values[np.isnan(lower_tri_values)] = 1

    # Calculate the mean of the absolute values, ignoring NaNs
    rho = np.mean(np.abs(lower_tri_values))

    
    # print(std_dev)
    if len(lower_tri_values) < 3:
        std_dev = 0
    else:
        std_dev = np.std(abs(lower_tri_values), ddof=1)
        
    # # Calculate the number of observations
    # n = len(lower_tri_values)
    # # Calculate the standard error of the mean
    # sem = std_dev / np.sqrt(n)

    return rho, std_dev


def stations_lmom(group):
    """
    Calculates L-moments and L-moment ratios for a group of precipitation data, along with basic statistics.

    This function assumes that the input 'group' is a pandas DataFrame that includes a 'precip' column
    containing precipitation values. It computes the first five L-moments and L-moment ratios, and returns
    these values along with the count and mean of the precipitation data.

    Parameters:
    - group (pd.DataFrame): A DataFrame with a column 'precip' containing precipitation data.

    Returns:
    - pd.Series: A series containing the count ('n'), mean ('mean'), first L-moment ('l1'), second L-moment ('l2'),
                 L-CV ('t'), L-skewness ('t3'), L-kurtosis ('t4'), and fifth L-moment ratio ('t5').
    """

    # Calculate L-moments and L-moment ratios for the provided precipitation data
    lmom, lmomr = samlmom(group['precip'], nmom=5)

    # Creating a series to hold L-moments, their ratios, and basic statistics
    lmoms = pd.Series({
        'n': group['precip'].size,  # Count of data points
        'mean': group['precip'].mean(),  # Mean of the precipitation data
        'l1': lmom[0],  # First L-moment
        'l2': lmom[1],  # Second L-moment
        't': lmomr[1],  # L-CV (L-moment ratio)
        't3': lmomr[2],  # L-skewness (third L-moment ratio)
        't4': lmomr[3],  # L-kurtosis (fourth L-moment ratio)
        't5': lmomr[4],  # Fifth L-moment ratio
    })

    # Ensure the count of data points is represented as an integer
    lmoms['n'] = lmoms['n'].astype(int)

    return lmoms
    

def stations_para(group):
    """
    Computes GEV (Generalized Extreme Value) distribution parameters for a given station's data.

    This function calculates the parameters of the GEV distribution based on L-moments derived
    from precipitation data at a station. It utilizes the l1 (first L-moment), t (L-CV, ratio
    of the second L-moment to the first), and t3 (L-skewness, third L-moment ratio).

    Parameters:
    - group (pd.Series): A series containing l1, t, and t3 values for a specific station.

    Returns:
    - pd.Series: A series containing the GEV parameters 'xi' (location), 'alpha' (scale),
                 and 'k' (shape) calculated from the L-moments.

    Notes:
    - This function is tailored for use within a DataFrame.apply() where 'group' is expected
      to be a row of a DataFrame representing station-specific data.
    - The function `pargev` calculates the GEV parameters based on the provided L-moments.
    """

    # Extract the L-moment values from the group
    l1 = group['l1'].item()
    l2 = l1 * group['t'].item()  # second L-moment (l2)
    t3 = group['t3'].item()  # L-skewness

    # Calculate GEV parameters using the pargev function
    paras = pargev([l1, l2, t3])

    # Create and return a Series with the GEV parameters
    return pd.Series({
        'xi': paras[0],    # Location parameter
        'alpha': paras[1], # Scale parameter
        'k': paras[2]      # Shape parameter
    })



def samlmomgev(ams, mam_prism):
    """
    Computes L-moments and estimates GEV parameters for multiple stations' precipitation data.

    This function first calculates L-moments and L-moment ratios for each station using the input
    data. It then adjusts the mean precipitation values with PRISM MAM and estimates GEV
    parameters for each station based on the calculated L-moments.

    Parameters:
    - ams (pd.DataFrame): A multi-level DataFrame where the first level of index represents stations,
                          and the 'precip' column contains precipitation data.
    - mam_prism (pd.DataFrame): A DataFrame used to adjust the mean values in 'ams'; it must be
                                compatible for combining with 'ams' DataFrame's mean values.

    Returns:
    - tuple: A tuple containing two DataFrames:
             1. L-moments and L-moment ratios enhanced with adjusted mean precipitation values.
             2. GEV parameters estimated for each station.

    Notes:
    - This function assumes that both input DataFrames are properly formatted and that 'stations_lmom'
      and 'stations_para' are defined and correctly implemented elsewhere in the code.
    """
    
    # Calculate L-moments for each station
    lmoms = ams.groupby(level=0).apply(stations_lmom)
    
    # Ensure the count of data points is represented as an integer
    lmoms['n'] = lmoms['n'].astype(int)

    # Combine mean values from 'mam_prism' with 'lmoms' for missing or supplementary data
    lmoms['mean'] = mam_prism.combine_first(lmoms['mean'])

    # Calculate GEV parameters for each station based on L-moments
    paras = lmoms.groupby(level=0).apply(stations_para)

    return lmoms, paras


def stations_quant(f, group):
    """
    Calculates the GEV quantiles for specified non-exceedance probabilities using GEV parameters.

    This function extracts GEV distribution parameters from a given group (DataFrame) and computes
    the corresponding quantiles for a list of non-exceedance probabilities.

    Parameters:
    - f (np.array or list): Non-exceedance probabilities (0 < f < 1) for which quantiles are computed.
    - group (pd.DataFrame): A DataFrame that must contain the columns 'xi', 'alpha', and 'k', which
                            represent the location, scale, and shape parameters of the GEV distribution.

    Returns:
    - pd.Series: A Series containing the quantiles corresponding to the probabilities in `f`.

    Notes:
    - The function assumes that the input DataFrame `group` contains only a single set of parameters.
    - `quagev` function must be defined and correctly implemented elsewhere in the code to compute GEV quantiles.
    """
    
    # Extract GEV parameters from the DataFrame
    paras = [group['xi'].item(), group['alpha'].item(), group['k'].item()]
    
    # Compute the quantiles using the extracted GEV parameters and given probabilities
    quantile = quagev(f, paras)
    
    # Return the quantiles as a pandas Series
    return pd.Series(quantile)


class Simulation:
    def __init__(self, sid, base, rho, sd_rho, ARI, lmoms, smam, nsim, distributions, boundprob):
        """
        Initialize the Simulation class with the required parameters and data.

        Parameters:
        - sid: Station identifier.
        - base: Duration
        - rho: Inter-site correlation coefficient for constructing covariance matrix.
        - ARI: Annual Recurrence Interval for quantile calculations.
        - lmoms: DataFrame containing L-moments and other statistical data per station.
        - smam: Mean annual maximum precipitation from PRISM if available or at-station MAM, used for regional adjustments.
        - nsim: Number of Monte Carlo simulations to perform.
        - boundprob: Tuple of lower and upper bounds for confidence interval calculation.
        """
        self.sid = sid
        self.rho = rho
        self.sd_rho = sd_rho
        self.base = base
        self.ARI = ARI
        self.lmoms = lmoms
        self.nsim = nsim
        self.boundprob = boundprob
        self.dist = distributions
        self.ndis = len(distributions)
        self.smam = smam # REFERENCE STATION PRISM MAM
        self.nsites = self.lmoms['n'].count()
        self.nmax = self.lmoms['n'].max()
        self.nmom = self.lmoms.columns.size - 2
        self.nyears = self.lmoms['n']
        self.min_nyears = min(self.nyears)
        self.max_nyears = max(self.nyears)
        self.weights = self.lmoms['n']/self.lmoms['n'].sum()
        #self.weights = self.lmoms['weights_years']
        lambda_columns = [f'l{i+1}' for i in range(self.nmom)]
        tau_columns = [f't{i+1}' for i in range(self.nmom)]
        self.columns = ['HDSC'] + lambda_columns + tau_columns

    def regStats(self):
        """
        Compute regional statistics based on the L-moments and weights derived from length of record.
        """        

        self.rtau = np.average(self.lmoms['t'], weights=self.weights)
        self.rtau3 = np.average(self.lmoms['t3'], weights=self.weights)
        rpara = pargev([self.smam, self.smam*self.rtau, self.rtau3]) 
        self.rquant = quagev(1 - 1/self.ARI, rpara)

    def covMatrix(self):
        """
        Create a covariance matrix based on the specified correlation coefficient and number of sites.
        """        
        self.R = (1 - self.rho_i) * np.eye(self.nsites) + self.rho_i * np.ones((self.nsites, self.nsites))


    def hetRegion(self):
        """
        Generate synthetic GEV parameters for a simulated region to use in Monte Carlo simulations by incorporating uncertainty
        in sample L-moment ratios.
        
        The L-moment-1 for the sites in a simulated region are generated by including the variability (standard deviation) of
        L-moment-1 from the actual sample data. A linearly varying L-moment-1 centered around the reference station MAM (L-moment-1)
        is used for the simulated region.
        
        A standard error of the regional weighted mean of L-moment ratio (rτ) is calculated and used to create a linearly varying 
        τ centered around the reference station rτ. This is then multiplied with the L-moment-1 (λ1) to generate the second
        L-moment (λ2).
        
        Similarly, a standard error of the regional weighted mean of L-moment ratio-3 (rτ3, L-skewness) is calculated and used to create a linearly varying 
        τ3 centered around the reference station rτ3.
        
        These L-moments are used to calculate the GEV parameters. Finally, nsim (1000) GEV parameters are randomly generated from these nsites GEV parameters.
        """
 
        effective_sample_size = (np.sum(self.weights) ** 2) / np.sum(self.weights**2)
        ######################## first l-moment ########################
        # standard deviation of first l-moment
        # std_mam = self.lmoms['l1'].std() # These l-moments are from the AMS of stations in a region 
        # exl1 = np.linspace(self.smam - std_mam, self.smam + std_mam, self.nsites) # Linearly varying first l-moment
        # exl1 = np.abs(exl1) ### The varaibility of the 'l1' is so high that one of the exl1 value will be negative. (id:26-4935,60d)

        weighted_variance_rl1 = np.average((self.lmoms['l1'] - self.lmoms['l1'].mean()) ** 2, weights=self.weights)
        standard_error_rl1 = np.sqrt(weighted_variance_rl1 / effective_sample_size)
        exl1 = np.linspace(self.smam - standard_error_rl1, self.smam + standard_error_rl1, self.nsites)
        ######################### second l-moment ########################
        weighted_variance_rtau = np.average((self.lmoms['t'] - self.rtau) ** 2, weights=self.weights)
        standard_error_rtau = np.sqrt(weighted_variance_rtau / effective_sample_size)
        exl2 = exl1 * np.linspace(self.rtau - standard_error_rtau, self.rtau + standard_error_rtau, self.nsites)

        ######################## L-skewness ########################
        # The variability in L-skewness is not included when the total number of stations in the region is less than 5.
        # This is because if any station has a higher τ3, it can make the upper bounds very wide for higher return
        # periods, which is unrealistic. Additionally, for a few SNOTEL daily gauge stations, the sub-daily total
        # years can be less than 100. This small sample size creates higher simulated regional τ3 variability.
        # Consequently, the upper bounds at high return periods become very wide. For regions with fewer than 5 sites, we assume
        # each site has the same regional weighted τ3 from the actual sample data.

        if self.nsites < 5:
            ext3 = [self.rtau3] * self.nsites
        else:            
            weighted_variance_rtau3 = np.average((self.lmoms['t3'] - self.rtau3) ** 2, weights=self.weights)
            standard_error_rtau3 = np.sqrt(weighted_variance_rtau3 / effective_sample_size)
    
            ext3 = np.linspace(self.rtau3 - standard_error_rtau3,
                               self.rtau3 + standard_error_rtau3, self.nsites)

        # GEV parameters
        hpara = np.array([pargev(e) for e in zip(exl1, exl2, ext3)])
        indices = np.random.choice(np.arange(self.nsites), size=self.nsim, replace=True)
        self.hetparas = np.column_stack(hpara[indices]).T
        

    def perform_simulation(self, i):
        """
        Execute a single simulation, modeling precipitation data using the covariance matrix
        and heterogeneous GEV parameters.
        """
        self.rho_i = np.clip(np.random.normal(self.rho, self.sd_rho), 0, 1)
        self.covMatrix()
        
        mean = np.zeros(self.nsites)
        y = np.random.multivariate_normal(mean, self.R, self.nmax)
        ycdf = norm.cdf(y).T
        
        num = random.randint(0,self.nsites-1)
        num3 = random.randint(0,self.ndis-1)

        res = []
        for index, (hdscid, irows) in enumerate(self.nyears.groupby(level=0)):

            len_record = random.randint(self.min_nyears, self.max_nyears)
            # caclulate the quantile for the cdf 
            # newdata = quagev(ycdf[index, :irows.values[0]], self.hetparas[i].tolist())
            newdata = quagev(ycdf[index, :len_record], self.hetparas[i].tolist())
            # sample l-moments
            lambda_, tau = samlmom(newdata, self.nmom)
            res.append([hdscid] + list(lambda_) + list(tau))
        
        mclmom = pd.DataFrame(res, columns=self.columns).set_index('HDSC', drop=True)
        mmam = mclmom.iloc[num,:]['l1'] # Randomly select the MAM 
        # regional l-moments ratios through weighted mean of l-moment ratios
        mct2 = np.average(mclmom['t2'], weights=self.weights)
        mct3 = np.average(mclmom['t3'], weights=self.weights)
        regional_lmoments = [mmam, mmam*mct2, mct3]

        dis = self.dist[num3]
        dist_params = dis.lmom_fit(lmom_ratios=regional_lmoments)
        fitted = dis(**dist_params)
        mcrgc = fitted.ppf(1 - 1 / self.ARI)
    
        # mcrpara = pargev([mmam, mmam*mct2, mct3])
        # mcrgc = quagev(1 - 1 / self.ARI, mcrpara)

        return mcrgc

    def run_simulations(self):
        """
        Execute all simulations and collect the results.
        """       
        #self.covMatrix() # Generate Covaraince Matrix
        self.regStats() # Regional L-moments and GEV parameters
        self.hetRegion() # Generate synthetic region (model for Monte Carlo Simulaiton)

        self.mcrgc = Parallel(n_jobs=-1)(delayed(self.perform_simulation)(i) for i in range(self.nsim))
        self.mcrgc = np.array(self.mcrgc).T

    def bounds(self):
        """
        Calculate confidence bounds for the simulations.
        """
        slbound = np.percentile(self.mcrgc, int(self.boundprob[0]*100), axis=1)
        subound = np.percentile(self.mcrgc, int(self.boundprob[1]*100), axis=1)

        return self.rquant, slbound, subound