import numpy as np
import pandas as pd
from scipy import stats

# CORRELATION MATRIX

def print_correlations(df, min_corr, max_corr=1):
    """
    Print the correlations between different variables if the value is between min_corr and max_corr.
    
    
    Parameters
    ----------
    corr : class:`pandas.DataFrame`
        The DataFrame from which we compute the correlation matrix.
    
    min_corr : float
        Minimum correlation value to print.
    
    max_corr : float, default: 1
        Maximum correlation value to print.
    """
    
    # We fill the upper right diagonal with NaNs so we don't print the correlation of the variable
    # with itself and we dont print 2 times the same pair of variables
    corr = df.corr()
    for i in range(corr.shape[0]):
        for j in np.arange(corr.shape[1])[i:]:
            corr.iloc[i, j] = np.nan

    mask = (corr > min_corr) & (corr <= max_corr)

    for col in corr.columns:
        values = corr[mask][col].dropna().values.round(3) # we round to the thousandth
        if values.size > 0:
            indexes = corr[mask][col].dropna().index
            for i in range(indexes.size):
                print(f'{col} / {indexes[i]}' + '-'*(60-(len(col)+len(indexes[i]))) + f'({values[i]})')
                
                
# ANOVA

def anova(X_name, Y_name, data):

    k = len(data[X_name].unique())  # number of groups
    N = len(data.values)  # total number of values
    n = data.groupby(X_name).size() # number of values in each group

    # Degrees of freedom
    databetween = k - 1
    datawithin = N - k
    datatotal = N - 1

    y = data[Y_name]
    y_mean = y.mean()
    yi_means = data.groupby(X_name).mean()[Y_name]
    
    # Sum of Squares Between
    SSbetween = (n*(yi_means - y_mean)**2).sum()

    # Sum of Squares Within
    groups = data[X_name].unique()
    group_var = []
    for group in groups:
        y_group = data.loc[data[X_name] == group, [Y_name]]
        group_var.append(((y_group.values - yi_means[group])**2).sum())
    SSwithin = np.array(group_var).sum()

    # Sum of Squares Total
    SStotal = ((y - y_mean)**2).sum()

    # Mean Square Between
    MSbetween = SSbetween/databetween

    # Mean Square Within
    MSwithin = SSwithin/datawithin

    # Calculating the F-value
    F = MSbetween/MSwithin
    #print('F-value :', F)

    # p-value
    p = stats.f.sf(F, databetween, datawithin)
    print('p-value :', p)

    # eta squared
    eta_sqrd = SSbetween/SStotal
    print('eta squared :', eta_sqrd.round(3))

    # omega squared
    om_sqrd = (SSbetween - (databetween * MSwithin))/(SStotal + MSwithin)
    #print('omega squared :', om_sqrd.round(3))
    

