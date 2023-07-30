import pandas as pd
import numpy as np
from scipy import stats

def print_correlations(df, min_corr, max_corr=1, print_correlations=True, return_variables=False):
    corr = df.corr(numeric_only=True)
    mask = pd.DataFrame(np.triu(np.ones_like(corr, dtype=bool)), columns=corr.columns, index=corr.index)
    corr = corr[~mask]

    mask = (corr.abs() > min_corr) & (corr.abs() <= max_corr)
    correlated_variables = []
    for col in corr.columns:
        values = corr[mask][col].dropna().values.round(3) # we round to the thousandth
        if values.size > 0:
            correlated_variables.append(col)
            indexes = corr[mask][col].dropna().index
            correlated_variables += list(indexes)
            if print_correlations == True:
                for i in range(indexes.size):
                    print(f'{col} / {indexes[i]}' + '-'*(60-(len(col)+len(indexes[i]))) + f'({values[i]})')
    
    if return_variables == True:
        correlated_variables = list(set(correlated_variables))
        return correlated_variables
    


def anova(X_name, Y_name, data, print_values=False):

    k = len(data[X_name].unique())  # number of groups
    N = len(data.values)  # total number of values
    n = data.groupby(X_name).size() # number of values in each group

    # Degrees of freedom
    databetween = k - 1
    datawithin = N - k
    datatotal = N - 1

    y = data[Y_name]
    y_mean = y.mean()
    yi_means = data.groupby(X_name)[Y_name].mean()
    
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

    # p-value
    p = stats.f.sf(F, databetween, datawithin)

    # eta squared
    eta_sqrd = SSbetween/SStotal

    # omega squared
    om_sqrd = (SSbetween - (databetween * MSwithin))/(SStotal + MSwithin)
    
    if print_values == True:
        print('eta squared :', eta_sqrd.round(3))
        #print('omega squared :', om_sqrd.round(3))
        #print('F-value :', F)
        print('p-value :', p)
        
    
    return eta_sqrd, p