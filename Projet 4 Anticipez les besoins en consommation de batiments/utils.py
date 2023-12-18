import pandas as pd
import numpy as np
from scipy import stats
import time
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, median_absolute_error

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


def model_evaluation_1(model, name, X_train, y_train, X_test, y_test):
    """
    Evaluate a machine learning model's performance.

    Parameters:
    - model (object): The machine learning model to be evaluated.
    - name (str): The name of the model.
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data target.
    - X_test (array-like): Test data features.
    - y_test (array-like): Test data target.

    Returns:
    - results (list): List containing model evaluation results:
        [name, train_score, test_score, fit_time, predict_time]
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    fit_time = t1 - t0

    t2 = time.time()
    y_pred = model.predict(X_test)
    t3 = time.time()
    predict_time = t3 - t2

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    results = [name, train_score, test_score, fit_time, predict_time]

    intpart_len = len(f'{int(fit_time)}')
    if intpart_len >= 4:
        fit_time = int(fit_time)
    else:
        fit_time = round(fit_time, 4 - intpart_len)

    formatter_result = (
        f"{name:12s}\t\t {train_score:.3f}\t\t {test_score:.3f}\t\t {fit_time}s\t\t {predict_time:.3f}s"
    )
    print(formatter_result.format(*results))

    return results


def model_comparison(models, X_train, y_train, X_test, y_test, preprocessing_pipeline=None):
    """
    Compare the performance of multiple machine learning models.

    Parameters:
    - models (dict): A dictionary containing model names as keys and corresponding model objects as values.
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data target.
    - X_test (array-like): Test data features.
    - y_test (array-like): Test data target.
    - preprocessing_pipeline (object, optional): Preprocessing pipeline to be applied before fitting the models.

    Returns:
    - model_results (DataFrame): A DataFrame containing model comparison results.
    """
    list_results = []
    print(85 * "_")
    print("model\t\t\t train_score\t test_score\t fit_time\t predict_time")
    print(85 * "=")
    for name, model in models.items():
        if preprocessing_pipeline is not None:
            pipeline = make_pipeline(preprocessing_pipeline, model)
            results = model_evaluation_1(pipeline, name, X_train, y_train, X_test, y_test)
        else:
            results = model_evaluation_1(model, name, X_train, y_train, X_test, y_test)
        list_results.append(results)
        print(85 * "-")

    model_results = pd.DataFrame(
        {results[0]: results[1:] for results in list_results},
        index=['train_score', 'test_score', 'fit_time', 'predict_time']
    ).T.round(4)

    model_results.index.name = 'model'
    return model_results


def model_evaluation_2(model, X_train, X_test, y_train, y_test, return_preds=False):
    y_fit = model.predict(X_train)
    y_pred = model.predict(X_test)

    # compute scores
    train_r2 = r2_score(y_train, y_fit).round(3)
    test_r2 = r2_score(y_test, y_pred).round(3)
    train_mape = mean_absolute_percentage_error(y_train, y_fit).round(3)
    test_mape = mean_absolute_percentage_error(y_test, y_pred).round(3)
    train_mae = mean_absolute_error(y_train, y_fit).round(3)
    test_mae = mean_absolute_error(y_test, y_pred).round(3)
    train_medae = median_absolute_error(y_train, y_fit).round(3)
    test_medae = median_absolute_error(y_test, y_pred).round(3)

    # print results
    table = [
        ["Score", "Train", "Test"],
        ["R2", train_r2, test_r2],
        ["MAPE", train_mape, test_mape],
        ["MAE", train_mae, test_mae],
        ["MedAE", train_medae, test_medae]
    ]
    col_width = 15  # You can adjust this based on your data
    for row in table:
        print("".join(str(word).ljust(col_width) for word in row))
        
    if return_preds:
        return y_fit, y_pred
