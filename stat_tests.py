import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf


def load_data(parameter_path: Path = None,
              timeseries_path: Path = None,
              inplace_path: Path = None
              ):
    para_df = pd.read_parquet(parameter_path)
    inpl_df = pd.read_parquet(inplace_path)
    ts_df = pd.read_parquet(timeseries_path)

    ts_df.columns = [col.replace(":", "_") for col in ts_df.columns]
    inpl_df.columns = [col.replace(":", "_") for col in inpl_df.columns]
    para_df.columns = [col.replace(":", "_") for col in para_df.columns]

    return (para_df, inpl_df, ts_df)


def filter_inplace_volumes(
        param_df: pd.DataFrame,
        inplace_df: pd.DataFrame,
        inplace_filters: dict,
        global_filters: dict,
        param_filters: list=[]):
    inplace_filtered = inplace_df.filter(
            items=[
                "ENSEMBLE",
                "REAL",
                "ZONE",
                "REGION",
                inplace_filters["RESPONSE"]
                ])
    inplace_filtered = inplace_filtered.loc[
        (inplace_filtered["ZONE"] == inplace_filters["ZONE"]) &
        (inplace_filtered["REGION"] == inplace_filters["REGION"]) &
        (inplace_filtered["ENSEMBLE"] == global_filters["ENSEMBLE"])]
    inplace_filtered = inplace_filtered.drop(columns=["ZONE", "REGION"])
    return pd.merge(param_df,
                    inplace_filtered,
                    on=["ENSEMBLE", "REAL"],
                    validate="m:m").drop(columns=["ENSEMBLE", "REAL", "RMSGLOBPARAMS_COHIBA_MODEL_MODE",
                           "COHIBA_MODEL_MODE"]+param_filters)


def filter_timeseries(param_df: pd.DataFrame,
                      timeseries_df: pd.DataFrame,
                      timeseries_filters: dict,
                      global_filters: dict,
                      param_filters: list=[]):

    timeseries_filtered = timeseries_df.filter(items=[
        "ENSEMBLE",
        "REAL",
        "DATE",
        timeseries_filters["RESPONSE"]])
    timeseries_filtered = timeseries_filtered.loc[
        (timeseries_filtered["DATE"] == timeseries_filters["DATE"]) &
         (timeseries_filtered["ENSEMBLE"] == global_filters["ENSEMBLE"])]
    return pd.merge(param_df, timeseries_filtered,
                    on=["ENSEMBLE", "REAL"],
                    validate="m:m").drop(columns=[
                                            "ENSEMBLE",
                                            "REAL",
                                            "DATE",
                                            "RMSGLOBPARAMS_COHIBA_MODEL_MODE",
                                            "COHIBA_MODEL_MODE"]+ param_filters)


def filter_df(param_df: pd.DataFrame,
              response_df: pd.DataFrame,
              filters: dict):
    filtered_df=pd.merge(param_df, response_df, on=["ENSEMBLE", "REAL"])
    keys=list(filters.keys())
    filtered_df = filtered_df.filter(items=keys)
    print(filtered)
    for key, value in filters.items():
        if key!="RESPONSE":
            filtered_df = filtered_df.loc[filtered_df[key] == value]
            print(filtered_df.head())
    return filtered_df


def forward_selected(data, response, maxvars=3):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score and len(selected) < maxvars:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


def fit_regression(filtered_df: pd.DataFrame, response: str, maxvars: int=5,testsize=0.25):
    df_train, df_test = train_test_split(filtered_df, test_size=0.25, random_state=42)
    best_model = forward_selected(df_train, response,maxvars=maxvars)

    print(best_model.summary())




param_path = "/home/jg/Documents/work/equinor/Equinor_R_models/equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/parameters100realizations.parquet"
ts_path = "/home/jg/Documents/work/equinor/Equinor_R_models/equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/response_timeseries_100realizations.parquet"
inplace_volumes_path = "/home/jg/Documents/work/equinor/Equinor_R_models/equinor-R-models/data files/100 realizations, 4 iterations (ensembles)/response_grid_volumes_100realizations.parquet"
ip_filters = {"ENSEMBLE": "iter-0", "ZONE": "UpperReek", "REGION": 1, "RESPONSE": "BULK_OIL"}
gl_filters = {"ENSEMBLE": "iter-0"}
ts_filters = {"ENSEMBLE": "iter-0", "DATE": pd.to_datetime("2001-01-01"), "RESPONSE": "FPR"}
parameter_filters = ['RMSGLOBPARAMS_FWL', 'MULTFLT_MULTFLT_F1', 'MULTFLT_MULTFLT_F2',
       'MULTFLT_MULTFLT_F3', 'MULTFLT_MULTFLT_F4', 'MULTFLT_MULTFLT_F5',
       'MULTZ_MULTZ_MIDREEK', 'INTERPOLATE_RELPERM_INTERPOLATE_GO',
       'INTERPOLATE_RELPERM_INTERPOLATE_WO']


para_df, ip_df, ts_df = load_data(parameter_path=param_path,
                                  timeseries_path=ts_path, 
                                  inplace_path=inplace_volumes_path)


ip_filtered = filter_inplace_volumes(param_df=para_df,
                                     inplace_df=ip_df,
                                     inplace_filters=ip_filters,
                                     global_filters=gl_filters,
                                     #param_filters=parameter_filters
                                     )
ts_filtered = filter_timeseries(param_df=para_df,
                                timeseries_df=ts_df,
                                timeseries_filters=ts_filters,
                                global_filters=gl_filters,
                                param_filters=parameter_filters
                                )
#ts2_filtered = filter_df(para_df,ts_df,ts_filters)
#print(ts2_filtered.head(),"CRACK IS WHACK")
print(ts_filtered.columns)
print(ts_filtered.corr())
fit_regression(ts_filtered, ts_filters["RESPONSE"], maxvars=4, testsize=0.1)

# FPR = coef1*var1+ 