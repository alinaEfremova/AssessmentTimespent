import csv
import json
import time
import pandas as pd
import params as params
import statsmodels.api as sm
import plotly.graph_objects as go
from statsmodels.formula.api import ols
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.power import tt_ind_solve_power, FTestAnovaPower
from scipy.stats import pearsonr, kendalltau
from plotly.subplots import make_subplots
from numpy import nan, log

pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)


def parse_answer(file_name, store=True):
    """
    parse_answer transform survey responses from csv to pandas DataFrame format,
    additionally, if required, save the data in json format.

    :param file_name: csv file with teachers' survey responses
    :param store: indicator whether the reformatted responses need to be stored in json
    :return: teacher responses in format of pandas DataFrame
    """

    if store:
        parsed_data = {}
        with open(file_name, encoding="utf8") as f:
            reader = csv.reader(f)
            header = next(reader)
            parsed_data["questions"] = header
            for field in header:
                parsed_data[field] = []
            for row in reader:
                for i, value in enumerate(row):
                    parsed_data[parsed_data["questions"][i]].append(value)

        with open(f"ParsedAnswers_{time.time()}.json", 'w') as outfile:
            json.dump(parsed_data, outfile)

    data_df = pd.read_csv(file_name)
    data_df.columns = params.header

    return data_df


def substitute(x, c_name):
    """
    substitute transform one value for the other one, if the substituton is defined

    :param x: original value
    :param c_name: column name
    :return: the value that should be present
    """
    if params.substitution[c_name].get(x) is not None:
        return params.substitution[c_name][x]
    else:
        return x


def content_substitution(dataframe):
    """
    content_substitution goes over the given DataFrame to substitute the long values
    for their shorter version defined in params.

    :param dataframe: pandas DataFrame with data
    :return: pandas DataFrame with shortened values
    """
    columns = dataframe.columns
    for column in columns:
        if params.substitution.get(column) is not None:
            dataframe[column] = dataframe[column].apply(lambda x: substitute(x, column))
        if column == "grade_source":
            dataframe[column] = dataframe[column].apply(lambda x: int(x.count(";") - x.count("; ")) + 1)

    return dataframe


def continous_correlation(independent, dependent):
    """
    continous_correlation calculates correlation between given DataFrames
    given that the independent is for the continous data

    :param independent: pandas DataFrame with independent variables
    :param dependent: pandas DataFrame with dependent variables
    :return: pandas DataFrame with correlation data
    """

    # Init of resulting DataFrame
    corr_data = pd.DataFrame()
    for indep_var in independent.columns:
        # Drop nans
        combined = pd.concat([independent[indep_var], dependent], axis=1).dropna()
        # Write down sample size
        corr_data.loc[indep_var, "sample_size"] = len(combined)
        # Calculate correlations with scipy methods
        corr_data.loc[indep_var, 'pearson_rho'], corr_data.loc[indep_var, 'pearson'] = pearsonr(combined[indep_var],
                                                                                                combined[
                                                                                                    dependent.name])
        corr, corr_data.loc[indep_var, 'kendall'] = kendalltau(combined[indep_var], combined[dependent.name])
        corr_data.loc[indep_var, 'log_pearson_rho'], corr_data.loc[indep_var, 'log_pearson'] = nan, nan
        corr_data.loc[indep_var, f'log({indep_var})_{dependent.name}_rho'], \
        corr_data.loc[indep_var, f'log({indep_var})_{dependent.name}_pearson'] = nan, nan
    return corr_data


def categorical_correlation(independent, dependent, alpha=0.05):
    """
    categorical_correlation calculates correlation between given DataFrames
    given that the independent is for the categorical data

    :param independent: pandas DataFrame with independent variables
    :param dependent: pandas DataFrame with dependent variables
    :param alpha: float, signifying significance level
    :return: pandas DataFrame with correlation data
    """

    corr_data = pd.DataFrame(columns=['time'], index=independent.columns)
    for indep_var in independent.columns:
        combined = pd.concat([independent[indep_var], dependent], axis=1).dropna()
        x1 = combined[combined[indep_var] == "Yes"][dependent.name]
        x2 = combined[combined[indep_var] == "No"][dependent.name]
        stat, p, df = ttest_ind(x1, x2)
        effect_size = abs((x1.mean() - x2.mean())) / x1.std()
        corr_data.loc[indep_var, 'time'] = p
        corr_data.loc[indep_var, 'power'] = tt_ind_solve_power(effect_size, len(x1), alpha, ratio=len(x2) / len(x1))
        corr_data.loc[indep_var, 'effect_size'] = effect_size
    return corr_data


def simple_cat_reg(param, dataframe):
    """
    simple_cat_reg calculates regression with the given formula
    'assess_time ~ C(max_assignment_type) + assignments_volume'

    :param param: formula used for the model
    :param dataframe: pandas DataFrame with the data
    :return: r-squared of the resulting model and its parameters
    """
    model = ols(param, data=dataframe).fit()
    print(f"R-squared = {model.rsquared}")
    print("Parameters are:")
    print(model.summary())
    return model.rsquared, model.params


def calc_anova(independent, dependent, alpha=0.05):
    """
    calc_anova calculates correlation between given DataFrames
    given that the independent is for the categorical data.

    :param independent: pandas DataFrame with the independent variables with one column
    :param dependent: pandas DataFrame with the dependent variables with one column
    :param alpha: float, signifying significance level
    :return: 'per variable' pandas DataFrame, pandas DataFrame with the correlation data,
        values of power and effect size of the tests
    """
    combined = pd.concat([independent, dependent], axis=1)
    combined = combined.rename(columns={independent.name: "indep", dependent.name: "dep"})
    mod = ols('dep ~ C(indep)', data=combined).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    effect_size = aov_table["sum_sq"]["C(indep)"] / aov_table["sum_sq"]["Residual"]
    power = FTestAnovaPower().power(effect_size, len(combined), alpha, len(combined.groupby(by="indep").groups))
    corr = []
    if aov_table['PR(>F)']['C(indep)'] < 0.05:
        pair_t = mod.t_test_pairwise('C(indep)')
        corr = pair_t.result_frame[pair_t.result_frame['reject-hs']].index.to_list()
        # print(corr)
    return aov_table['PR(>F)']['C(indep)'], corr, power, effect_size


def anova_corr(independent, dependent, alpha=0.05):
    """
    anova_corr calculates correlation between given DataFrames
    given that the independent is for the categorical data.

    :param independent: pandas DataFrame with the independent variables
    :param dependent: pandas DataFrame with the dependent variables with one column
    :param alpha: float, signifying significance level
    :return: 'per variable' pandas DataFrame, pandas DataFrame with the correlation data,
        values of power and effect size of the tests
    """
    corr_data = pd.DataFrame(columns=[dependent.name], index=independent.columns)
    add_corr_data = pd.DataFrame(columns=[dependent.name], index=independent.columns)
    for indep_var in independent.columns:
        combined = pd.concat([independent[indep_var], dependent], axis=1).dropna()
        p, table, power, effect_size = calc_anova(combined[indep_var], combined[dependent.name], alpha)
        corr_data.loc[indep_var, dependent.name] = p
        add_corr_data.loc[indep_var, dependent.name] = table
        corr_data.loc[indep_var, f"{dependent.name}_power"] = power
        corr_data.loc[indep_var, f"{dependent.name}_effect_size"] = effect_size
    return corr_data, add_corr_data


def rem_outliers(s):
    """
    rem_outliers applies three-sigma rule to clean the incoming pandas Series.

    :param s: pandas Series with data
    :return: indices of non-outliers
    """
    s_mean = s.mean()
    s_std = s.std()
    s_min = s_mean - 3 * s_std
    s_max = s_mean + 3 * s_std
    return s.loc[(s_min < s.loc[:]) & (s.loc[:] < s_max)].index.to_list()


def separate_correlations(categorical, continous, dependent, drop_outliers=False, alpha=0.05, tps=False,
                          replacement=0.01):
    """
    separate_correlations calculates correlation between independent and dependent variables.

    :param categorical: pandas DataFrame with categorical variables
    :param continous: pandas DataFrame with continous variables
    :param dependent: pandas DataFrame with dependent variables
    :param drop_outliers: boolean variable, setting whether to drop dependent and continous variables outliers
    :param alpha: float, signifying statistical significance
    :param tps: boolean variable, modyfing the logic for the 'per student' mode
    :param replacement: float, zeros in the data will be replaced with it
    :return: pandas DataFrames with correlation data for the continous data and for the categorical one
        along with some additional data from the ANOVA test for the categorical data
    """
    categ_corr_data = pd.DataFrame()
    cont_corr_data = pd.DataFrame()
    add_info = pd.DataFrame()
    for i, dep_col in enumerate(dependent.columns):
        if drop_outliers:
            combined = pd.concat([categorical, continous, dependent], axis=1)
            dep_mean = dependent[dep_col].mean()
            dep_std = dependent[dep_col].std(ddof=0)
            dep_min = dep_mean - 3 * dep_std
            dep_max = dep_mean + 3 * dep_std
            comb_no_outliers = combined.loc[(dep_min < combined[dep_col]) & (combined[dep_col] < dep_max)]
            cat_no_outliers = comb_no_outliers.loc[:, categorical.columns]
            dep_no_outliers = comb_no_outliers.loc[:, dep_col]
            if not categorical.empty:
                new_anova_corr, new_add_info = anova_corr(cat_no_outliers[params.anova_categorical], dep_no_outliers,
                                                          alpha)
                new_categ_data = pd.concat([categorical_correlation(cat_no_outliers[params.t_categorical],
                                                                    dep_no_outliers, alpha)
                                           .rename(columns={"time": dep_col,
                                                            "power": f"{dep_col}_power",
                                                            "effect_size": f"{dep_col}_effect_size"}),
                                            new_anova_corr])
            else:
                new_add_info = pd.DataFrame()
                new_categ_data = pd.DataFrame()
            cont_cols = continous.columns.to_list()
            if (i != len(dependent.columns) - 1) and not tps:
                cont_cols = cont_cols + dependent.columns.tolist()[i + 1:]
            cont_mean = comb_no_outliers.loc[:, cont_cols].mean()
            cont_std = comb_no_outliers.loc[:, cont_cols].std()
            cont_min = cont_mean - 3 * cont_std
            cont_max = cont_mean + 3 * cont_std
            new_cont_data = pd.DataFrame()
            for cont_col in cont_cols:
                cont_no_outliers = comb_no_outliers.loc[(cont_min[cont_col] < comb_no_outliers[cont_col]) &
                                                        (comb_no_outliers[cont_col] < cont_max[cont_col])]
                dep_no_outliers = cont_no_outliers.loc[:, dep_col]
                cont_no_outliers = pd.DataFrame(cont_no_outliers.loc[:, cont_col])
                cont_col_corr = continous_correlation(cont_no_outliers, dep_no_outliers) \
                    .rename(columns={"sample_size": f"sample_size_{dep_col}",
                                     "pearson_rho": f"pearson_rho_{dep_col}",
                                     "pearson": f"pearson_{dep_col}",
                                     "kendall": f"kendall_{dep_col}",
                                     "log_pearson_rho": f"log_pearson_rho_{dep_col}",
                                     "log_pearson": f"log_pearson_{dep_col}"})
                if new_cont_data.empty:
                    new_cont_data = cont_col_corr
                else:
                    new_cont_data = pd.concat([new_cont_data, cont_col_corr])
        else:
            if not categorical.empty:
                new_anova_corr, new_add_info = anova_corr(categorical[params.anova_categorical], dependent[dep_col],
                                                          alpha)
                new_categ_data = pd.concat(
                    [categorical_correlation(categorical[params.t_categorical], dependent[dep_col], alpha)
                         .rename(columns={"time": dep_col,
                                          "power": f"{dep_col}_power",
                                          "effect_size": f"{dep_col}_effect_size"}),
                     new_anova_corr])
            else:
                new_add_info = pd.DataFrame()
                new_categ_data = pd.DataFrame()
            new_cont_data = continous_correlation(continous, dependent[dep_col], replacement) \
                .rename(columns={"sample_size": f"sample_size_{dep_col}",
                                 "pearson_rho": f"pearson_rho_{dep_col}",
                                 "pearson": f"pearson_{dep_col}",
                                 "kendall": f"kendall_{dep_col}",
                                 "log_pearson_rho": f"log_pearson_rho_{dep_col}",
                                 "log_pearson": f"log_pearson_{dep_col}"})
        if cont_corr_data.empty:
            categ_corr_data = new_categ_data
            cont_corr_data = new_cont_data
            add_info = new_add_info
        else:
            categ_corr_data = pd.concat([categ_corr_data, new_categ_data], axis=1)
            cont_corr_data = pd.concat([cont_corr_data, new_cont_data], axis=1)
            add_info = pd.concat([add_info, new_add_info], axis=1)
    return categ_corr_data, cont_corr_data, add_info


def build_scatter_subplot(fig, x, y, row_num, text, xaxis_text, yaxis_text):
    """
    build_scatter_subplot creates a subplot, placing markers with provided data,
    building and placing trendline. It also places provided text as a subplot title,
    and writes corresponding axis texts.

    :param fig: plotly Figure
    :param x: array-like object with the data along x-axis
    :param y: array-like object with the data along y-axis
    :param row_num: number of the figure, defines the method to set caption
    :param text: caption, that'll be placed for the figure
    :param xaxis_text: x-axis label
    :param yaxis_text: y-axis label
    """

    # Place provided data
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10)), row=row_num, col=1)
    # Approximate linear function
    res = sm.OLS(y, sm.add_constant(x)).fit().fittedvalues
    # Place trendline
    fig.add_trace(go.Scatter(x=x, y=res, mode="lines"), row=row_num, col=1)
    # Place subplot (for the first subplot, we need to refer to the title of the figure itself) title
    if row_num == 1:
        fig.update_layout(title={"text": text})
    else:
        fig.layout.annotations[row_num - 2].update(text=text)
    # Place axis text
    fig.update_xaxes(title_text=xaxis_text, row=row_num, col=1)
    fig.update_yaxes(title_text=yaxis_text, row=row_num, col=1)


def build_scatter_plots(dep_vars, cont_corr_data, continous, combined_data, drop_outliers=True, tps=False,
                        replacement=0.01):
    """
    build_scatter_plots creates several scatter plots over the provided data. Used for continous data

    :param dep_vars: pandas DataFrame with dependent variables
    :param cont_corr_data: pandas DataFrame with correlation data of the continous variables
    :param continous: pandas DataFrame with continous variables
    :param combined_data: pandas DataFrame with both the independent and dependent variables
    :param drop_outliers: boolean variable, setting whether to remove outliers among data
    :param tps: boolean variable, modyfing the logic for the 'per student' mode
    :param replacement: float, zeros in data will be replaced with it
    """

    for i, column in enumerate(dep_vars.columns):
        # Calculating number of graphs
        cont_cols = continous.columns.to_list()
        if i != len(dep_vars.columns) - 1 and not tps:
            cont_cols = cont_cols + dep_vars.columns.to_list()[i + 1:]
        if not tps:
            num_rows = len(cont_cols) + 2 * len(continous.columns) + 2 * len(dep_vars.columns.to_list()[i + 1:])
        else:
            num_rows = 3 * len(cont_cols)
        fig = make_subplots(rows=num_rows, subplot_titles=range(num_rows))
        row_num = 1
        for value in cont_cols:
            # Getting data
            scatter_data = pd.DataFrame({"x": combined_data[value],
                                         "y": combined_data[column]})
            if drop_outliers:
                # Apply 3 sigma rule
                scatter_data = scatter_data.loc[rem_outliers(scatter_data["x"])]
                scatter_data = scatter_data.loc[rem_outliers(scatter_data["y"])]
            # Replace zeros with little values and create logarithmic variation
            scatter_data["y"] = scatter_data["y"].apply(lambda x: replacement if x == 0 else x)
            log_y = scatter_data["y"].apply(lambda x: log(x))
            log_x = scatter_data["x"].apply(lambda x: replacement if x == 0 else x).apply(lambda x: log(x))

            # Creating graph on initial data, stating first pearson p-value, kendall p-value and sample size
            p_value_pearson = cont_corr_data.loc[f"pearson_{column}", value]
            p_value_kendall = cont_corr_data.loc[f"kendall_{column}", value]
            sample_size = int(cont_corr_data.loc[f"sample_size_{column}", value])
            text = f"pearson: {p_value_pearson:.4f}, kendall: {p_value_kendall:.4f}, sample: {sample_size}"
            build_scatter_subplot(fig, scatter_data["x"], scatter_data["y"], row_num, text, value, column)
            row_num = row_num + 1

            # Creating graph on log data, stating second pearson p-value, kendall p-value and sample size
            p_value_pearson = cont_corr_data.loc[f"log_pearson_{column}", value]
            p_value_kendall = cont_corr_data.loc[f"kendall_{column}", value]
            sample_size = int(cont_corr_data.loc[f"sample_size_{column}", value])
            text = f"log({column}) pearson: {p_value_pearson:.4f}, kendall: {p_value_kendall:.4f}, " \
                   f"sample: {sample_size}"
            build_scatter_subplot(fig, scatter_data["x"], log_y, row_num, text, value, column)
            row_num = row_num + 1

            # Creating graph on log indep variable, stating first pearson p-value, kendall p-value and sample size
            p_value_pearson = cont_corr_data.loc[f"log({value})_{column}_pearson", value]
            sample_size = int(cont_corr_data.loc[f"sample_size_{column}", value])
            text = f"log({value}) pearson: {p_value_pearson:.4f}, sample: {sample_size}"
            build_scatter_subplot(fig, log_x, scatter_data["y"], row_num, text, value, column)
            row_num = row_num + 1
        if not tps:
            fig.update_layout(height=4800, width=600)
        else:
            fig.update_layout(height=600, width=600)
        fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
        fig.write_html(f'figures/scatter_{column}_{time.time()}.html', auto_open=True)


def build_box_plots(dep_vars, categorical, combined_data, categ_corr_data, drop_outliers=True):
    """
    build_scatter_plots creates several box plots over provided data. Used for categorical variables

    :param dep_vars: pandas DataFrame with dependent variables
    :param categorical: pandas DataFrame with categorical variables
    :param combined_data: pandas DataFrame with both independent and dependent variables
    :param categ_corr_data: pandas DataFrame with correlation data of categorical variables
    :param drop_outliers: boolean variable, setting whether to remove outliers among data
    """

    for column in dep_vars.columns:
        num_rows = len(params.t_categorical) + len(params.anova_categorical)
        fig = make_subplots(rows=num_rows, cols=1, subplot_titles=[i for i in range(num_rows)])
        row_num = 1
        for value in categorical.columns:
            box_data = pd.DataFrame({"Cat": combined_data[value],
                                     "Var": combined_data[column]})
            if drop_outliers:
                var_mean = box_data["Var"].mean()
                var_std = box_data["Var"].std()
                var_min = var_mean - 3 * var_std
                var_max = var_mean + 3 * var_std
                box_data = box_data.loc[(var_min < box_data["Var"]) &
                                        (box_data["Var"] < var_max)]
            # Calculating medians and sorting the list
            box_data = box_data.set_index(pd.Series([i for i in range(len(box_data))]))
            box_data.dropna(inplace=True)
            sorted_med = box_data.loc[:, ["Cat", "Var"]].groupby(["Cat"]).median().sort_values(by="Var")
            fig.add_trace(go.Box(y=box_data["Var"],
                                 x=box_data["Cat"]),
                          row=row_num,
                          col=1)
            fig.update_xaxes(title_text=column, row=row_num, col=1, type="category",
                             categoryorder="array", categoryarray=sorted_med.index)
            fig.update_yaxes(title_text=value, row=row_num, col=1)
            fig.update_layout(showlegend=False)
            p_value = categ_corr_data.loc[value, column]
            power = categ_corr_data.loc[value, f'{column}_power']
            effect_size = categ_corr_data.loc[value, f'{column}_effect_size']
            text = f"{p_value:.4f}, power: {power:.4f}, effect_size: {effect_size:.4f}, {len(box_data)}"
            if row_num == 1:
                fig.update_layout(title={"text": text})
            else:
                fig.layout.annotations[row_num - 2].update(text=text)
            row_num = row_num + 1
        fig.update_layout(height=3600, width=600, plot_bgcolor='rgba(0,0,0,0)')
        fig.write_html(f'figures/box_{column}_{time.time()}.html', auto_open=True)
        # fig.show()
