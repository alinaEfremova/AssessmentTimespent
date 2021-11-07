import func as f
import params as params
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import re
from numpy import nan, log

# setting global variables
alpha = 0.05
drop_outliers = True
replacement = 0.01

# parsing csv file
data = f.parse_answer("teacherResponses.csv", False).set_index("id")

# cleansing dependant variables
dep_vars = f.content_substitution(data[params.dependant])

# cleansing independent variables
indep_vars = f.content_substitution(data[params.independent])
indep_vars['assignments_volume'] = indep_vars['volume'] / (indep_vars['NGA_num'] + indep_vars['GA_num'])
indep_vars = indep_vars.drop(columns="volume")
indep_vars['assign_num'] = indep_vars['NGA_num'] + indep_vars['GA_num']
indep_vars['TA'] = indep_vars['TA_FTE'] * indep_vars['TA_help']
indep_vars['TA'] = indep_vars['TA'].fillna(0)
indep_vars['groups'] = indep_vars['groups'].apply(lambda x: 1 if x == 0 else x)
indep_vars['test_group'] = indep_vars['students_num'] / indep_vars['groups']
indep_vars['TA_part'] = 100 * indep_vars['TA'] / (
        dep_vars['prep_time'] + dep_vars['grad_time'] + dep_vars['assess_time'] + dep_vars['feedback_time'])

# separating all the variables by type
continous = pd.DataFrame(indep_vars, columns=[col for col in indep_vars.columns if col in params.continuous])
categorical = pd.concat([pd.DataFrame(indep_vars, columns=[col for col in indep_vars.columns if
                                                           col in params.t_categorical]),
                         pd.DataFrame(indep_vars, columns=[col for col in indep_vars.columns if
                                                           col in params.anova_categorical])], axis=1)

categ_corr_data, cont_corr_data, add_info = f.separate_correlations(categorical, continous, dep_vars,
                                                                    drop_outliers, alpha=alpha,
                                                                    replacement=replacement)

# saving the correlation data
cont_corr_data.transpose().to_excel("cont_corr_data.xlsx")
cont_corr_data = cont_corr_data.transpose()
categ_corr_data.transpose().to_excel("categ_corr_data.xlsx")
add_info.transpose().to_excel("add_info.xlsx")

comb = pd.concat([indep_vars, dep_vars], axis=1)
new_df_cols = ["variables"]
for dep in dep_vars.columns:
    new_df_cols += [f"{dep}_size", f"{dep}_reg", f"{dep}_rho", f"{dep}_p"]
new_df = pd.DataFrame(columns=new_df_cols)
new_df["variables"] = params.continuous + params.t_categorical + params.anova_categorical
new_df.to_excel("new_df.xlsx")

# calculating regression
for dep in dep_vars.columns:
    for indep in new_df["variables"]:
        comb_without_outliers = pd.DataFrame()
        r_squared = nan
        if indep in params.continuous:
            comb_without_outliers = comb.loc[set(f.rem_outliers(comb[dep])) &
                                             set(f.rem_outliers(comb[indep]))].dropna(subset=[dep, indep])
            expr = f"{dep} ~ {indep}"
            # expr = f"{dep} ~ {indep} + I({indep}**2)"
            r_squared, model = f.simple_cat_reg(expr, comb_without_outliers)
            new_df.loc[new_df["variables"] == indep, f"{dep}_rho"] = cont_corr_data.loc[f"pearson_rho_{dep}", indep]
            new_df.loc[new_df["variables"] == indep, f"{dep}_p"] = cont_corr_data.loc[f"pearson_{dep}", indep]
        elif indep in params.t_categorical:
            comb_without_outliers = comb.loc[f.rem_outliers(comb[dep])].dropna(subset=[dep, indep])
            expr = f"{dep} ~ C({indep})"
            # expr = f"{dep} ~ {indep} + I(C({indep})**2)"
            r_squared, model = f.simple_cat_reg(expr, comb_without_outliers)
            new_df.loc[new_df["variables"] == indep, f"{dep}_rho"] = nan
            new_df.loc[new_df["variables"] == indep, f"{dep}_p"] = categ_corr_data.loc[indep, dep]
        elif indep in params.anova_categorical:
            comb_without_outliers = comb.loc[f.rem_outliers(comb[dep])].dropna(subset=[dep, indep])
            expr = f"{dep} ~ C({indep})"
            # expr = f"{dep} ~ {indep} + I({indep}**2)"
            r_squared, model = f.simple_cat_reg(expr, comb_without_outliers)
            new_df.loc[new_df["variables"] == indep, f"{dep}_rho"] = nan
            new_df.loc[new_df["variables"] == indep, f"{dep}_p"] = categ_corr_data.loc[indep, dep]
        new_df.loc[new_df["variables"] == indep, f"{dep}_size"] = len(comb_without_outliers)
        new_df.loc[new_df["variables"] == indep, f"{dep}_reg"] = r_squared
new_df.to_excel("new_df.xlsx")
cols = [v for v in new_df.columns if '_time_reg' in v]
ax = sns.heatmap(new_df[cols].astype(float), annot=True, yticklabels=new_df["variables"])
plt.show()

# calculating intercorrelation
intercorr = pd.DataFrame(columns=["variables"] + new_df["variables"].to_list())
intercorr["variables"] = params.continuous + params.t_categorical + params.anova_categorical
for indep in new_df["variables"]:
    for dep in new_df["variables"]:
        if indep == dep:
            intercorr.loc[intercorr["variables"] == indep, dep] = nan
        elif indep in params.continuous and dep in params.continuous:
            comb_without_outliers = comb.loc[set(f.rem_outliers(comb[indep])) &
                                             set(f.rem_outliers(comb[dep]))].dropna(subset=[dep, indep])
            intercorr.loc[intercorr["variables"] == indep, dep] = \
                f.continous_correlation(pd.DataFrame(comb_without_outliers[indep]),
                                        comb_without_outliers[dep]).loc[indep, "pearson"]
        elif (indep in params.t_categorical or indep in params.anova_categorical) and (
                dep in params.t_categorical or dep in params.anova_categorical):
            intercorr.loc[intercorr["variables"] == indep, dep] = nan
        elif indep in params.t_categorical and dep in params.continuous:
            comb_without_outliers = comb.loc[f.rem_outliers(comb[dep])].dropna(subset=[dep])
            intercorr.loc[intercorr["variables"] == indep, dep] = \
                f.categorical_correlation(pd.DataFrame(comb_without_outliers[indep]),
                                          comb_without_outliers[dep]).loc[indep, "time"]
        elif indep in params.anova_categorical and dep in params.continuous:
            comb_without_outliers = comb.loc[f.rem_outliers(comb[dep])].dropna(subset=[dep])
            intercorr.loc[intercorr["variables"] == indep, dep], corr, power, effect_size = \
                f.calc_anova(comb_without_outliers[indep], comb_without_outliers[dep])
intercorr.to_excel("intercorr.xlsx")

# calculating per one student
times_per_student = pd.DataFrame(index=dep_vars.index,
                                 columns=dep_vars.columns.to_list() + ["students_num"])
times_per_student["students_num"] = indep_vars["students_num"]
times_per_student[dep_vars.columns] = dep_vars[dep_vars.columns]
for dep_col in dep_vars.columns:
    times_per_student[dep_col] = times_per_student[dep_col].div(indep_vars["students_num"])

times_per_student.to_excel("times_per_student.xlsx")

# calculating correlation per one student
tps_cat_corr_data, tps_cont_corr_data, add_info = \
    f.separate_correlations(pd.DataFrame(), pd.DataFrame(times_per_student["students_num"]),
                            times_per_student[dep_vars.columns], drop_outliers, alpha=alpha, tps=True,
                            replacement=replacement)
tps_cont_corr_data = tps_cont_corr_data.transpose()
tps_cont_corr_data.to_excel("tps_cont_corr_data.xlsx")

combined_data = pd.merge(indep_vars, dep_vars, on='id')

# Scatter plots
f.build_scatter_plots(dep_vars, cont_corr_data, continous, combined_data, drop_outliers, replacement=replacement)

# Box plots
f.build_box_plots(dep_vars, categorical, combined_data, categ_corr_data, drop_outliers)

# Times per student scatter plots
f.build_scatter_plots(times_per_student[dep_vars.columns], tps_cont_corr_data,
                      pd.DataFrame(times_per_student["students_num"]), times_per_student, tps=True,
                      replacement=replacement)

# Sorting continous correlation coefficients
p_values = pd.DataFrame()
for dep in dep_vars.columns:
    pearson_p_values = cont_corr_data.loc[f"pearson_{dep}"]
    log_pearson_p_values = cont_corr_data.loc[f"log_pearson_{dep}"]
    p_values[dep] = cont_corr_data.loc[f"pearson_{dep}", log_pearson_p_values >= pearson_p_values]
    p_values[f"log_{dep}"] = cont_corr_data.loc[f"log_pearson_{dep}", log_pearson_p_values < pearson_p_values]
p_values = p_values.transpose()
p_values = pd.concat([p_values, categ_corr_data.transpose().loc[dep_vars.columns]], axis=1)
p_values.to_excel("p_values.xlsx")

# Splitting p-values
low_p_values = pd.DataFrame(index=p_values.columns.to_list(), columns=p_values.index.to_list())
high_p_values = pd.DataFrame(index=p_values.columns.to_list(), columns=p_values.index.to_list())
for dep in p_values.index.to_list():
    low_p_values[dep] = p_values.loc[dep, p_values.loc[dep] <= alpha]
    high_p_values[dep] = p_values.loc[dep, p_values.loc[dep] > alpha]
low_p_values = low_p_values.dropna(how="all").transpose().dropna(how="all")
high_p_values = high_p_values.dropna(how="all").transpose().dropna(how="all")
low_p_values.to_excel("low_p_values.xlsx")
high_p_values.to_excel("high_p_values.xlsx")

# Obtaining data for analysis
# First comes the low part
low_indep = combined_data.loc[:, low_p_values.columns.to_list()]
low_dep = pd.DataFrame(index=dep_vars.index.to_list())
for el in low_p_values.index.to_list():
    if re.match("log_", el):
        low_dep = pd.concat([low_dep, dep_vars.loc[:, el[4:]].apply(lambda x: replacement if x == 0 else x)
                            .apply(lambda x: log(x))], axis=1)
        low_dep.rename(columns={el[4:]: f"log_{el[4:]}"}, inplace=True)
    else:
        low_dep = pd.concat([low_dep, dep_vars.loc[:, el]], axis=1)
low_combined = pd.concat([low_indep, low_dep], axis=1)
low_combined.to_excel("low_combined.xlsx")

# And the high part
high_indep = combined_data.loc[:, high_p_values.columns.to_list()]
high_dep = pd.DataFrame(index=dep_vars.index.to_list())
for el in high_p_values.index.to_list():
    if re.match("log_", el):
        high_dep = pd.concat([high_dep, dep_vars.loc[:, el[4:]].apply(lambda x: 0.01 if x == 0 else x)
                             .apply(lambda x: log(x))], axis=1)
        high_dep.rename(columns={el[4:]: f"log_{el[4:]}"}, inplace=True)
    elif re.search("_power", el):
        pass
    else:
        high_dep = pd.concat([high_dep, dep_vars.loc[:, el]], axis=1)
# high_dep = dep_vars.loc[:, dep_list]
high_combined = pd.concat([high_indep, high_dep], axis=1)
high_combined.to_excel("high_combined.xlsx")

# calculating regression
regression = pd.DataFrame()
p_value = f.continous_correlation(pd.DataFrame(combined_data["duration"]),
                                  math.e ** (combined_data["students_num"] * 0.0217)
                                  ).loc["duration", "pearson"]
regression.loc["duration", "student_num"] = p_value

p_value = f.continous_correlation(pd.DataFrame(combined_data["duration"]),
                                  log(combined_data["GA_num"])).loc["duration", "pearson"]
regression.loc["duration", "GA_num"] = p_value

p_value = f.categorical_correlation(pd.DataFrame(combined_data["soft_use"]),
                                    combined_data["assign_num"]).loc["soft_use", "time"]
regression.loc["soft_use", "assign_num"] = p_value

p_value, table, power, effect_size = f.calc_anova(combined_data["max_assignment_type"],
                                                  combined_data["assign_num"])
regression.loc["max_assignment_type", "assign_num"] = p_value

p_value, table, power, effect_size = f.calc_anova(combined_data["comp_level"],
                                                  combined_data["duration"])
regression.loc["comp_level", "duration"] = p_value

p_value, table, power, effect_size = f.calc_anova(combined_data["comp_level"],
                                                  math.e ** (combined_data["students_num"] * 0.0217))
regression.loc["comp_level", "students_num"] = p_value

regression.to_excel("regression.xlsx")
