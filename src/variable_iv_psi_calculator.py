import pandas as pd
import numpy as np

class DataBinningSummary:
    def __init__(self):
        self.binning_summary_categorical_mon = pd.DataFrame()
        self.binning_summary_numerical_mon = pd.DataFrame()
        self.quantile_value_lists = {}
        self.binning_summary_categorical_time = {}
        self.binning_summary_numerical_time = {}
        self.binning_summary_all_time = {}
        self.special_values = []

    def add_special_value(self, value_list, special_value_list, idx):
        return value_list[:idx] + special_value_list + value_list[idx:]

    def calculate_psi(self, data, base, compare):
        return (data[base] - data[compare]) * np.log(data[base] / data[compare])

    def calculate_iv(self, data, good_ratio_col, bad_ratio_col):
        return (data[good_ratio_col] - data[bad_ratio_col]) * np.log(data[good_ratio_col] / data[bad_ratio_col])

    def make_categorical_binning_summary(self, variable, bad_column, month):
        self.binning_data = self.data[[variable, bad_column]]
        self.binning_summary = self.binning_data.groupby(by=variable).size().reset_index()
        self.binning_summary.rename(columns={0: 'count'}, inplace=True)
        self.binning_summary.rename(columns={variable: 'bin', 0: 'count'}, inplace=True)
        self.bad_data = self.binning_data.groupby(by=variable)[bad_column].sum().reset_index()
        self.bad_data.rename(columns={variable: 'bin', bad_column: 'bad'}, inplace=True)
        self.binning_summary = pd.merge(self.binning_summary, self.bad_data, how='left', left_on='bin', right_on='bin')
        self.binning_summary["good"] = self.binning_summary["count"] - self.binning_summary["bad"]
        self.binning_summary["cumulative_bad"] = self.binning_summary["bad"].cumsum()
        self.binning_summary["cumulative_good"] = self.binning_summary["good"].cumsum()
        self.binning_summary["total_ratio"] = self.binning_summary["count"] / self.binning_summary["count"].sum()
        self.binning_summary["bad_rate"] = self.binning_summary["bad"] / self.binning_summary["count"]
        self.binning_summary["bad_ratio"] = self.binning_summary["bad"] / self.binning_summary["bad"].sum()
        self.binning_summary["good_ratio"] = self.binning_summary["good"] / self.binning_summary["good"].sum()
        self.binning_summary["cumulative_bad_ratio"] = self.binning_summary["bad_ratio"].cumsum()
        self.binning_summary["cumulative_good_ratio"] = self.binning_summary["good_ratio"].cumsum()
        self.binning_summary["IV_1"] = self.calculate_iv(self.binning_summary, "good_ratio", "bad_ratio")
        self.binning_summary = self.binning_summary.replace(np.inf, 0)
        self.binning_summary["IV"] = self.binning_summary["IV_1"].sum()
        self.binning_summary.insert(0, "Variable", variable)
        self.binning_summary.insert(2, "bin_class", self.binning_summary['bin'])
        self.binning_summary_categorical_mon = pd.concat([self.binning_summary_categorical_mon, self.binning_summary], ignore_index=True)
        self.binning_summary_categorical_time[month] = self.binning_summary_categorical_mon

    def make_quantile(self, variable):
        target = (self.data.groupby(by=variable).size() / len(self.data)).reset_index()
        target.rename(columns={0: variable + "_" + 'ratio'}, inplace=True)
        target[variable + "_" + 'cumulative_ratio'] = target.cumsum()[variable + "_" + 'ratio']
        quantile_value_list = []
        quantile_value_list_1 = []
        quantile_value_list_2 = []
        for p, q in enumerate(self.quantiles):
            if len(target.loc[target.iloc[:, 2] <= q]) > 0:
                quantile_target = target.loc[target.iloc[:, 2] <= q]
                quantile_value_list += [quantile_target.iloc[:, 0].max()]
            else:
                pass

        for value in quantile_value_list:
            if value not in quantile_value_list_1:
                quantile_value_list_1.append(value)
        quantile_value_list_1.insert(0, -np.inf)
        quantile_value_list_1.insert(len(quantile_value_list) + 1, np.inf)

        special_values_1 = []
        for i in range(len(self.special_values)):
            if (self.data[[variable]] == self.special_values[i]).sum()[0] > 0:
                special_values_1 += [self.special_values[i]]
            else:
                pass
        quantile_value_list_1 = self.add_special_value(quantile_value_list_1, special_values_1, 1)
        for v in quantile_value_list_1:
            if v not in quantile_value_list_2:
                quantile_value_list_2.append(v)
        self.quantile_value_lists[variable] = quantile_value_list_2

    def make_numerical_binning_summary(self, variable, bad_column, month):
        self.binning_data = self.data[[variable, bad_column]]
        self.binning_data["bin"] = pd.cut(self.data[variable], self.quantile_value_lists[variable], labels=False)
        self.binning_summary = self.binning_data.groupby(by="bin").size().reset_index()
        self.binning_summary.rename(columns={0: 'count'}, inplace=True)
        self.binning_summary["bad"] = self.binning_data.groupby(["bin"])[bad_column].sum()
        self.binning_summary["good"] = self.binning_summary["count"] - self.binning_summary["bad"]
        self.binning_summary["cumulative_bad"] = self.binning_summary["bad"].cumsum()
        self.binning_summary["cumulative_good"] = self.binning_summary["good"].cumsum()
        self.binning_summary["total_ratio"] = self.binning_summary["count"] / self.binning_summary["count"].sum()
        self.binning_summary["bad_rate"] = self.binning_summary["bad"] / self.binning_summary["count"]
        self.binning_summary["bad_ratio"] = self.binning_summary["bad"] / self.binning_summary["bad"].sum()
        self.binning_summary["good_ratio"] = self.binning_summary["good"] / self.binning_summary["good"].sum()
        self.binning_summary["cumulative_bad_ratio"] = self.binning_summary["bad_ratio"].cumsum()
        self.binning_summary["cumulative_good_ratio"] = self.binning_summary["good_ratio"].cumsum()
        self.binning_summary["IV_1"] = self.calculate_iv(self.binning_summary, "good_ratio", "bad_ratio")
        self.binning_summary = self.binning_summary.replace(np.inf, 0)
        self.binning_summary["IV"] = self.binning_summary["IV_1"].sum()
        self.binning_summary.insert(0, "Variable", variable)
        bin_class = []
        for i in range(len(self.quantile_value_lists[variable]) - 1):
            bin_class.insert(i, self.quantile_value_lists[variable][i:i + 2])
        self.binning_summary.insert(2, "bin_class", bin_class)
        self.binning_summary_numerical_mon = pd.concat([self.binning_summary_numerical_mon, self.binning_summary], ignore_index=True)
        self.binning_summary_numerical_time[month] = self.binning_summary_numerical_mon

    def make_total_binning_summary(self, month):
        if len(self.binning_summary_numerical_mon) != 0 and len(self.binning_summary_categorical_mon) != 0:
            self.binning_summary_all_time[month] = pd.concat([self.binning_summary_numerical_mon, self.binning_summary_categorical_mon], ignore_index=True)
        elif len(self.binning_summary_categorical_mon) == 0:
            self.binning_summary_all_time[month] = self.binning_summary_numerical_mon
        elif len(self.binning_summary_numerical_mon) == 0:
            self.binning_summary_all_time[month] = self.binning_summary_categorical_mon

class VariableIvPsi(DataBinningSummary):
    def __init__(self, binning_summary_all_time):
        super().__init__()
        variable_iv_all = binning_summary_all_time['01']['Variable'].unique()
        self.variable_iv_all = pd.DataFrame(variable_iv_all, columns=['Variable'])
        self.variable_bin = binning_summary_all_time['01'][['Variable', 'bin']]
        self.variable_psi_all = self.variable_bin
        self.variable_psi_f = self.variable_iv_all
        self.binning_summary_all_time = binning_summary_all_time

    def calculate_variable_iv(self, month):
        variable_iv = self.binning_summary_all_time[month].groupby(self.binning_summary_all_time[month]['Variable'])['IV'].max().reset_index()
        variable_iv.rename(columns={'IV': 'IV' + "_" + month}, inplace=True)
        self.variable_iv_all = pd.merge(self.variable_iv_all, variable_iv, how='left', left_on='Variable', right_on='Variable')

    def calculate_variable_psi(self, month):
        variable_psi = self.binning_summary_all_time[month][['Variable', 'bin', 'total_ratio']]
        variable_psi.rename(columns={"total_ratio": "total_ratio" + "_" + month}, inplace=True)
        self.variable_psi_all = pd.merge(self.variable_psi_all, variable_psi, how='left', left_on=['Variable', 'bin'], right_on=['Variable', 'bin'])
        self.variable_psi_all['psi' + "_" + month] = self.calculate_psi(self.variable_psi_all, "total_ratio_01", "total_ratio" + "_" + month)
        self.variable_psi_all = self.variable_psi_all.fillna(0)
        variable_psi_all_1 = self.variable_psi_all.groupby(self.variable_psi_all['Variable'])['psi' + "_" + month].sum().reset_index()
        variable_psi_all_1.rename(columns={'psi' + "_" + month: 'psi' + "_" + month}, inplace=True)
        self.variable_psi_f = pd.merge(self.variable_psi_f, variable_psi_all_1, how='left', left_on='Variable', right_on='Variable')

# Usage
data = pd.read_csv("data.csv")
binning_summary = DataBinningSummary()
binning_summary.data = data
binning_summary.quantiles = [0.2, 0.4, 0.6, 0.8]
binning_summary.special_values = [99, 88]

# Perform the binning summary
binning_summary.make_categorical_binning_summary("CategoricalVar", "Bad", "01")
binning_summary.make_quantile("NumericalVar")
binning_summary.make_numerical_binning_summary("NumericalVar", "Bad", "01")
binning_summary.make_total_binning_summary("01")

# Calculate IV and PSI
variable_iv_psi = VariableIvPsi(binning_summary.binning_summary_all_time)
variable_iv_psi.calculate_variable_iv("01")
variable_iv_psi.calculate_variable_psi("01")
