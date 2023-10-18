import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from utils import *

warnings.filterwarnings(action='ignore')


class DataBinningSummary:
    def __init__(self, trainset, testset, quantiles=[0.2, 0.4, 0.6, 0.8],
                 special_values=[-99999999]):
        trainset['data_type'] = 'TRAIN'
        testset['data_type'] = 'TEST'
        self.data = pd.concat([trainset, testset]).drop(columns=data_keys)

        self.categorical_variables = list(
            set(self.data.select_dtypes(include=['object']).columns) - set(['PERF', 'data_type']))
        self.numerical_variables = list(
            set(self.data.select_dtypes(exclude=['object']).columns) - set(['PERF', 'data_type']))

        self.quantiles = quantiles
        self.special_values = special_values

        self.binning_summary_categorical_mon = pd.DataFrame()
        self.binning_summary_numerical_mon = pd.DataFrame()
        self.quantile_value_lists = {}
        self.binning_summary_all_time = {}
        self.special_values = []

    def add_special_value(self, value_list, special_value_list, idx):
        return value_list[:idx] + special_value_list + value_list[idx:]

    def calculate_psi(self, df, base, compare):
        return (df[base] - df[compare]) * np.log(df[base] / df[compare])

    def calculate_iv(self):
        return (self.binning_summary["good_ratio"] - self.binning_summary["bad_ratio"]) * np.log(
            self.binning_summary["good_ratio"] / self.binning_summary["bad_ratio"])

    def make_categorical_binning_summary(self):
        for variable in self.categorical_variables:
            self.binning_data = self.data[[variable, 'PERF']]
            self.binning_summary = self.binning_data.groupby(by=variable).size().reset_index()
            self.binning_summary.rename(columns={0: 'count'}, inplace=True)
            self.binning_summary.rename(columns={variable: 'bin', 0: 'count'}, inplace=True)
            self.bad_data = self.binning_data.groupby(by=variable)['PERF'].sum().reset_index()
            self.bad_data.rename(columns={variable: 'bin', 'PERF': 'bad'}, inplace=True)
            self.binning_summary = pd.merge(self.binning_summary, self.bad_data, how='left', left_on='bin',
                                            right_on='bin')
            self.binning_summary["good"] = self.binning_summary["count"] - self.binning_summary["bad"]
            self.binning_summary["cumulative_bad"] = self.binning_summary["bad"].cumsum()
            self.binning_summary["cumulative_good"] = self.binning_summary["good"].cumsum()
            self.binning_summary["total_ratio"] = self.binning_summary["count"] / self.binning_summary["count"].sum()
            self.binning_summary["bad_rate"] = self.binning_summary["bad"] / self.binning_summary["count"]
            self.binning_summary["bad_ratio"] = self.binning_summary["bad"] / self.binning_summary["bad"].sum()
            self.binning_summary["good_ratio"] = self.binning_summary["good"] / self.binning_summary["good"].sum()
            self.binning_summary["cumulative_bad_ratio"] = self.binning_summary["bad_ratio"].cumsum()
            self.binning_summary["cumulative_good_ratio"] = self.binning_summary["good_ratio"].cumsum()
            self.binning_summary["IV_1"] = self.calculate_iv()
            self.binning_summary = self.binning_summary.replace(np.inf, 0)
            self.binning_summary["IV"] = self.binning_summary["IV_1"].sum()
            self.binning_summary.insert(0, "Variable", variable)
            self.binning_summary.insert(2, "bin_class", self.binning_summary['bin'])
            self.binning_summary_categorical_mon = pd.concat(
                [self.binning_summary_categorical_mon, self.binning_summary], ignore_index=True)

    def make_quantile(self):
        for variable in self.numerical_variables:
            target = (self.data.groupby(by=variable).size() / len(self.data)).reset_index()
            target.rename(columns={0: variable + "_" + 'ratio'}, inplace=True)
            target[f'{variable}_cumulative_ratio'] = target.cumsum()[f'{variable}_ratio']
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

    def make_numerical_binning_summary(self):
        self.make_quantile()
        for variable in self.numerical_variables:
            self.binning_data = self.data[[variable, 'PERF']]
            self.binning_data["bin"] = pd.cut(self.data[variable], self.quantile_value_lists[variable], labels=False)
            self.binning_summary = self.binning_data.groupby(by="bin").size().reset_index()
            self.binning_summary.rename(columns={0: 'count'}, inplace=True)
            self.binning_summary["bad"] = self.binning_data.groupby(["bin"])['PERF'].sum()
            self.binning_summary["good"] = self.binning_summary["count"] - self.binning_summary["bad"]
            self.binning_summary["cumulative_bad"] = self.binning_summary["bad"].cumsum()
            self.binning_summary["cumulative_good"] = self.binning_summary["good"].cumsum()
            self.binning_summary["total_ratio"] = self.binning_summary["count"] / self.binning_summary["count"].sum()
            self.binning_summary["bad_rate"] = self.binning_summary["bad"] / self.binning_summary["count"]
            self.binning_summary["bad_ratio"] = self.binning_summary["bad"] / self.binning_summary["bad"].sum()
            self.binning_summary["good_ratio"] = self.binning_summary["good"] / self.binning_summary["good"].sum()
            self.binning_summary["cumulative_bad_ratio"] = self.binning_summary["bad_ratio"].cumsum()
            self.binning_summary["cumulative_good_ratio"] = self.binning_summary["good_ratio"].cumsum()
            self.binning_summary["IV_1"] = self.calculate_iv()
            self.binning_summary = self.binning_summary.replace(np.inf, 0)
            self.binning_summary["IV"] = self.binning_summary["IV_1"].sum()
            self.binning_summary.insert(0, "Variable", variable)

            bin_class = []
            for i in range(len(self.quantile_value_lists[variable]) - 1):
                bin_class += [self.quantile_value_lists[variable][i:i + 2]]
            self.binning_summary.insert(2, "bin_class", bin_class)
            self.binning_summary_numerical_mon = pd.concat([self.binning_summary_numerical_mon, self.binning_summary],
                                                           ignore_index=True)

    def make_total_binning_summary(self, data_type='TRAIN'):
        if len(self.binning_summary_numerical_mon) != 0 and len(self.binning_summary_categorical_mon) != 0:
            self.binning_summary_all_time[data_type] = pd.concat(
                [self.binning_summary_numerical_mon, self.binning_summary_categorical_mon], ignore_index=True)
        elif len(self.binning_summary_categorical_mon) == 0:
            self.binning_summary_all_time[data_type] = self.binning_summary_numerical_mon
        elif len(self.binning_summary_numerical_mon) == 0:
            self.binning_summary_all_time[data_type] = self.binning_summary_categorical_mon

    def save_binning_summary(self, savename=None):
        if not os.path.exists(ivpsi_path):
            os.makedirs(ivpsi_path)
        # print(self.binning_summary_all_time)

class VariableIvPsi(DataBinningSummary):
    def __init__(self, trainset, testset, binning_summary_all_time):
        super().__init__(trainset, testset)
        variable_iv_all = binning_summary_all_time['TRAIN']['Variable'].unique()
        self.variable_iv_all = pd.DataFrame(variable_iv_all, columns=['Variable'])
        self.variable_bin = binning_summary_all_time['TRAIN'][['Variable', 'bin']]
        self.variable_psi_all = self.variable_bin
        self.variable_psi_f = self.variable_iv_all
        self.binning_summary_all_time = binning_summary_all_time

    def calculate_variable_iv(self, data_type='TRAIN'):
        variable_iv = \
        self.binning_summary_all_time[data_type].groupby(self.binning_summary_all_time[data_type]['Variable'])[
            'IV'].max().reset_index()
        variable_iv.rename(columns={'IV': 'IV' + "_" + data_type}, inplace=True)
        self.variable_iv_all = pd.merge(self.variable_iv_all, variable_iv, how='left', left_on='Variable',
                                        right_on='Variable')

    def calculate_variable_psi(self, data_type='TRAIN'):
        variable_psi = self.binning_summary_all_time[data_type][['Variable', 'bin', 'total_ratio']]
        variable_psi.rename(columns={"total_ratio": "total_ratio" + "_" + data_type}, inplace=True)
        self.variable_psi_all = pd.merge(self.variable_psi_all, variable_psi, how='left', left_on=['Variable', 'bin'],
                                         right_on=['Variable', 'bin'])
        self.variable_psi_all['psi' + "_" + data_type] = self.calculate_psi(self.variable_psi_all, "total_ratio_TRAIN",
                                                                            "total_ratio" + "_" + data_type)
        self.variable_psi_all = self.variable_psi_all.fillna(0)
        variable_psi_all_1 = self.variable_psi_all.groupby(self.variable_psi_all['Variable'])[
            'psi' + "_" + data_type].sum().reset_index()
        variable_psi_all_1.rename(columns={'psi' + "_" + data_type: 'psi' + "_" + data_type}, inplace=True)
        self.variable_psi_f = pd.merge(self.variable_psi_f, variable_psi_all_1, how='left', left_on='Variable',
                                       right_on='Variable')

    def save_iv_psi(self, savename=None):
        if not os.path.exists(ivpsi_path):
            os.makedirs(ivpsi_path)
        if savename is None:
            savename = datetime.now().strftime('%y%m%d_%H%M%S')
        self.variable_iv_all.to_csv(f'{ivpsi_path}/iv_{savename}.csv')
        self.variable_psi_f.to_csv(f'{ivpsi_path}/psi_{savename}.csv')


if __name__ == "__main__":
    train_data = pd.read_csv(f'{data_path}/sample_data_202211.csv')
    test_data = pd.read_csv(f'{data_path}/sample_data_202304.csv')

    binning_summary = DataBinningSummary(train_data, test_data)

    # Perform the binning summary
    binning_summary.make_categorical_binning_summary()
    binning_summary.make_numerical_binning_summary()
    binning_summary.make_total_binning_summary()
    binning_summary.save_binning_summary()

    # Calculate IV and PSI
    variable_iv_psi = VariableIvPsi(train_data, test_data, binning_summary.binning_summary_all_time)
    variable_iv_psi.calculate_variable_iv()
    variable_iv_psi.calculate_variable_psi()
    variable_iv_psi.save_iv_psi()