import pandas as pd
import time
from data_gen.var_lib import *
from data_gen.Features_pass import Pass_features
import tushare as ts
import psutil
from data_gen import Features_call_create
from read_data import *


class CleanData(object):

    def __init__(self):
        self.startdate = startdate  # 如果是日频的这里 是否只需要到1年即可 这是我考虑的 后面的因子大部分都是基于此添加的
        # self.mydb = db
        self.code = code
        self.label_str = label_str
        self.index_daily = pd.DataFrame()
        self.factortable = read_all(file_path)
        self.holc = self.factortable[['high', 'open', 'low', 'close']].copy()

    def get_index(self):
        pro = ts.pro_api()
        df = pro.index_daily(ts_code=self.code, strat_date=self.startdate, end_date=self.enddate)
        df.trade_date = df.trade_date.apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
        df = df.rename(columns={"ts_code": "code", 'tradeDate': 'tradeDate', 'pct_chg': 'index_daily'})
        df = df[['tradeDate', 'index_daily']]
        self.index_daily = df

    def call_features(self):

        self.factortable = Features_call_create.call_features(self.mydb, self.stockcode, self.startdate, self.enddate,
                                                              self.newprice)
        del self.newprice

    def build_features(self):

        # 不允许使用 shift(-1), 不允许用到未来数据， pct_change（）这些只能和shift(1)搭配使用，数据排列是从早到晚
        # 构建你的特征 如果是需要输入多行数据做移动窗口的请使用这个 并自己构建函数 返回的data记得自己构建列名

        # def func(Series):
        #     array = Series.mean()
        #     return array
        #
        # features = ['open']
        # window = 10
        # self.factortable = Tools(self.factortable, func, window, features).building()
        # print('features_building_done')

        self.factortable = Features_call_create.build_features(self.factortable)

    def label_select(self):
        if self.label_str == 1:
            self.factortable['label'] = self.factortable.groupby(level='code').close.pct_change().groupby(
                level='code').shift(-1)
        elif self.label_str == 0:
            self.factortable['label'] = self.factortable.groupby(level='code').open.pct_change().groupby(
                level='code').shift(-2)
        else:
            self.factortable['label'] = (self.factortable.groupby(level='code').close.shift(
                -2) - self.factortable.groupby(level='code').open.shift(
                -1)) / self.factortable.groupby(level='code').open.shift(-1)
        print('label_selecting_done')

    def creat_pass(self):  # 输入的资料需要从早到晚的时间顺序
        features = self.factortable.columns
        # features = self.factortable.columns.difference(['','',''])
        self.factortable = Pass_features(self.factortable, past_days, features).cre_past()
        print('creat_pass_done')

    def del_nan(self):
        # self.factortable.fillna(method='ffill', inplace=True)
        self.factortable.groupby(level='code').fillna(method='bfill', inplace=True)
        self.factortable.dropna(inplace=True)
        print('nan_deleting_done')

    def stdlize_data(self):
        def _factorFiterAndNormize(factordf):
            zscore = (factordf - factordf.median()) / (factordf - factordf.median()).abs().median()
            zscore[zscore > 3] = 3
            zscore[zscore < -3] = -3
            factordf = zscore * (factordf - factordf.median()).abs().median() + factordf.median()
            factordf = (factordf - factordf.mean()) / factordf.std()
            return factordf

        def _factorFiterAndNormize_(data):
            std = data.std()
            mean = data.mean()
            std = std.replace(0, 1)
            data.iloc[:split_test] = (data.iloc[:split_test] - mean) / std
            data.iloc[split_test:] = (data.iloc[split_test:] - mean) / std
            return data

        def get_mean_std(data):
            new_data = pd.DataFrame()
            data['standard'] = data.std().values
            data['mean'] = data.mean().values
            return pd.Series(data)

        info = psutil.virtual_memory()
        if info.percent > 0.5:
            times = int(info.percent / (1 - info.percent))
        else:
            times = 1

        split_test = int(self.factortable.shape[0] * train_rate)

        # 计算训练集的分布数据
        for i in range(times):
            temp_split_pre = split_test * (i / times)
            temp_split_aft = split_test * ((i + 1) / times)

        self.factortable.iloc[:split_test, :-1] = self.factortable.iloc[:split_test, :-1].groupby(level='code').apply(
            _factorFiterAndNormize_)
        self.factortable.iloc[split_test:, :-1] = self.factortable.iloc[split_test:, :-1].groupby(level='code').apply(
            _factorFiterAndNormize_)
        self.factortable.iloc[:split_test, :-1] = self.factortable.iloc[:split_test, :-1].groupby(
            level='tradeDate').apply(_factorFiterAndNormize_)
        self.factortable.iloc[split_test:, :-1] = self.factortable.iloc[split_test:, :-1].groupby(
            level='tradeDate').apply(_factorFiterAndNormize_)
        print('data_standardizing_done')

    def main(self):
        # self.get_code()
        # self.get_data()
        # self.clean_data()
        # self.call_features()
        self.build_features()
        self.creat_pass()
        print(self.factortable.shape)
        self.label_select()
        # self.stdlize_data()
        self.del_nan()
        print('All_tasks_in_Features_building_done')
        return self.factortable.astype('float32')
