# from backtest.dataframe import TradeSession, PMPandas
# from backtest.base_api import event_generator
# import pandas as pd


# class PMVWAP(PMPandas):
#     """基金经理（策略）的具体实现"""
#     ratio_limit = 0.001

#     def init(self):
#         # 获取市场数据， 粗略计算距离退市的时间
#         data = self.get_mkt_data()
#         data['ndelist'] = data.groupby(
#             'code', group_keys=False)['tradeDate'].apply(
#                 lambda x: pd.Series(range(len(x), 0, -1), index=x.index))

#         # 初始化一个计数的变量
#         self.open_count = 0

#     def after_open(self):
#         if self.open_count % 5 == 0:
#             self.cancel_orders(self.active_orders.query("targetVol > 0"))
#             data = self.get_mkt_frame()
#             data = data.sort_values('settlePrice').head(5).query(
#                 "ndelist > 30")
#             data['ratio_relative'] = data['settlePrice']  # 以价格为成交比例
#             data['code'] = data.index
#             self.send_orders(
#                 data[['code', 'ratio_relative']].copy(),
#                 ratio_to_netasset=0.5,  # 二分之一的现金去交易
#                 vwap_volume_limit_ratiot=self.ratio_limit,
#                 order_type='ratio_vwap',
#                 order_seq=1)  # 交易优先级靠后， 保证有足够的资金释放
#         self.open_count += 1

#     def after_close(self):
#         if (self.open_count + 1) % 5 == 0:
#             self.sell_position(self.position,
#                                order_seq=0)  # 交易优先级靠前， 保证有足够的资金释放

#     def __str__(self):
#         return ("vwap订单测试")


# ts = TradeSession(PMVWAP, start_datetime='2021-01-01',
#                   end_datetime='2021-03-01', markets='stock_daily')
# obs = ts.run()


# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:11:06 2023

@author: houji
"""
import pandas as pd
import numpy as np
from backtest.dataframe import TradeSession, PMPandas
from backtest.base_api import event_generator

# 单因子值,格式为有name的pd.series
fct = pd.read_pickle(r"D:\jupyter_notebook\同步盘\回测需求整理\factor.pickle")
fct = fct.dropna().reset_index()
fct['date'] = pd.to_datetime(fct['date'])
fct.columns = ['date', 'code', 'value']
# not_include
not_include = pd.DataFrame(pd.read_pickle(r"D:\jupyter_notebook\同步盘\回测需求整理\not_include_list.pickle"),
                           columns=['date', 'code'])
not_include['date'] = pd.to_datetime(not_include['date'])

not_include.head()


class PMSimple(PMPandas):
    def init(self):
        mkt = self.get_mkt_obj()
        mkt.buy_fee_ratio = 5e-4 + 5e-4  # 佣金与冲击成本
        mkt.sell_fee_ratio = 5e-4 + 5e-4 + 1e-3  # 佣金与冲击成本与印花税
        mkt.vwap_volume_limit_ratio = 1  # 成交量占当日的成交量上限
        self.week_count = 0  # 用于打印回测进度
        self.add_factor()
        self.send_record = pd.DataFrame()

    def after_close(self):
        # 如果成交量小于1手则取消订单
        active_orders = self.active_orders
        l_long = active_orders['targetVol'] > active_orders['filledVol']
        l_amount = active_orders['targetVol'] - \
            active_orders['filledVol'] < 10  # 一手的量
        self.cancel_orders(active_orders.loc[l_amount & l_long])

    def after_week_end(self):
        # 打印日志
        self.week_count += 1
        if self.week_count % 20 == 0:
            print(self.observer.time, self.balance['netAsset'])

        # 每周末， 取消所有的活跃订单(0号和1号订单代表不会被取消的订单)
        self.cancel_orders(self.active_orders)

        # 计算买单
        # 计算标的之间的相对持仓比例， 如果是等权持仓随便传入一个正数即可
        self.select_secu_tobuy()
        self.order_data['ratio_relative'] = 7.5

        self.send_record = pd.concat((self.send_record, self.order_data))

        # 发送订单
        # target_ratio_VWAP 会根据输入的ratio_relative（标的相对的权重）与ratio_to_netasset（总的买入资金占净资产的比例）
        # 订单策略会根据输入的比例，自动计算目标仓位，并结合当前的持仓进行开、平操作，避免额外的费用
        # 注意，因为发送订单的时候并不知道未来价格波动以及订单是否会被成交，因此给出的结果是近似结果
        self.send_orders(
            # 订单的信息， 需要的字段由 order_type决定， 大部分情况下包含 code，ratio_relative两列 或 code， targetVol两列组合
            self.order_data,
            order_type='target_ratio_vwap',  # 订单算法详见 cb_daily.CBTargetRatioVWAP
            ratio_to_netasset=1,  # 买入的金额占组合净资产的比例， 如果是1则表示满仓买入
            order_seq_short=5,  # 卖单的交易顺序
            order_seq_long=10)  # 买单的交易顺序

    def add_factor(self):
        "初始化的时候把不复合条件的标的删除"
        res = pd.merge(fct, not_include, on=[
                       'date', 'code'], how='outer', indicator=True)
        l = res['_merge'] == 'left_only'
        self.fct = res.loc[l]
        del self.fct['_merge']

    def select_secu_tobuy(self):
        # 选择要持仓的股票
        mkt = self.get_mkt_obj()
        fct_temp = self.get_last_fct()
        l_top20 = fct_temp['value'] < fct_temp['value'].quantile(0.2)  # 小于20%
        l_valide = fct_temp['code'].isin(
            mkt.current_frame.index)  # 有些股票有因子值但是没有行情信息
        self.order_data = fct_temp.loc[l_top20 & l_valide].copy()

    def get_last_fct(self):
        # 获取最新交易日对应的因子值
        # 注意， 这种方法获取的因子值只能收盘后使用（xx日开盘的时候因子值很可能尚未计算出来）
        mkt = self.get_mkt_obj()
        last_trade_date = mkt.trade_date
        return (self.fct.query("date ==@last_trade_date"))


class PMSimple4(PMSimple):
    def after_close(self):
        # 先根据not_include以及买入金额小于15万的条件取消订单，卖出持仓
        super().after_close()

        # 动态调参
        if hasattr(self, 'set_buylist'):

            # 计算当前符合条件的股票
            self.select_secu_tobuy()
            self.set_buylist_now = set(self.order_data['code'])

            # 取消不在当前列表的买单
            active_orders = self.active_orders
            l_buy = active_orders['targetVol'] > active_orders['filledVol']
            l_tocancel = ~active_orders['code'].isin(self.set_buylist_now)
            l_not01 = active_orders['orderSeq'] > 1  # orderSeq=0，1标记永远不会被取消的订单
            self.cancel_orders(active_orders.loc[l_buy & l_not01 & l_tocancel])

            # 卖出不在当前列表的持仓
            l_to_net = ~self.position['code'].isin(self.set_buylist_now)
            data = self.reverse_position(self.position.loc[l_to_net].copy())
            self.send_orders(data, order_type='vwap', order_seq=5)

            # 买入新进的股票
            if len(self.set_buylist_now) > 0:
                set_tobuy = self.set_buylist_now - self.set_buylist
                data = pd.DataFrame(set_tobuy, columns=['code'])
                data['ratio_relative'] = 7.5
                self.send_orders(
                    data,  # 订单的信息， 需要的字段由 order_type决定， 大部分情况下包含 code，ratio_relative两列 或 code， targetVol两列组合
                    order_type='ratio_vwap',  # 订单算法详见 cb_daily.CBTargetRatioVWAP
                    # 买入的金额占组合净资产的比例， 如果是1则表示满仓买入
                    ratio_to_netasset=len(set_tobuy) / \
                    len(self.set_buylist_now),
                    order_seq=10)

            # 更新股票列表
            self.set_buylist = self.set_buylist_now

    def after_week_end(self):
        super().after_week_end()
        self.set_buylist = set(self.order_data['code'])


ts = TradeSession(PMPandas,
                  cash=1e8,
                  start_datetime='2017-12-28',
                  end_datetime='2023-05-01',
                  cb_daily_data_path=r'D:\mypkgs\backtest\data_mock\cbmkt.pickle')
date_weekend = pd.date_range(start='2017', end='2024', freq='w')
events_weekend = event_generator(
    date_weekend.to_pydatetime(), 'week_end', event_seq=5)
ts.add_events(events_weekend)

ts.PortfolioManager = PMSimple
obs = ts.run()


# class PM(PMPandas):

# #     # 回测之前做的数据准备工作
# #     def init(self):
# #         # 增加即将退市股票的标识， 避免买入
# #         data = self.get_mkt_data()
# #         data['ndelist'] = data.groupby('code', group_keys=False)['tradeDate'].apply(
# #             lambda x: pd.Series(range(len(x), 0, -1), index=x.index))

# #     # 开盘后用下单10%的资金买入价格最低10只股票, 以股价为比例
# #     def after_open(self):
# #         data = self.get_mkt_frame()
# #         data = data.sort_values('settlePrice').head(10).query("ndelist > 30")
# #         data['ratio_amount'] = data['settlePrice']
# #         data['code'] = data.index
# #         self.send_orders(
# #             data[['code', 'ratio_amount']],
# #             ratio_asset=0.1,  # 十分之一的现金去交易
# #             order_type='ratio_market',
# #             order_seq=11)  # 交易优先级靠后， 保证有足够的资金释放

# #     # 收盘后下单用10%的资金买入价格最高的10只股票
# #     def after_close(self):

# #         data = self.get_mkt_frame()
# #         data = data.sort_values('settlePrice').tail(10).query("ndelist > 30")
# #         data['ratio_amount'] = data['settlePrice']
# #         data['code'] = data.index
# #         self.send_orders(
# #             data[['code', 'ratio_amount']],
# #             ratio_asset=0.1,  # 十分之一的仓位去交易
# #             order_type='ratio_market',
# #             order_seq=11)  # 交易优先级靠后， 保证有足够的资金释放

# #     # 周末， 先取消所有的订单， 然后卖出所有的持仓
# #     def after_record(self):
# #         print(self.observer.time)
# #         print(self.position.shape[0])
# #         self.cancel_orders(self.active_orders)
# #         self.sell_position(self.position)


# # class PM(PMPandas):
# #     """基金经理（策略）的具体实现"""
# #     ratio_limit = 0.001

# #     def init(self):
# #         # 获取市场数据， 粗略计算距离退市的时间
# #         data = self.get_mkt_data()
# #         data['ndelist'] = data.groupby('code', group_keys=False)['tradeDate'].apply(
# #             lambda x: pd.Series(range(len(x), 0, -1), index=x.index))

# #         # 初始化一个计数的变量
# #         self.open_count = 0
# #         print('-'*10 + '初始化工作完成' + '-'*10)

# #     def after_open(self):
# #         if self.open_count % 5 == 0:
# #             data = self.get_mkt_frame()
# #             data = data.sort_values('settlePrice').head(
# #                 5).query("ndelist > 30")
# #             data['ratio_amount'] = data['settlePrice']  # 以价格为成交比例
# #             data['code'] = data.index
# #             self.send_orders(
# #                 data[['code', 'ratio_amount']],
# #                 ratio_asset=0.5,  # 二分之一的现金去交易
# #                 ratio_limit=self.ratio_limit,
# #                 order_type='VWAP',
# #                 order_seq=1)  # 交易优先级靠后， 保证有足够的资金释放
# #         self.open_count += 1

# #     def after_close(self):
# #         if (self.open_count+1) % 5 == 0:
# #             self.sell_position(self.position,
# #                                order_seq=0)  # 交易优先级靠前， 保证有足够的资金释放

# #     def __str__(self):
# #         return("vwap订单测试")


# # ts = TradeSession(PM, start_datetime='2021-01-01', end_datetime='2022-02-01')
# # # 事件的时间、市场的时间理应要大于回测的时间'2020-01-01'
# # # ts.add_market(['daily_stock'], start_date='2019-12-01')

# # # # 添加事件
# # # s = pd.date_range(start='2019', end='2023', freq='w')
# # # events1 = event_generator(s.to_pydatetime(), 'week_end', event_seq=5)
# # # events2 = event_generator(s.to_pydatetime(), 'record', seq=6)  # 周末做记录
# # # ts.add_events(events1 + events2)

# # # # 运行
# # obs = ts.run()
# # res1 = obs.check_trade_net('2020', '2022')
# # res2 = obs.check_trade_position('2020', '2022')
# # res3 = obs.check_position_net()
# # res4 = obs.trade_records
# # # ts.observer.balance_records
# # # ts.observer.position_records
