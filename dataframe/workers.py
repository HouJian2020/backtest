# -*- coding: utf-8 -*-
"""

这是一个示例模块的文档

该模块包含了以DataFrame为数据格式的回测过程中的角色。

模块中包含的类:

- ObserverPandas: 基金会计类， 记录回测过程中的所有信息。
- TraderPandas: 交易员类， 负责管理订单策略。
- PMPandas： 基金经理基类， 负责根据持仓数据与外部信息做出买卖决策。

Created on Fri Mar 17 15:05:06 2023

@author: houjian
"""

import numpy as np
import pandas as pd
from typing import List
from ..base_api import Event, ObserverBase, TraderBase, PortfolioManagerBase, OrderStrategyBase
from ..constant import SymbolName, PriceName, VolumeName, TimeName, \
    SecuType, Direction, OrderType, Status
from . import data_model
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.style.use('seaborn-poster')

# %% FOF研究员


class FOF:
    """对回测结果进行分析"""
    @staticmethod
    def max_drawdown(fund_net: list):
        """最大回撤计算"""
        max_retr, max_net = 0, fund_net[0]
        for i in range(1, len(fund_net)):
            max_net = max(max_net, fund_net[i])
            retr = 1-fund_net[i]/max_net
            max_retr = max(retr, max_retr)
        return (max_retr)

    @staticmethod
    def resample_dataframe(df: pd.DataFrame, freq: str = 'week'):
        "按需求降低频率，可选week， month， year， quater， day"
        if freq == "month":
            resampled_df = df.resample("M").last()
        elif freq == "quater":
            resampled_df = df.resample("Q").last()
        elif freq == "week":
            resampled_df = df.resample("W-SUN").last()  # 以每周的周日为结束日期
        elif freq == "day":
            resampled_df = df.resample("D").last()  # 以每周的周日为结束日期
        elif freq == "year":
            resampled_df = df.resample("Y").last()
        else:
            raise ValueError("Invalid frequency")
        return resampled_df

    @staticmethod
    def net_indicator(net,
                      freq: str = 'day',
                      col_bench=None,
                      chicol=False):
        """
        基金的指标的计算.
        """
        net = net.dropna()

        n_map = {'day': 244, 'week': 52, 'month': 12, 'quarter': 4, 'year': 1}
        n_ysamp = n_map[freq]

        if type(net) == pd.Series:
            net = net.to_frame()

        col = ['total_return', 'annual_return', 'annual_risk',
               'max_drawdown', 'sharpe']
        if col_bench is not None:
            col.append('info_ratio')
        result = pd.DataFrame(0, index=net.columns, columns=col)

        net_values = net.values
        result['total_return'] = net_values[-1, :]/net_values[0, :]-1
        result['annual_return'] = (result['total_return']+1) ** \
            (n_ysamp/(net.shape[0]-1))-1
        net_yield = net_values[1:, :]/net_values[:-1, :]-1
        result['annual_risk'] = net_yield.std(axis=0)*(n_ysamp**0.5)
        result['sharpe'] = result['annual_return']/result['annual_risk']
        result['max_drawdown'] = list(
            map(FOF.max_drawdown, list(net_values.T)))

        # 计算信息比率
        if col_bench is not None:
            ind = net.columns.get_loc(col_bench)
            net_yield_relative = net_yield - net_yield[:, [ind]]
            std_relative = net_yield_relative.std(axis=0)*(n_ysamp**0.5)
            annual_return_relative = result['annual_return'] - \
                result['annual_return'].iloc[ind]
            result['info_ratio'] = annual_return_relative / std_relative

        # 是否展示中文表头
        if chicol:
            new_col = ['总收益率', '年化收益率', '年化风险',
                       '最大回撤', '夏普比率']
            if col_bench is not None:
                new_col.append('信息比率')
            result.columns = new_col
        return (result)

    @staticmethod
    def net_plot(df_net: pd.DataFrame, ax=None, title=None):
        df_net = df_net/df_net.iloc[0, :]
        new_fig = ax is None
        if new_fig:
            fig, ax = plt.subplots(figsize=(10, 5))
        df_net.plot(ax=ax)
        ax.set_xlabel('')
        ax.xaxis.set_tick_params(rotation=45)
        ax.set_ylabel('净值', fontsize=20)
        ax.set_title(title, fontsize=20)
        ax.grid(axis='y')
        ax.legend(fontsize=17, loc='upper left')
        if new_fig:
            plt.close()
            return (fig)


# %% 观察者


class ObserverPandas(ObserverBase):
    """
    以DataFrame为数据记录格式， 实现一个具体的Observer(观察者/基金会计)类.

    采用观察者设计模式， 代表基金会计？角色, 负责投资组合所有的信息的记录功能.
    负责记录的内容（属性）：
    1. 当前活跃订单  self.active_orders.
    2. 当前的基金净值 self.balance.
    3. 当前的组合持仓 self.position.
    4. 历史的组合持仓 self.position_records_.
    5. 已经完成的交易记录 self.trade_records.
    6. 已经作废的订单记录 self.invalid_orders.
    7. 历史的基金净值数据 self.balance_records_.

    记录的更新方法（方法）.
    1. on_trade，
    2. add_active_orders，
    3. on_cancelled_orders.
    4. dispense_active_orders.
    5. cal_frozen(计算冻结的持仓和现金，方便下单的时候使用).
    6. settlement(类似基金会计的结算功能).
    7. update_position_records(记录历史持仓).
    8. update_balance_records(记录历史净值).
    """

    def init_active_orders(self):
        return (data_model.init_pandas_data(data_model.ORDER_DATA))

    def init_position(self):
        return (data_model.init_pandas_data(data_model.POSITION_DATA))

    def init_balance(self, cash):
        balance_data = {
            VolumeName.NETASSET: cash,
            VolumeName.CASH: cash,
            VolumeName.FROZENCASH: 0.0, }
        return (balance_data)

    def init_position_records(self):
        return (data_model.init_pandas_data(data_model.POSITION_RECORDS))

    def init_balance_records(self):
        return (data_model.init_pandas_data(data_model.BALANCE_RECORDS))

    def init_trade_records(self):
        return (data_model.init_pandas_data(data_model.TRADE_DATA))

    def init_invalid_orders(self):
        return (data_model.init_pandas_data(data_model.ORDER_DATA))

    def dispense_active_orders(self, secu_types: List[str]):
        """
        交易员尝试执行某个市场活跃的订单，观察者需要做的事情：
        1. 删除属于secu_type类型的所有活跃订单
        2. 把删除的订单return给交易员去处理
        """
        l_select = self.active_orders[SymbolName.SECUTYPE].isin(secu_types)
        data = self.active_orders.loc[l_select].copy()
        self.active_orders = self.active_orders.loc[~l_select].copy()
        return (data)

    def add_tradeid(self, trade_data_raw):
        # 为已经完成的交易添加交易ID

        n_data = trade_data_raw.shape[0]
        trade_data_raw[SymbolName.TRADEID] = list(
            range(self.trade_id, self.trade_id + n_data))

        trade_data = trade_data_raw[data_model.TRADE_DATA.keys()].copy()
        trade_data[SymbolName.TRADEID] = trade_data[SymbolName.TRADEID].astype(
            int)
        self.trade_id += n_data
        return (trade_data)

    def add_orderid(self, order_data_raw):
        # 为订单添加订单id
        n_data = order_data_raw.shape[0]
        order_data_raw[SymbolName.ORDERID] = list(
            range(self.order_id, self.order_id + n_data))

        order_data = order_data_raw[data_model.ORDER_DATA.keys()].copy()
        order_data[SymbolName.ORDERID] = order_data[SymbolName.ORDERID].astype(
            int)
        self.order_id += n_data
        return (order_data)

    def update_position(self, trade_data):
        """
        处理交易发生后对持仓的影响
        1. 如果是平仓的交易：
            a. 计算交易带来的收益（交易价格-结算价格）*交易量
            b. 更新持有的标的数量
            c. 删除持仓量接近0的持仓数据
        2. 如果是开仓的交易：
            a. 把交易数据转化为持仓数据
            b. 更新持仓记录
        """
        l_net = trade_data[SymbolName.DIRECTION] == Direction.NET.value
        # 根据平仓的价格和结算价差异，计算netAsset
        if sum(l_net) > 0:
            df_net = pd.merge(trade_data.loc[l_net],
                              self.position[[
                                  SymbolName.TRADEID, PriceName.SETTLEP]],
                              left_on=SymbolName.OFFSETID,
                              right_on=SymbolName.TRADEID,
                              how='left')
            # 平仓时候仓位是负数，要变为持仓的仓位计算pnl
            pnl = -1 * sum(df_net[VolumeName.TRANSVOL] * (
                df_net[PriceName.TRANSP] - df_net[PriceName.SETTLEP]))
            self.balance[VolumeName.NETASSET] += pnl

            # 根据持仓的交易ID（唯一标识）聚合计算持仓变动量
            p_diff = df_net.groupby(SymbolName.OFFSETID)[
                VolumeName.TRANSVOL].sum()
            p_diff = p_diff.reindex(
                self.position[SymbolName.TRADEID]).fillna(0).to_numpy()
            self.position[VolumeName.VOLUME] += p_diff

            # 删除持仓量接近0的记录
            l_zero = self.position[VolumeName.VOLUME].abs() <= 1e-8
            self.position = self.position.loc[~l_zero].copy()

        # 添加开仓模式下的的交易数据
        df_add = trade_data.loc[~l_net].copy()
        df_add.rename(columns={VolumeName.TRANSVOL: VolumeName.VOLUME},
                      inplace=True)
        df_add[VolumeName.FROZENVOL] = 0
        df_add[PriceName.SETTLEP] = df_add[PriceName.TRANSP]
        df_add.drop(columns=[SymbolName.DIRECTION,
                             SymbolName.OFFSETID,
                             VolumeName.TRANSCOST], inplace=True)
        self.position = pd.concat((self.position, df_add), ignore_index=True)

    def update_balance(self, trade_data):
        """计算balance中的netasset与cash"""
        transaction_cost = sum(trade_data[VolumeName.TRANSCOST])
        self.balance[VolumeName.NETASSET] -= transaction_cost
        self.balance[VolumeName.CASH] -= transaction_cost
        buy_cost = sum(trade_data[VolumeName.TRANSVOL] *
                       trade_data[PriceName.TRANSP])
        self.balance[VolumeName.CASH] -= buy_cost

    def update_trade_records(self, trade_data):
        self.trade_records = pd.concat((self.trade_records,
                                        trade_data), ignore_index=True)

    def update_active_orders(self, order_data):
        if SymbolName.ORDERID not in order_data.columns:
            order_data = self.add_orderid(order_data)
        self.active_orders = pd.concat((self.active_orders,
                                        order_data), ignore_index=True)

    def update_invalid_orders(self, order_data):
        # orderID是所有活动订单以及止损单的唯一标识
        # 订单作废需要从止损单和活动单集合中剔除订单
        order_id = set(order_data[SymbolName.ORDERID])
        self.active_orders = self.active_orders.query(
            f"{SymbolName.ORDERID} not in @order_id").copy()
        self.invalid_orders = pd.concat((self.invalid_orders,
                                         order_data), ignore_index=True)

    def update_active_orders_params(self):
        self.active_order_params = {k: v for k, v in self.active_order_params.items(
        ) if k in self.active_orders[SymbolName.ORDERID]}

    def settlement(self, secu_types: [SecuType]):
        """
        组合结算.
        """
        for secu_type in secu_types:
            l_secu = self.position[SymbolName.SECUTYPE] == secu_type.value
            if sum(l_secu) > 0:
                mkt_obj = self.mkt_dict[secu_type.value]
                old_position = self.position.loc[~l_secu].copy()
                sub_position = self.position.loc[l_secu].copy()
                # 调用市场的结算方法
                pnl, new_position = mkt_obj.settlement(sub_position)

                self.position = pd.concat(
                    (old_position, new_position), ignore_index=True)
                self.balance[VolumeName.NETASSET] += pnl

    def update_position_records(self, record_time):
        """
        增加持仓记录.
        """
        position = self.position.copy(deep=True)
        position[TimeName.RECORDTIME] = record_time
        self.position_records_ = pd.concat((self.position_records_,
                                           position), ignore_index=True)

    def update_balance_records(self, record_time):
        """
        增加组合净值记录.
        """
        balance = self.balance.copy()
        balance[TimeName.RECORDTIME] = record_time
        balance = pd.DataFrame(balance, index=[0])
        self.balance_records_ = pd.concat((self.balance_records_,
                                           balance), ignore_index=True)

    def cal_frozen(self):
        """
        计算冻结的资金和持仓, 仅仅是粗略的估计值，供基金经理下单的参考.
        注意：对传统的下单方式估计值比较准， 但是对于按金额下单以及按比例下单的订单，冻结金额无法计算很准。
        冻结资金计算：PriceName.TARGETP一般为成交估计价格；
        冻结持仓的计算： 根据持仓的交易ID聚合计算；
        """
        # 冻结金额的计算（估算）
        active_orders = self.active_orders.copy()
        active_orders[VolumeName.TARGETVOL] -= active_orders[VolumeName.FILLEDVOL]
        trade_amount = active_orders[PriceName.TARGETP] * \
            active_orders[VolumeName.TARGETVOL]
        l_long = active_orders[VolumeName.TARGETVOL] > 0
        self.balance[VolumeName.FROZENCASH] = sum(trade_amount.loc[l_long])

        # 冻结持仓计算：根据持仓的交易ID（唯一标识）聚合计算冻结的持仓
        l_net = active_orders[SymbolName.DIRECTION] == Direction.NET.value
        p_diff = active_orders.loc[l_net].groupby(
            SymbolName.OFFSETID)[VolumeName.TARGETVOL].sum()
        p_diff = p_diff.reindex(
            self.position[SymbolName.TRADEID]).fillna(0).to_numpy()
        self.position[VolumeName.FROZENVOL] = p_diff*-1

    def check_trade_position(self, start_datetime=None, end_datetime=None):
        """基于期末持仓等于期初持仓+期间的交易检查回测系统"""
        t_record = self.position_records_[TimeName.RECORDTIME].unique()
        t_start, t_end = self.get_check_point(
            t_record, start_datetime, end_datetime)
        if t_start is None:
            return None
        else:
            p_start = self.position_records_.query(
                f"{TimeName.RECORDTIME} == @t_start").groupby(
                    [SymbolName.SECUTYPE, SymbolName.CODE])[
                    VolumeName.VOLUME].sum().to_frame()

            p_end = self.position_records_.query(
                f"{TimeName.RECORDTIME} == @t_end").groupby(
                    [SymbolName.SECUTYPE, SymbolName.CODE])[
                    VolumeName.VOLUME].sum().to_frame()

            p_trade = self.trade_records.query(
                f"({TimeName.TRANSTIME} >= @t_start) & ({TimeName.TRANSTIME} <= @t_end)").groupby(
                    [SymbolName.SECUTYPE, SymbolName.CODE])[
                        VolumeName.TRANSVOL].sum().to_frame()
            res = pd.concat([p_start, p_trade, p_end], axis=1, join='outer')
            res.columns = [VolumeName.VOLUME+'_start',
                           VolumeName.VOLUME+'_trade',
                           VolumeName.VOLUME+'_end', ]
            res[VolumeName.VOLUME+'_net'] = res.iloc[:, 0].fillna(
                0) + res.iloc[:, 1].fillna(0) - res.iloc[:, 2].fillna(0)
            return(res)

    def check_trade_net(self, start_datetime=None, end_datetime=None):
        """基于期初期末持仓价值变动+期间的交易的费用=期初期末净资产的变动的逻辑检查回测系统"""
        t_record1 = self.position_records_[TimeName.RECORDTIME].unique()
        t_record2 = self.balance_records_[TimeName.RECORDTIME].unique()

        t_record = np.array(list(set(t_record1) & set(t_record2)))
        t_start, t_end = self.get_check_point(
            t_record, start_datetime, end_datetime)
        if t_start is None:
            return None
        else:
            p_start = self.position_records_.query(
                f"{TimeName.RECORDTIME} == @t_start")
            pv_start = sum(p_start[VolumeName.VOLUME]
                           * p_start[PriceName.SETTLEP])

            p_end = self.position_records_.query(
                f"{TimeName.RECORDTIME} == @t_end")
            pv_end = sum(p_end[VolumeName.VOLUME] * p_end[PriceName.SETTLEP])

            nav_start = self.balance_records_.set_index(
                TimeName.RECORDTIME).loc[t_start, VolumeName.NETASSET]
            nav_end = self.balance_records_.set_index(
                TimeName.RECORDTIME).loc[t_end, VolumeName.NETASSET]

            trade_records = self.trade_records.query(
                f"({TimeName.TRANSTIME} >= @t_start) & ({TimeName.TRANSTIME} <= @t_end)")
            trade_amount = sum(
                trade_records[VolumeName.TRANSVOL]*trade_records[PriceName.TRANSP])
            trade_cost = trade_records[VolumeName.TRANSCOST].sum()

            nav_end_cal = nav_start - trade_amount - trade_cost + pv_end - pv_start
            res = {'nav_end': nav_end,
                   'nav_start': nav_start,
                   'trade_amount': trade_amount,
                   'trade_cost': trade_cost,
                   'pv_end': pv_end,
                   'pv_start': pv_start,
                   'nav_end_cal': nav_end_cal,
                   }
            return(res)

    def check_position_net(self):
        """基于持仓资产+现金等于基金净值的逻辑检查回测系统"""
        p_asset_name = 'position_asset'
        position = self.position_records_.groupby(TimeName.RECORDTIME).apply(
            lambda x: sum(x[VolumeName.VOLUME] * x[PriceName.SETTLEP]))
        position.name = p_asset_name
        net = self.balance_records_.set_index(TimeName.RECORDTIME)
        res = pd.concat((net, position), axis=1)[
            [VolumeName.NETASSET, VolumeName.CASH, p_asset_name]]
        # res.rename(columns={PriceName.SETTLEP: }, inplace=True)
        res[VolumeName.NETASSET + '_cal'] = res[VolumeName.CASH] + \
            res[p_asset_name].fillna(0)
        return (res)

    def cal_hsl(self,
                start_datetime=None,
                end_datetime=None,
                include_trade_cost=True):
        """换手率 = 期间成交额 / （期初组合净资产 + 期末组合净资产）"""
        t_record = self.balance_records_[TimeName.RECORDTIME].unique()
        t_start, t_end = self.get_check_point(
            t_record, start_datetime, end_datetime)

        if t_start is None:
            return None
        else:
            nav_start = self.balance_records_.set_index(
                TimeName.RECORDTIME).loc[t_start, VolumeName.NETASSET]
            nav_end = self.balance_records_.set_index(
                TimeName.RECORDTIME).loc[t_end, VolumeName.NETASSET]

            trade_records = self.trade_records.query(
                f"({TimeName.TRANSTIME} >= @t_start) & ({TimeName.TRANSTIME} <= @t_end)")
            trade_amount = sum(trade_records[VolumeName.TRANSVOL].abs() *
                               trade_records[PriceName.TRANSP])
            trade_cost = trade_records[VolumeName.TRANSCOST].sum()

            if include_trade_cost:
                hsl = (trade_amount + trade_cost) / (nav_start + nav_end)
            else:
                hsl = trade_amount / (nav_start + nav_end)
            res = {
                'start_record_time': t_start,
                'end_record_time': t_end,
                'nav_end': nav_end,
                'nav_start': nav_start,
                'trade_amount': trade_amount,
                'trade_cost': trade_cost,
                'hsl': hsl,
            }
            return (res)

    def cal_hsl_batch(self,
                      start_datetime_list=None,
                      end_datetime_list=None,
                      include_trade_cost=True):
        """时间序列批量计算换手率"""
        hsl = []
        for t_s, t_e in zip(start_datetime_list, end_datetime_list):
            res = self.cal_hsl(t_s, t_e)
            if res is not None:
                hsl += [res]
        return pd.DataFrame(hsl)

    def pnl_analyse(self,
                    start_datetime: str = None,
                    end_datetime: str = None,
                    freq: str = 'week',
                    benchmark: pd.Series = None,
                    chicol: bool = True):  # 是否展示中文名称
        net = self.get_pnl_net(start_datetime=start_datetime,
                               end_datetime=end_datetime,
                               benchmark=benchmark,
                               freq=freq)
        if (net is None) or (net.shape[0] == 0):
            return None
        else:
            col_bench = None if benchmark is None else 'benchmark'
            df_indicator = FOF.net_indicator(
                net, freq=freq, col_bench=col_bench, chicol=chicol)
        return (df_indicator)

    def pnl_plot(self,
                 start_datetime: str = None,
                 end_datetime: str = None,
                 benchmark: pd.Series = None,
                 freq: str = 'week',
                 ax=None,
                 title: str = None):
        net = self.get_pnl_net(start_datetime=start_datetime,
                               end_datetime=end_datetime,
                               benchmark=benchmark,
                               freq=freq)

        if (net is None) or (net.shape[0] == 0):
            return None
        else:
            fig = FOF.net_plot(net, ax=ax, title=title)
            return (fig)

    def get_pnl_net(self,
                    start_datetime: str = None,
                    end_datetime: str = None,
                    freq: str = 'week',
                    benchmark: pd.Series = None):
        t_record = self.balance_records_[TimeName.RECORDTIME].unique()
        t_start, t_end = self.get_check_point(
            t_record, start_datetime, end_datetime)
        if t_start is None:
            return None
        else:
            net = self.balance_records_.query(
                f"({TimeName.RECORDTIME} >= @t_start) & ({TimeName.RECORDTIME} <= @t_end)").copy()
            net = net.set_index(TimeName.RECORDTIME)[VolumeName.NETASSET]
            net = FOF.resample_dataframe(net, freq)
            if benchmark is not None:
                benchmark = FOF.resample_dataframe(benchmark, freq)
                net = pd.concat((net, benchmark), axis=1, join='inner')
                net.columns = [VolumeName.NETASSET, 'benchmark']
            else:
                net = net.to_frame()
        return (net.dropna())

    def get_check_point(self, t_record, start_datetime, end_datetime):
        """
        根据输入的开始时间和输入时间，尽可能的匹配t_record中的开始和结束时间点。
        返回一个时间对，要么都有值，要么都为None
        """
        if len(t_record) == 0:
            return (None, None)

        if start_datetime is None:
            t_min = min(t_record)
        else:
            t_min = pd.to_datetime(start_datetime)

        if end_datetime is None:
            t_max = max(t_record)
        else:
            t_max = pd.to_datetime(end_datetime)

        if t_min > t_max:
            t_min, t_max = t_max, t_min

        start_set = t_record[t_record >= t_min]
        if len(start_set) == 0:
            return (None, None)
        else:
            t_start = pd.to_datetime(min(start_set))

        end_set = t_record[t_record <= t_max]
        if len(end_set) == 0:
            return (None, None)
        else:
            t_end = pd.to_datetime(max(end_set))
        return(t_start, t_end)

    @property
    def position_records(self):
        record = self.position_records_.copy()
        record[TimeName.DATE] = record[TimeName.RECORDTIME].dt.date
        v_diff = record[VolumeName.VOLUME] - record[VolumeName.FROZENVOL]
        record[VolumeName.MKTV] = record[PriceName.SETTLEP] * v_diff
        return(record)

    @property
    def balance_records(self):
        record = self.balance_records_.copy()
        record[TimeName.DATE] = record[TimeName.RECORDTIME].dt.date
        return(record)


# %% 交易员


class TraderPandas(TraderBase):
    """
    以DataFrame为数据格式， 基于data_model对数据格式的定义实现的交易员类.
    """

    def on_cancel(self, order_data_id):
        """
        1. order_data_id可是订单id，或者是包含订单id的DataFrame
        2. 根据订单id获得需要处理的订单， 并通过订单策略的cancel_orders方法取消订单
        """
        if isinstance(order_data_id, pd.DataFrame):
            order_data_id = order_data_id[SymbolName.ORDERID]
        order_data = self.observer.active_orders.query(
            f"{SymbolName.ORDERID} in @order_data_id")
        for (secu_type_str, order_type_str), data in order_data.groupby(
                [SymbolName.SECUTYPE, SymbolName.ORDERTYPE]):
            # 获取订单策略
            secu_type = SecuType(secu_type_str)
            order_type = OrderType(order_type_str)
            order_strategy = self.get_order_strategy(secu_type, order_type)
            # 调用取消订单的函数
            order_strategy.cancel_orders(data)

    def on_trigger_events(self, events: List[Event]):
        """
        1. 行情推动事件发生后，交易员开始进行交易。
        2. 注意：执行顺序是以订单的顺序执行。
        """
        # 字典{市场： 事件}
        secu_event = {
            event.secu_type.value: event.event_type for event in events}

        # 获取事件相关的订单
        order_data = self.observer.dispense_active_orders(secu_event.keys())
        for sequ in np.sort(order_data[SymbolName.ORDERSEQ].unique()):
            # 按照订单的ORDERSEQ顺序处理订单，SEQ越小则越优先处理
            order_data_sub = order_data.query(
                f"{SymbolName.ORDERSEQ} == @sequ")
            for (secu_type_str, order_type_str), data in order_data_sub.groupby(
                    [SymbolName.SECUTYPE, SymbolName.ORDERTYPE]):
                # 获取订单策略
                secu_type = SecuType(secu_type_str)
                order_type = OrderType(order_type_str)
                order_strategy = self.get_order_strategy(secu_type, order_type)

                # 同一个市场同一事件仅有一个trigger类事件
                method = getattr(order_strategy, 'on_' +
                                 secu_event[secu_type_str].value, None)
                # 处理订单
                if method is None:
                    self.observer.add_active_orders(data)
                else:
                    order_strategy.on_trigger_event(method, data)


# %% 基金经理


class PMPandas(PortfolioManagerBase):

    def reverse_position(self, position):
        position[VolumeName.TARGETVOL] = position[VolumeName.FROZENVOL] - \
            position[VolumeName.VOLUME]
        position = position.loc[position[VolumeName.TARGETVOL].abs(
        ) > 1e-6].copy()
        return position

    def sell_position(self, position, order_seq=0):
        """输入持仓数据， 发送反向市价订单"""
        position = self.reverse_position(position)
        for secu_type, data in position.groupby(SymbolName.SECUTYPE):
            self.send_orders(data,
                             order_type=OrderType.MARKET,
                             secu_type=secu_type,
                             order_seq=order_seq)

    def get_mkt_frame(self, secu_type=None):
        """
        获取对应市场行情数据.
        """
        return (self.get_mkt_obj(secu_type).current_frame)

# %% 订单策略工具函数


class OrderStrategyPandas(OrderStrategyBase):
    def new_order(self, n_orders):
        """根据订单数量生成订单模板，以免重复编辑信息"""
        order_data = data_model.init_pandas_data(data_model.ORDER_DATA)
        del order_data[SymbolName.ORDERID]
        order_data[SymbolName.SECUTYPE] = [
            self.secu_type.value] * n_orders  # 扩展表格至n_orders行

        order_data[TimeName.ORDERTIME] = self.observer.time
        order_data[SymbolName.ORDERTYPE] = self.order_type.value
        # 默认为long 根据实际情况调整
        order_data[SymbolName.DIRECTION] = Direction.LONG.value
        order_data[VolumeName.FILLEDVOL] = 0
        order_data[SymbolName.OFFSETID] = -1
        order_data[SymbolName.STATUS] = Status.UNSUBMIT.value
        return (order_data)

    def net_order(self, order_data):
        """
        结合持仓情况修改订单数据， 修改字段含DIRECTION、 OFFSETID。
        如果订单中有和持仓标的相同， 但方向相反的订单这进行net操作：
            1. 订单量较大的则生成两个订单。
            2. 订单量较小的， 修改订单的DIRECTION、 OFFSETID。
        提交订单后需要更新冻结的资金与持仓信息。
        """
        # 获取持仓信息
        position = self.observer.position.query(f"{SymbolName.SECUTYPE} == @self.secu_type.value")[
            [SymbolName.CODE, SymbolName.TRADEID, VolumeName.VOLUME, VolumeName.FROZENVOL]].copy()
        position['net_volume'] = position[VolumeName.VOLUME] - \
            position[VolumeName.FROZENVOL]
        position.drop(columns=[VolumeName.VOLUME,
                      VolumeName.FROZENVOL], inplace=True)

        # 计算持仓与订单重合的标的，精简数据
        code_set = set(order_data[SymbolName.CODE]) & set(
            position[SymbolName.CODE])
        position = position.query(f"{SymbolName.CODE} in @code_set").copy()
        orders = order_data.query(
            f"{SymbolName.CODE} in @code_set").to_dict(orient='records')
        order_data = order_data.query(
            f"{SymbolName.CODE} not in @code_set").copy()

        valid_orders = []
        while orders:
            order = orders.pop()
            l_code = position[SymbolName.CODE] == order[SymbolName.CODE]
            # 订单方向与持仓方向相反才可能对冲
            l_net = np.sign(order[VolumeName.TARGETVOL]) * \
                np.sign(position['net_volume']) < 0
            index = np.where(l_net & l_code)[0]
            if len(index) != 0:
                index = index[-1]
                if abs(order[VolumeName.TARGETVOL]) > abs(position.iloc[index, 2]) + 1e-6:

                    # 如果订单量大于持仓量， 则新生成一个待处理的订单，初始订单量等于持仓量*-1，
                    order_copy = order.copy()
                    order_copy[VolumeName.TARGETVOL] = order[VolumeName.TARGETVOL] + \
                        position.iloc[index, 2]
                    orders.append(order_copy)
                    order[VolumeName.TARGETVOL] = -1*position.iloc[index, 2]
                    # 持仓量变为0
                    position.iloc[index, 2] = 0
                else:
                    # 如果持仓量很大， 那么持仓量减去订单量即可， 原始的订单量不变
                    position.iloc[index, 2] += order[VolumeName.TARGETVOL]
                # 只要发生net行为，那么原始订单的OFFSETID就等于持仓的TRADEID
                order[SymbolName.OFFSETID] = position.iloc[index, 1]

            valid_orders += [order]

        order_data = pd.concat(
            (order_data, pd.DataFrame(valid_orders)), ignore_index=True)
        l1 = order_data[VolumeName.TARGETVOL] > 0
        l2 = order_data[SymbolName.OFFSETID] >= 0

        order_data.loc[l1, SymbolName.DIRECTION] = Direction.LONG.value
        order_data.loc[~l1, SymbolName.DIRECTION] = Direction.SHORT.value
        order_data.loc[l2, SymbolName.DIRECTION] = Direction.NET.value
        return (order_data)

    def order_to_trade(self, order_data):
        """
        把订单数据转化为交易数据。
        1. 先处理卖单然后处理买单。
        2. 如果没有足够的金额完成交易， 则按照比例完成买单交易。
        """
        trade_data = order_data.copy()
        trade_data[TimeName.TRANSTIME] = self.observer.time
        trade_data[PriceName.TRANSP] = trade_data[PriceName.TARGETP]

        # 成交额、成交量、手续费初步估计
        trade_volume = trade_data[VolumeName.TARGETVOL] - \
            trade_data[VolumeName.FILLEDVOL]
        trade_amount = trade_data[PriceName.TARGETP] * trade_volume

        l_short = trade_volume < 0
        trade_cost = trade_amount * self.mkt_obj.buy_fee_ratio
        trade_cost.loc[l_short] = trade_amount.loc[l_short].abs() * \
            self.mkt_obj.sell_fee_ratio

        # 资金不够买入， 那么买单按比例成交
        cash = self.observer.balance[VolumeName.CASH]

        if sum(trade_cost) + sum(trade_amount) > cash + 1e-6:
            # 处理卖单后的现金
            cash = cash - \
                trade_cost.loc[l_short].sum() - trade_amount.loc[l_short].sum()
            # 买单需要的现金
            cash_need = trade_cost.loc[~l_short] + trade_amount.loc[~l_short]

            if cash_need.sum() < 1e-5:
                print(1)
            # 买单的比例
            ratio = cash / cash_need.sum()
            # 更新交易手续费和成交额
            trade_cost.loc[~l_short] *= ratio
            trade_volume.loc[~l_short] *= ratio

            unfinished_order = order_data.loc[~l_short].copy()
            unfinished_order[VolumeName.FILLEDVOL] += trade_volume.loc[~l_short]
        else:
            unfinished_order = order_data.head(0)  # 保留个表头

        # 更新交易数据的成交额与手续费
        trade_data[VolumeName.TRANSCOST] = trade_cost
        trade_data[VolumeName.TRANSVOL] = trade_volume

        # 订单格式更新
        columns = list(data_model.TRADE_DATA.keys())
        columns.remove(SymbolName.TRADEID)
        trade_data_raw = trade_data[columns].copy()
        return (trade_data_raw, unfinished_order)

    def netasset_ratio_to_volume(self,
                                 ratio: pd.Series,
                                 target_price: pd.Series):
        """根据交易金额/净资产比例计算目标交易量"""
        net_asset = self.observer.balance[VolumeName.NETASSET]
        if isinstance(ratio, pd.Series):
            ratio = ratio.values
        if isinstance(target_price, pd.Series):
            target_price = target_price.values
        target_amount = net_asset * ratio/(
            1+self.mkt_obj.buy_fee_ratio)
        target_volume = target_amount / target_price
        return (target_volume)

    def save_order_params(self, params: pd.DataFrame):
        "订单参数转化为字典"
        params = params.to_dict(orient='index')
        self.observer.active_order_params.update(params)

    def get_order_params(self, order_ids: pd.Series):
        "根据订单id获取订单参数"
        params = pd.DataFrame.from_dict(
            {k: self.observer.active_order_params[k] for k in
             order_ids}, orient='index')
        # 虽然字典是有序的，重新排序后会更加保险
        params = params.loc[order_ids]
        return params
