# -*- coding: utf-8 -*-
"""
模块包含基于某个具体市场相关类的实现:
1. 基于MarketBase接口的市场类的实现;
    a. 类属性： 市场类型， 手续费率
    b. 实例属性： 市场数据， 行情推动事件trigger_events
    c. 抽象方法： init, settlment， on_triger
2. 基于OrderStrategyBase的若干个订单策略的实现.
    a. 类属性： 市场类型， 订单属性
    b. 抽象方法： send_orders, 基于trigger_events的方法，如'on_open',其中'open'是trigger_events的一个事件类型


Created on Tue Mar 21 08:41:50 2023

@author: houjian
"""
import datetime
import numpy as np
import pandas as pd
from . import data_model
from ..base_api import Event, MarketBase, OrderStrategyBase
from .workers import OrderStrategyPandas
from ..constant import SecuType, OrderType, EventType, SymbolName, TimeName,\
    PriceName, VolumeName, Status, Direction

# %% 具体市场类


class CBMarket(MarketBase):
    """
    1. 可转债市场， 封装该市场的相关数据
        （交易费率， 行情推动事件， 行情数据）。
    2. 标的行情数据占用内存较大，因此数据被所有角色（PM，observer， trader）共享，
        建议只可读取，不建议修改。
    """

    secu_type: SecuType = SecuType.CBDAILY  # 证券类型
    buy_fee_ratio = 0.0005  # 买入手续费率
    sell_fee_ratio = 0.0015  # 卖出手续费率
    vwap_volume_limit_ratio: float = 0.05  # 单票当日vwap订单成交量占比不超过5%
    # 1手可转债面额为100， 市场计volume为100
    # 系统内的volume 恒等于 金额/价格
    volume_multiplier = 100  # 同一份标的市场的计量值和系统内部计量值的比例
    min_send_buy_volume = 10  # 下单时刻如果量小于n张则取消订单

    data_path = r'D:\mypkgs\backtest\data_mock\cbmkt.pickle'

    def __init__(self, start_date: str = '2012-01-01'):
        # 加载市场行情数据至self.data中
        self.get_data(start_date=start_date)
        # 计算self.trigger_events(行情推动事件集合)
        self.init_trigger_events()

    def get_data(self, start_date):
        """
        获市场行情数据， 格式要参考data_model.CB_MKTDATA.
        模拟的A股市场数据，根据使用者的数据库结构做数据结构转化
        """
        # import os
        # root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # data_path = os.path.join(root_path, 'data_mock', 'cb', 'cbmkt.pickle')
        # 市场数据读取与初步处理
        df_data = pd.read_pickle(self.data_path)
        df_data.reset_index(inplace=True)
        df_data['date'] = pd.to_datetime(df_data['date'])

        # TODO 交易日数据是否需要单独维护
        self.trade_dates = pd.Series(np.sort(df_data['date'].unique()))

        # TODO 数据清洗逻辑
        # 用成交量判断当日是否有交易， 选择每一个标的当日有交易的最大与最小日期
        l_nan = df_data['成交量'].isnull() | (df_data['成交量'] == 0)
        df_info = df_data.loc[~l_nan].groupby(
            'windcode')['date'].agg(['min', 'max'])

        # 数据最的处理方法， 方案一和二选择一个即可，注释掉另外一个代码块
        # 方案一： 标的数据的最大日期为原始数据“对应标的”数据的最大日期，（哪怕是停牌数据，也要暂认为标的处于存续期），该方案不支持持有退市标的
        # max_date = df_data.groupby('windcode')['date'].max()
        # secu_subsist = max_date.index[max_date == self.trade_dates.max()]
        # df_info.loc[secu_subsist.intersection(df_info.index), 'max'] = max_date

        # 方案二： 标的数据的最大日期为原始数据“所有标的”数据的最大日期，该方案支持持有退市标的
        df_info['max'] = self.trade_dates.max()

        # 根据最大最小日期，计算标的数据的开始与结束日期
        date_set = df_info.apply(
            lambda x: pd.Series(pd.date_range(x['min'], x['max'])),
            axis=1).stack().reset_index()

        date_set.columns = ['windcode', 'id', 'date']
        del date_set['id']
        date_set = date_set.query("date in @self.trade_dates")

        cols = ['windcode', 'date', '开盘价', '最高价',
                '最低价', '收盘价', 'VWAP', '成交量', '成交额']
        df_data['VWAP'] = self.volume_multiplier * \
            df_data['成交额_净价'] / df_data['成交量']
        df_data = pd.merge(date_set, df_data[cols],
                           how='left', on=['windcode', 'date'])

        # 1. 量额用0填充缺失值
        cols = ['成交量', '成交额']
        df_data[cols] = df_data[cols].fillna(0)

        # 2. 如果收盘价缺失或者成交量为0， 则开、高、低、收数据无效
        l_close_nan = df_data['收盘价'].isnull() | (df_data['成交量'] < 0.1)
        df_data.loc[l_close_nan, ['开盘价', '最高价', '最低价', '收盘价', 'VWAP']] = np.nan

        # 3. 收盘用前值填充缺失值
        df_data['收盘价'] = df_data.groupby(
            'windcode')['收盘价'].fillna(method='ffill')

        # 4. 开、高、 低用收盘价填缺失值
        for col in ['开盘价', '最高价', '最低价', 'VWAP']:
            df_data[col] = df_data[col].fillna(df_data['收盘价'])

        # 结果汇总
        self.data = data_model.init_pandas_data(data_model.CB_MKTDATA)
        self.data[SymbolName.CODE] = df_data['windcode']
        self.data[TimeName.TDATE] = df_data['date']
        self.data[PriceName.OPENADJ] = df_data['开盘价']
        self.data[PriceName.HIGHADJ] = df_data['最高价']
        self.data[PriceName.LOWADJ] = df_data['最低价']
        self.data[PriceName.CLOSEADJ] = df_data['收盘价']
        self.data[VolumeName.VOLUME] = df_data['成交量']
        self.data[VolumeName.AMOUNT] = df_data['成交额']
        self.data[PriceName.VWAPADJ] = df_data['VWAP']

        self.data = self.data.query(f"{TimeName.TDATE} >= @start_date").copy()

    def init_trigger_events(self):
        """
        初始化行情推动事件.
        同一个市场不同的脚本行情推动事件是不一样的
        如daily_stock.py中行情推动事件是开盘和收盘，
        但是tick_stock.py中行情推动事件是tick.
        如果要做不同精细程度的回测需要增加新的脚本.
        """

        # 根据市场数据的范围限制行情驱动的事件范围
        l1 = self.trade_dates >= self.data[TimeName.TDATE].min()
        l2 = self.trade_dates <= self.data[TimeName.TDATE].max()

        self.trigger_events = []
        for trade_date in self.trade_dates.loc[l1 & l2]:
            day = trade_date.to_pydatetime()
            event = Event(
                # 开盘时间为9：30
                datetime=day + datetime.timedelta(hours=9, minutes=30),
                event_seq=0,
                is_trigger=True,
                secu_type=self.secu_type,
                event_type=EventType.OPEN,
                # trade_date=trade_date.date(),
            )
            self.trigger_events.append(event)

            event = Event(
                # 收盘时间为15：00
                datetime=day + datetime.timedelta(hours=15),
                event_seq=0,
                is_trigger=True,
                secu_type=self.secu_type,
                event_type=EventType.CLOSE,
                # trade_date=trade_date.date(),
            )
            self.trigger_events.append(event)
        # 转化为集合
        self.trigger_events = set(self.trigger_events)

    def init(self, t: datetime.datetime):
        """
        初始化市场对象至t时刻前的状态。
        目的：为了保证回测时市场的状态和实盘时的状态一致。
        """
        events = [event for event in self.trigger_events if event.datetime < t]
        self.on_trigger(max(events))

    def on_trigger(self, event: Event):
        """
        1. 获取结算价格
        2. 获取当前市场数据供订单策略的使用
        """
        self.trade_date = pd.Timestamp(event.datetime.date())  # 市场当前的交易日
        lt = self.data[TimeName.TDATE] == self.trade_date
        self.current_frame = self.data.loc[lt].copy()
        if event.event_type == EventType.OPEN:
            self.current_frame[PriceName.SETTLEP] = self.current_frame[PriceName.OPENADJ]
            # 删除当前无法获得的数据
            self.current_frame.drop(columns=[
                PriceName.HIGHADJ, PriceName.LOWADJ, PriceName.CLOSEADJ,
                # VolumeName.VOLUME, VolumeName.AMOUNT
            ],
                inplace=True)
        elif event.event_type == EventType.CLOSE:
            self.current_frame[PriceName.SETTLEP] = self.current_frame[PriceName.CLOSEADJ]
        self.current_frame = self.current_frame.set_index(SymbolName.CODE)

    def settlement(self, position):
        """
        结算当前持仓.
        """
        settlep = self.current_frame.loc[position[SymbolName.CODE],
                                         PriceName.SETTLEP].to_numpy()
        pnl = sum((settlep-position[PriceName.SETTLEP])
                  * position[VolumeName.VOLUME])
        position[PriceName.SETTLEP] = settlep
        return (pnl, position)

# %% 按量下单的订单策略


class VolumeOrderStrategy(OrderStrategyPandas):
    """
    1. 根据订单中的targeVolume决定成交量
    2. 根据当前市场结算价settlePrice替代目标价格
    """

    def send_orders(self,
                    data,  # 至少包含代码SymbolName.CODE、 目标交易量SymbolName.TARGETVOL两个字段
                    order_seq: int = None,  # 订单交易序号，序列号越小交易越优先，一般卖单比买单优先
                    ):
        """
        1. 市价单最少要提交的数据有：标的代码、成交量。
        2. 如果优先级（order_seq）为None， 则买单优先级为10， 卖单优先级为5.
        """
        if SymbolName.CODE not in data.columns:
            raise (ValueError(
                f"输入的订单数据必须包含{SymbolName.CODE}字段"))

        if VolumeName.TARGETVOL not in data.columns:
            raise (ValueError(
                f"输入的订单数据必须包含{VolumeName.TARGETVOL}字段"))

        # 初始化订单
        order_data = self.new_order(data.shape[0])

        # 编辑订单信息
        order_data[SymbolName.CODE] = data[SymbolName.CODE].values
        order_data[VolumeName.TARGETVOL] = data[VolumeName.TARGETVOL].values
        order_data[SymbolName.ORDERSEQ] = order_seq
        if order_seq is None:
            l_long = order_data[VolumeName.TARGETVOL] > 0
            order_data[SymbolName.ORDERSEQ] = np.where(l_long, 10, 5)

        # 市价单用结算价替代目标价格
        info = self.mkt_obj.current_frame.loc[data[SymbolName.CODE]]
        order_data[PriceName.TARGETP] = info[PriceName.SETTLEP].values

        # 如果订单量小于self.mkt_obj.min_send_buy_volume那么就取消订单
        l_v = order_data[VolumeName.TARGETVOL] >= self.mkt_obj.min_send_buy_volume
        l_short = order_data[VolumeName.TARGETVOL] < 0
        return order_data.loc[l_v | l_short].copy()

# %% 市价单策略


class CBMarketOrder(VolumeOrderStrategy):
    """
    市价单策略
    1. 交易时如果资金不足则在买单中按比例分配。
    2. 未交易成功的部分，订单会被保留等待成交。
    3. 不考虑冲击成本， 适合小资金策略回测.
    """

    secu_type = SecuType.CBDAILY
    order_type = OrderType.MARKET

    def send_orders(self, data, order_seq):
        order_data = super().send_orders(data, order_seq=order_seq)
        # 结合持仓修改订单信息, 考虑平仓
        order_data = self.net_order(order_data)
        # # 添加订单id
        # order_data = self.observer.add_orderid(order_data)
        return order_data

    def on_close(self, order_data):
        """
        收盘:
        1. 根据订单状态是否为“UNSUBMIT”以及买卖方向共分四种情况判断是否可成交。
        2. 可成交订单走order_to_trade方法执行交易。
        3. 不可成交的订单修改订单状态后返回给observer中的active_orders中。
        """

        info = self.mkt_obj.current_frame.loc[
            order_data[SymbolName.CODE]].copy()

        # 改变订单状态，状态为"unsubmit"的订单改为"active"
        l_unsubmit = order_data[SymbolName.STATUS] == Status.UNSUBMIT.value
        order_data.loc[l_unsubmit, SymbolName.STATUS] = Status.ACTIVE.value

        # 更新当前结算价
        order_data[PriceName.TARGETP] = info[PriceName.SETTLEP].values

        # 是否可交易
        l_trade = (info[VolumeName.VOLUME] > 0).values  # 当日有成交

        # 执行可交易的订单
        trade_data_raw, unfinished_order = self.order_to_trade(
            order_data.loc[l_trade])

        # 更新不可执行的订单， 并更新订单的PriceName.TARGETP
        unfilled_orders = pd.concat(
            (order_data.loc[~l_trade], unfinished_order), ignore_index=True)
        return (unfilled_orders, None, trade_data_raw)

    def on_open(self, order_data):
        """
        开盘: 交易逻辑与收盘相同。
        """
        res = self.on_close(order_data)
        return res


class CBMarketOrderFAK(CBMarketOrder):
    """
    market订单。
    没有成交的部分直接取消（FAK型订单）
    """
    order_type = OrderType.MARKETFAK

    def on_close(self, order_data):
        unfilled_orders, _, trade_data_raw = super().on_close(order_data)
        return None, unfilled_orders, trade_data_raw


class CBRatioMarketOrder(CBMarketOrder):
    """
    按净资产的一定比例成交的买单策略。
    1. 把比例参数存储到观察者订单参数字典当中（self.observer.active_order_params）；
    2. 如果成交时资金不足则按照比例买入；
    注意：订单中仅含买单。
    """

    order_type = OrderType.RATIOMARKET
    ratio_relative = 'ratio_relative'  # 输入数据必须包含的字段

    def send_orders(self,
                    data,  # 至少包含代码SymbolName.CODE、 ratio_relative两个字段
                    ratio_to_netasset: float = 1.0,  # 总的买入资金占净资产的比例
                    confirm_volume: bool = False,  # 是否此时确定成交量
                    order_seq: int = None,  # 订单交易序号，序列号越小交易越优先，一般卖单比买单优先
                    ):
        """
        1. 订单根据占总资产比例ratio_asset确定订单的目标量
            a. 假设净资产20000， 标的和比例的对应关系为【A、B  C】：【1，2，3】
            b. 占净资产比例ratio_to_netasset = 10%；
            c. A的目标成交量 =  20000* 10% * (1/(1+2+3))
        2. 根据ratio_volume_limit确定成交金额的上限制；
        3. event_confirm_volume为成交量确定的事件，一旦确定后成交量就不变，订单成交的逻辑按照普通的vwap订单执行, 建议默认参数False， 更复合客观情况。
        """

        if self.ratio_relative not in data.columns:
            raise (ValueError(
                f"{self.order_type.value}的输入数据必须包含{self.ratio_relative}字段，代表订单之间的相对比例"))

        if sum(data[self.ratio_relative] < 0) > 0:
            raise (ValueError("比例市价单不能含有卖单"))

        # 生成订单
        # 创建一个列，用于初始订单生成
        data = data.copy()
        data[VolumeName.TARGETVOL] = self.mkt_obj.min_send_buy_volume + 1
        order_data = super(CBMarketOrder, self).send_orders(data=data,
                                                            order_seq=order_seq)
        order_data = self.observer.add_orderid(order_data)

        # 参数保存在self.observer.active_order_params中
        params = order_data.set_index(SymbolName.ORDERID)[[]]
        params[self.ratio_relative] = ratio_to_netasset * \
            data[self.ratio_relative].values / sum(data[self.ratio_relative])

        # 从比例计算目标仓位
        target_price = order_data[PriceName.TARGETP].values
        order_data[VolumeName.TARGETVOL] = self.netasset_ratio_to_volume(
            params[self.ratio_relative], target_price)

        # 取消小订单的标的
        l_v = order_data[VolumeName.TARGETVOL] >= self.mkt_obj.min_send_buy_volume
        l_short = order_data[VolumeName.TARGETVOL] < 0
        order_data = order_data.loc[l_v | l_short].copy()

        # 是否此刻确定目标成交量
        if not confirm_volume:
            self.save_order_params(params)
        else:
            order_data[SymbolName.ORDERTYPE] = OrderType.MARKET.value
        return order_data

    def on_close(self, order_data):
        """
        1. 根据比例以及价格更新持仓
        2. 订单按照走CBMarketOrder的交易逻辑执行
        3. 更改订单类型至Market
        """
        # 获取订单参数
        params = self.get_order_params(order_data[SymbolName.ORDERID])
        # 更新持仓
        target_price = self.mkt_obj.current_frame.loc[
            order_data[SymbolName.CODE], PriceName.SETTLEP].values
        order_data[VolumeName.TARGETVOL] = self.netasset_ratio_to_volume(
            params[self.ratio_relative], target_price)
        # 更新订单类型
        order_data[SymbolName.ORDERTYPE] = OrderType.MARKET.value

        # 执行交易
        unfilled_orders, _, trade_data_raw = super().on_close(order_data)
        return unfilled_orders, _, trade_data_raw

    def on_open(self, order_data):
        """

        开盘: 交易逻辑与收盘相同。
        """
        res = self.on_close(order_data)
        return res


# %% VWAP 订单策略

class CBVWAPOrder(VolumeOrderStrategy):
    """
    VWAP成交订单策略。
    成交量不得超过当日的成交量的一定比例， 如5%。
    """
    secu_type = SecuType.CBDAILY
    order_type = OrderType.VWAP

    def send_orders(self,
                    data,  # 至少包含代码SymbolName.CODE、 目标交易量SymbolName.TARGETVOL两个字段
                    order_seq: int = None,  # 订单交易序号，序列号越小交易越优先，一般卖单比买单优先
                    ):
        """
        1. data 最少要提交的数据有：标的代码、成交量两个字段.
        2. 优先级 如果不输入优先级， 则买单优先级为10， 卖单优先级为5.
        3. ratio_volume_limit存放在订单的字典参数中。
        """
        order_data = super().send_orders(data=data, order_seq=order_seq)
        order_data = self.net_order(order_data)
        return order_data

    def on_open(self, order_data):
        """
        开盘：更改目标结算价格， 不做任何交易
        """
        order_data[PriceName.TARGETP] = self.mkt_obj.current_frame.loc[
            order_data[SymbolName.CODE], PriceName.SETTLEP].values
        return order_data, None, None

    def on_close(self, order_data):
        """
        收盘: 结合市场的vwap_volume_limit_ratio参数确定交易上限.
        """
        tol = 1e-6  # 过滤成交量的阈值

        info = self.mkt_obj.current_frame.loc[
            order_data[SymbolName.CODE]].copy()
        order_data[PriceName.TARGETP] = info[PriceName.VWAPADJ].values

        # 复制一份order_data用于计算未完成的量
        order_data_copy = order_data.copy()

        # 当日成交的交易量的限制
        volume_limit = self.mkt_obj.vwap_volume_limit_ratio * \
            info[VolumeName.VOLUME].values / self.mkt_obj.volume_multiplier

        # 根据成交量限额， 以order_seq为顺序，确定能可以完成的订单
        order_tradable = self.volume_limit_filter(
            order_data, volume_limit, tol)

        # 根据当前资金量，确定可以完成的订单
        trade_data_raw, unfinished_order = self.order_to_trade(
            order_tradable)

        # 根据成交的情况， 重构活跃的订单
        # 未成交的订单等于原始订单减去已经成交的订单
        trade_agg = trade_data_raw.groupby(SymbolName.ORDERID, as_index=False)[
            VolumeName.TRANSVOL].sum()
        active_orders = pd.merge(order_data_copy,  # 原始订单
                                 trade_agg,  # 已经完成的交易量
                                 on=SymbolName.ORDERID,
                                 how='left')
        active_orders[VolumeName.TRANSVOL].fillna(0, inplace=True)
        active_orders[VolumeName.FILLEDVOL] += active_orders[VolumeName.TRANSVOL]
        del active_orders[VolumeName.TRANSVOL]
        # 确定哪些订单没有完成成交， 并继续成交
        l_unfinish = (active_orders[VolumeName.TARGETVOL] -
                      active_orders[VolumeName.FILLEDVOL]).abs() > tol
        active_orders = active_orders.loc[l_unfinish]
        return (active_orders, None, trade_data_raw)

    def volume_limit_filter(self, order_data, volume_limit, tol):
        cols = order_data.columns
        # 当天限额
        order_data['volume_limit'] = volume_limit
        # 需要成交的订单量
        order_data['volume_totrade'] = order_data[VolumeName.TARGETVOL] - \
            order_data[VolumeName.FILLEDVOL]
        # 需要成交的订单量的绝对值
        order_data['volumeabs_totrade'] = order_data['volume_totrade'].abs()
        # 按标的代码累加需要成交的订单量的绝对值
        order_data.sort_values([SymbolName.CODE, SymbolName.ORDERSEQ],
                               ascending=True, inplace=True)
        order_data['volumeabs_totrade_cumsum'] = order_data.groupby(
            SymbolName.CODE)['volumeabs_totrade'].cumsum()
        # 真实的累积成交量绝对值要小于订单的限额
        order_data['volumeabs_tradable_cumsum'] = order_data[
            ['volume_limit', 'volumeabs_totrade_cumsum']].min(axis=1)
        # 真实的累积成交量绝对值还原为真实可成交的订单量绝对值
        order_data['volumeabs_tradable'] = order_data.groupby(SymbolName.CODE, group_keys=False)[
            'volumeabs_tradable_cumsum'].apply(lambda x: x-x.shift(1).fillna(0)).values
        l_trade_able = order_data['volumeabs_tradable'] > tol
        # 更改订单的VolumeName.TARGETVOL，返回可成交的订单数据
        order_data[VolumeName.TARGETVOL] = order_data[VolumeName.FILLEDVOL] + \
            order_data['volumeabs_tradable'] * \
            np.sign(order_data['volume_totrade'])
        return(order_data.loc[l_trade_able, cols])

        # 需要完成的订单量
        # trade_volume = order_data[VolumeName.TARGETVOL].values - \
        #     order_data[VolumeName.FILLEDVOL].values

        # # 限制后的待完成的订单量
        # volume_amount = np.abs(trade_volume)
        # volume_sign = np.sign(trade_volume)
        # trade_volume_limited = np.minimum(
        #     volume_amount, target_volume_limit) * volume_sign
        # # 重构订单并提交订单
        # order_data[VolumeName.TARGETVOL] = trade_volume_limited + \
        #     order_data[VolumeName.FILLEDVOL].values
        # l_trade = np.abs(trade_volume_limited) > tol  # 成交量大于0


class CBVWAPOrderFAK(CBVWAPOrder):
    """
    vwap订单。
    没有成交的部分直接取消（FAK型订单）
    """
    order_type = OrderType.VWAPFAK

    def on_close(self, order_data):
        # 把未完成的订单分配给取消订单的窗口
        unfilled_orders, _, trade_data_raw = super().on_close(order_data)
        return None, unfilled_orders, trade_data_raw


class CBRatioVWAPOrder(CBVWAPOrder):
    """
    VWAP订单。
    根据输入的比例计算订单的目标成交量。
    """
    order_type = OrderType.RATIOVWAP
    ratio_relative = 'ratio_relative'  # 输入数据必须包含的字段

    def send_orders(self,
                    data,  # 至少包含代码SymbolName.CODE、 ratio_relative两个字段
                    ratio_to_netasset: float = 1.0,  # 总的买入资金占净资产的比例
                    event_confirm_volume: str = 'send',  # 成交量什么时候确定下来
                    order_seq: int = None,  # 订单交易序号，序列号越小交易越优先，一般卖单比买单优先
                    ):
        """
        1. 订单根据占总资产比例ratio_asset确定订单的目标量
            a. 假设净资产20000， 标的和比例的对应关系为【A、B  C】：【1，2，3】
            b. 占净资产比例ratio_to_netasset = 10%；
            c. A的目标成交量 =  20000* 10% * (1/(1+2+3))
        2. 根据ratio_volume_limit确定成交金额的上限制；
        3. event_confirm_volume为成交量确定的事件，一旦确定后成交量就不变，订单成交的逻辑按照普通的vwap订单执行。
        """

        if self.ratio_relative not in data.columns:
            raise (ValueError(
                f"{self.order_type.value}的输入数据必须包含{self.ratio_relative}字段，代表订单之间的相对比例"))

        if sum(data[self.ratio_relative] < 0) > 0:
            raise (ValueError("比例市价单不能含有卖单"))

        # 创建一个列，用于初始订单生成
        data[VolumeName.TARGETVOL] = self.mkt_obj.min_send_buy_volume + 1
        order_data = super(CBVWAPOrder, self).send_orders(data=data,
                                                          order_seq=order_seq)
        order_data = self.observer.add_orderid(order_data)

        # 参数保存在self.observer.active_order_params中
        params = order_data.set_index(SymbolName.ORDERID)[[]]
        params['event_confirm_volume'] = event_confirm_volume
        params[self.ratio_relative] = ratio_to_netasset * \
            data[self.ratio_relative].values / sum(data[self.ratio_relative])
        self.save_order_params(params)

        # 从比例计算持仓
        target_price = order_data[PriceName.TARGETP].values
        order_data[VolumeName.TARGETVOL] = self.netasset_ratio_to_volume(
            params[self.ratio_relative], target_price)

        # 取消小买单的标的
        l_v = order_data[VolumeName.TARGETVOL] >= self.mkt_obj.min_send_buy_volume
        l_short = order_data[VolumeName.TARGETVOL] < 0
        order_data = order_data.loc[l_v | l_short].copy()

        if 'send' in event_confirm_volume:
            order_data[SymbolName.ORDERTYPE] = OrderType.VWAP.value
        return order_data

    def on_close(self, order_data):
        """
        1. 根据比例以及vwap价格更新持仓
        2. 订单按照走CBVWAPOrder的交易逻辑执行
        3. 更改订单类型至vwap
        """
        # 获取订单参数
        params = self.get_order_params(order_data[SymbolName.ORDERID])
        # 按照比例和vwap数据计算持仓
        target_price = self.mkt_obj.current_frame.loc[
            order_data[SymbolName.CODE], PriceName.VWAP].values
        order_data[VolumeName.TARGETVOL] = self.netasset_ratio_to_volume(
            params[self.ratio_relative], target_price)

        # 更新订单类型
        order_data[SymbolName.ORDERTYPE] = OrderType.VWAP.value

        # 执行交易
        unfilled_orders, _, trade_data_raw = super().on_close(order_data)
        return unfilled_orders, _, trade_data_raw

    def on_open(self, order_data):
        """
        1. 根据比例更新持仓
        2. 筛选包含open字样的订单， 更改订单类型
        """
        # 获取订单参数
        params = self.get_order_params(order_data[SymbolName.ORDERID])
        # 按照比例计算持仓
        target_price = self.mkt_obj.current_frame.loc[
            order_data[SymbolName.CODE], PriceName.SETTLEP].values
        order_data[VolumeName.TARGETVOL] = self.netasset_ratio_to_volume(
            params[self.ratio_relative], target_price)

        # 更新订单类型
        l_open = params['event_confirm_volume'].strcontains('open').values
        order_data.loc[l_open, SymbolName.ORDERTYPE] = OrderType.VWAP.value
        return order_data, None, None


class CBTargetVolumeVWAP(CBVWAPOrder):
    """
    VWAP订单。
    根据输入的目标持仓量计算订单。
    """
    order_type = OrderType.TARGETVOLUMEVWAP

    def send_orders(self,
                    data,  # 至少包含代码SymbolName.CODE、 两个字段
                    order_seq_short: int = None,  # 空单交易序号
                    order_seq_long: int = None,  # 多单交易序号
                    **kwargs):
        """
        1. targetVolume是目标持仓量，而不是目标交易量
        2. 订单和持仓对比，交易差值的部分
        """
        tol = 1e-6  # 订单量阈值
        if data[SymbolName.CODE].nunique() != data.shape[0]:
            raise (ValueError('输入的标的有重复值'))

        position = self.observer.position.query(
            f"{SymbolName.SECUTYPE} == @self.secu_type.value").copy()
        position = position.groupby(SymbolName.CODE, as_index=False)[
            [VolumeName.VOLUME, VolumeName.FROZENVOL]].sum()

        data = pd.merge(data[[SymbolName.CODE, VolumeName.TARGETVOL]],
                        position, on=SymbolName.CODE, how='outer')
        data.fillna(0, inplace=True)
        data[VolumeName.TARGETVOL] -= data[VolumeName.VOLUME] - \
            data[VolumeName.FROZENVOL]

        l_long = data[VolumeName.TARGETVOL] > tol
        l_short = data[VolumeName.TARGETVOL] < -1 * tol
        order_data1 = super().send_orders(data=data.loc[l_long],
                                          order_seq=order_seq_long)
        order_data2 = super().send_orders(data=data.loc[l_short],
                                          order_seq=order_seq_short)
        order_data = pd.concat((order_data1, order_data2))
        order_data[SymbolName.ORDERTYPE] = OrderType.VWAP.value
        return order_data


class CBTargetRatioVWAP(CBTargetVolumeVWAP):
    """
    VWAP订单。
    根据输入的目标持仓比例计算订单。
    因为真实的交易会发生很多意外情况比如交易费率，无法卖出等（情况较少），
    所以基于目标持仓比例的计算方式是近似精确的.
    """
    order_type = OrderType.TARGETRATIOVWAP
    ratio_relative = 'ratio_relative'

    def send_orders(self,
                    data,  # 至少包含代码SymbolName.CODE、 两个字段
                    ratio_to_netasset: float = 1.0,  # 目标持仓市值占净资产的比例
                    order_seq_short: int = None,  # 空单交易序号
                    order_seq_long: int = None,  # 多单交易序号
                    **kwargs):
        """
        1. targetVolume是目标持仓量，而不是目标交易量
        2. 订单和持仓对比，交易差值的部分
        """
        if self.ratio_relative not in data.columns:
            raise(ValueError(
                f"{self.order_type.value}的输入数据必须包含{self.ratio_relative}字段，代表订单之间的相对比例"))

        if sum(data[self.ratio_relative] < 0) > 0:
            raise (ValueError("比例市价单不能含有卖单"))

        target_ratio = ratio_to_netasset * \
            data[self.ratio_relative].values / sum(data[self.ratio_relative])

        target_price = self.mkt_obj.current_frame.loc[
            data[SymbolName.CODE], PriceName.SETTLEP].values

        data[VolumeName.TARGETVOL] = self.netasset_ratio_to_volume(
            target_ratio, target_price)

        order_data = super().send_orders(data=data,
                                         order_seq_short=order_seq_short,
                                         order_seq_long=order_seq_long,
                                         **kwargs)

        return order_data


# %% 其它订单策略


class StockLimitOrder(OrderStrategyBase):
    pass


class StockStopOrder(OrderStrategyBase):
    pass
