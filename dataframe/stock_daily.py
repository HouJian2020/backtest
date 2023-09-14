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
import os
import numpy as np
import pandas as pd
from . import data_model
from .workers import OrderStrategyPandas
from ..base_api import Event, MarketBase, OrderStrategyBase
from ..constant import SecuType, OrderType, EventType, SymbolName, TimeName,\
    PriceName, VolumeName, OtherName, Status, Direction

# %% 具体市场类


class StockMarket(MarketBase):
    """
    1. 股票市场， 封装该市场的相关数据
        （交易费率， 行情推动事件， 行情数据）。
    2. 标的行情数据占用内存较大，因此数据被所有角色（PM，observer， trader）共享，
        建议只可读取，不建议修改。
    """

    secu_type: SecuType = SecuType.STOCKDAILY  # 证券类型
    buy_fee_ratio = 0.0005  # 买入手续费率
    sell_fee_ratio = 0.0015  # 卖出手续费率
    vwap_volume_limit_ratio: float = 0.05  # 单票当日vwap订单成交量占比不超过5%
    data_path = r'D:\mypkgs\backtest\data_mock'

    def __init__(self, start_date: str = '2015-01-01'):
        # 加载市场行情数据至self.data中
        self.get_data(start_date=start_date)
        # 计算self.trigger_events(行情推动事件集合)
        self.init_trigger_events()

    def get_data(self, start_date, renew=False):
        """
        获市场行情数据， 格式要参考data_model.STOCK_MKTDATA.
        模拟的A股市场数据，根据使用者的数据库结构做数据结构转化
        """

        data_path = os.path.join(self.data_path, 'stock.pkl')
        df_data = pd.read_pickle(data_path)

        self.data = data_model.init_pandas_data(data_model.STOCK_MKTDATA)
        self.data[SymbolName.CODE] = df_data['code']
        self.data[TimeName.TDATE] = df_data['tradeDate']
        self.data[PriceName.OPENADJ] = df_data['open'] * df_data['adjFactor']
        self.data[PriceName.HIGHADJ] = df_data['high'] * df_data['adjFactor']
        self.data[PriceName.LOWADJ] = df_data['low'] * df_data['adjFactor']
        self.data[PriceName.CLOSEADJ] = df_data['close'] * df_data['adjFactor']
        self.data[PriceName.VWAPOPENADJ] = df_data['open'] * \
            df_data['adjFactor']
        self.data[PriceName.VWAPCLOSEADJ] = df_data['close'] * \
            df_data['adjFactor']
        self.data[VolumeName.VOLUME] = df_data['volume']
        self.data[VolumeName.AMOUNT] = df_data['amount']
        self.data[OtherName.IFSUSEPEND] = df_data['Ifsuspend'] == 1
        self.data.sort_values([SymbolName.CODE, TimeName.TDATE], inplace=True)
        # 模拟的涨停价、跌停价， 根据本地接口调整
        self.data['pre_close'] = self.data.groupby(SymbolName.CODE)[
            PriceName.CLOSEADJ].shift(1)
        self.data[PriceName.DOWNLIMIT] = self.data['pre_close']*0.9
        self.data[PriceName.UPLIMIT] = self.data['pre_close']*1.1
        del self.data['pre_close']
        self.data = self.data.query(f"{TimeName.TDATE} >= @start_date").copy()

    def init_trigger_events(self):
        """
        初始化行情推动事件.
        同一个市场不同的脚本行情推动事件是不一样的
        如daily_stock.py中行情推动事件是开盘和收盘，
        但是tick_stock.py中行情推动事件是tick.
        如果要做不同精细程度的回测需要增加新的脚本.
        """
        data_path = os.path.join(self.data_path, 'stock_trade_date.csv')
        trade_dates = pd.read_csv(data_path, parse_dates=[
                                  'tradeDate'])['tradeDate']

        # 根据市场数据的范围限制行情驱动的事件范围
        l1 = trade_dates >= self.data[TimeName.TDATE].min()
        l2 = trade_dates <= self.data[TimeName.TDATE].max()

        self.trigger_events = []
        for trade_date in trade_dates.loc[l1 & l2]:
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
        self.trade_dates = trade_dates
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
                PriceName.VWAPCLOSEADJ, VolumeName.VOLUME, VolumeName.AMOUNT],
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

# %% 具体订单策略类


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

        return order_data


class StockMarketOrder(VolumeOrderStrategy):
    """
    市价单策略
    1. 交易时如果资金不足则在买单中按比例分配。
    2. 未交易成功的部分，订单会被保留等待成交。
    """

    secu_type = SecuType.STOCKDAILY
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
        tol = 1e-6  # 浮点数之间进行判断的容忍度

        info = self.mkt_obj.current_frame.loc[order_data[SymbolName.CODE]].copy(
        )

        # 是否未提交订单
        l_unsubmit = order_data[SymbolName.STATUS] == Status.UNSUBMIT.value
        l_long = order_data[VolumeName.TARGETVOL] > 0
        l_suspend = info[OtherName.IFSUSEPEND].values  # 是否停牌
        # 当天是否有机会买入
        l_limit_buy = info[PriceName.LOWADJ] >= info[PriceName.UPLIMIT] - tol
        # 当天是否有机会卖出
        l_limit_sell = info[PriceName.HIGHADJ] <= info[PriceName.DOWNLIMIT] + tol
        # 收盘是否有机会买入
        l_limit_buy_close = info[PriceName.CLOSEADJ] >= info[PriceName.UPLIMIT] - tol
        # 收盘是否有机会卖出
        l_limit_sell_close = info[PriceName.CLOSEADJ] <= info[PriceName.DOWNLIMIT] + tol

        # 四种情况下是否可交易
        l_1 = l_unsubmit & l_long & (
            ~l_limit_buy_close).values  # 未提交状态的市价单默认收盘提交
        l_2 = l_unsubmit & (~l_long) & (~l_limit_sell_close).values
        l_3 = (~l_unsubmit) & l_long & (~l_limit_buy).values
        l_4 = (~l_unsubmit) & (~l_long) & (~l_limit_sell).values

        l_trade = (l_1 | l_2 | l_3 | l_4) & (~l_suspend)  # 当天不停牌方可交易

        # 已经提交的订单以开盘价交易， 尚未提交的订单以收盘价交易
        order_data[PriceName.TARGETP] = info[PriceName.OPENADJ].values
        order_data.loc[l_unsubmit, PriceName.TARGETP] = info.loc[
            l_unsubmit.values, PriceName.CLOSEADJ].values

        # 改变订单状态，状态为"unsubmit"的订单改为"active"
        order_data.loc[l_unsubmit, SymbolName.STATUS] = Status.ACTIVE.value

        # 执行可交易的订单
        trade_data_raw, unfinished_order = self.order_to_trade(
            order_data.loc[l_trade])

        # 更新不可执行的订单， 并更新订单的PriceName.TARGETP
        untraded_orders = pd.concat(
            (order_data.loc[~l_trade], unfinished_order), ignore_index=True)
        untraded_orders[PriceName.TARGETP] = self.mkt_obj.current_frame.loc[
            untraded_orders[SymbolName.CODE], PriceName.SETTLEP].values
        return (untraded_orders, None, trade_data_raw)

    def on_open(self, order_data):
        """
        开盘:
        1. 因为开盘前是非交易事件，所以能否成交和订单状态是否为“UNSUBMIT”没有关系。
        2. 可成交订单走order_to_trade方法执行交易。
        3. 不可成交的订单修改订单状态后返回给observer中的active_orders中。
        """
        tol = 1e-6  # 浮点数之间进行判断的容忍度

        info = self.mkt_obj.current_frame.loc[order_data[SymbolName.CODE]].copy(
        )

        # 判断多空订单是否可交易
        l_long = order_data[VolumeName.TARGETVOL] > 0
        l_suspend = info[OtherName.IFSUSEPEND].values  # 是否停牌
        # 开盘是否有机会买入
        l_limit_buy_open = info[PriceName.OPENADJ] >= info[PriceName.UPLIMIT] - tol
        # 开盘是否有机会卖出
        l_limit_sell_open = info[PriceName.OPENADJ] <= info[PriceName.DOWNLIMIT] + tol

        l_1 = l_long & (~l_limit_buy_open).values  # 未提交状态的市价单默认收盘提交
        l_2 = (~l_long) & (~l_limit_sell_open).values
        l_trade = (l_1 | l_2) & (~l_suspend)  # 当天不停牌方可交易

        # 订单以开盘价交易
        order_data[PriceName.TARGETP] = info[PriceName.OPENADJ].values

        # 改变订单状态，状态为"unsubmit"的订单改为"active"
        l_unsubmit = order_data[SymbolName.STATUS] == Status.UNSUBMIT.value
        order_data.loc[l_unsubmit, SymbolName.STATUS] = Status.ACTIVE.value

        # 执行可交易的订单
        trade_data_raw, unfinished_order = self.order_to_trade(
            order_data.loc[l_trade])

        # 更新不可执行的订单， 并更新订单的PriceName.TARGETP
        untraded_orders = pd.concat(
            (order_data.loc[~l_trade], unfinished_order), ignore_index=True)
        untraded_orders[PriceName.TARGETP] = self.mkt_obj.current_frame.loc[
            untraded_orders[SymbolName.CODE], PriceName.SETTLEP].values
        return (untraded_orders, None, trade_data_raw)


class StockMarketOrderFAK(StockMarketOrder):
    """
    market订单。
    没有成交的部分直接取消（FAK型订单）
    """
    order_type = OrderType.MARKETFAK

    def on_close(self, order_data):
        unfilled_orders, _, trade_data_raw = super().on_close(order_data)
        return None, unfilled_orders, trade_data_raw

    def on_open(self, order_data):
        unfilled_orders, _, trade_data_raw = super().on_open(order_data)
        return None, unfilled_orders, trade_data_raw


class StockRatioMarketOrder(StockMarketOrder):
    """
    按净资产的一定比例成交的买单策略。
    1. 把比例参数存储到观察者订单参数字典当中（self.observer.active_order_params）；
    2. 如果成交时资金不足则按照比例买入；
    注意：订单中仅含买单。
    """

    order_type = OrderType.RATIOMARKET
    ratio_relative = 'ratio_relative'  # 输入数据必须包含的字段

    def send_orders(self,
                    data,  # 至少包含代码SymbolName.CODE、 ratio_netasset两个字段
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
        3. event_confirm_volume为成交量确定的事件，一旦确定后成交量就不变，订单成交的逻辑按照普通的vwap订单执行。
        """

        if self.ratio_relative not in data.columns:
            raise (ValueError(
                f"{self.order_type.value}的输入数据必须包含{self.ratio_relative}字段，代表订单之间的相对比例"))

        if sum(data[self.ratio_relative] < 0) > 0:
            raise (ValueError("比例市价单不能含有卖单"))

        data[VolumeName.TARGETVOL] = 100
        order_data = super(StockMarketOrder, self).send_orders(
            data=data, order_seq=order_seq)
        order_data = self.observer.add_orderid(order_data)

        # 参数保存在self.observer.active_order_params中
        params = order_data.set_index(SymbolName.ORDERID)[[]]
        params[self.ratio_relative] = ratio_to_netasset * \
            data[self.ratio_relative].values / sum(data[self.ratio_relative])

        # 从比例计算目标仓位
        target_price = order_data[PriceName.TARGETP].values
        order_data[VolumeName.TARGETVOL] = self.netasset_ratio_to_volume(
            params[self.ratio_relative], target_price)

        # 是否此刻确定目标成交量
        if not confirm_volume:
            self.save_order_params(params)
        else:
            order_data[SymbolName.ORDERTYPE] = OrderType.MARKET.value
        return order_data

    def on_close(self, order_data):
        """
        收盘:
        1. 交易前更新成交量。
        2. 交给vwap继续完成订单。
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
        开盘:
        1. 交易前更新成交量。
        2. 可成交订单走order_to_trade方法执行交易。
        3. 不可成交的订单直接作废。
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
        unfilled_orders, _, trade_data_raw = super().on_open(order_data)
        return unfilled_orders, _, trade_data_raw


class StockRatioVWAPOrder(OrderStrategyPandas):
    """
    按VWAP成交订单策略， 成交量不得超过当日成交量的某个比例。
    1. 把比例参数存储到观察者订单参数字典当中（self.observer.active_order_params）；
    2. 如果成交时资金不足则按照比例买入；
    3. 订单中没成交的部分直接取消订单。
    注意：订单中仅含买单。
    """

    secu_type = SecuType.STOCKDAILY
    order_type = OrderType.RATIOVWAP
    ratio_relative = 'ratio_relative'  # 输入数据必须包含的字段

    def send_orders(self,
                    data,  # # 至少包含代码SymbolName.CODE、 成交比例self.ratio_relative两个字段
                    ratio_to_netasset: float = 1.0,  # 总的买入资金占净资产的比例
                    vwap_volume_limit_ratio: float = 0.05,  # 成交金额占当日的成交比例上限
                    order_seq: int = 11,  # 订单交易序号，序列号越小交易越优先，一般卖单比买单优先
                    ):
        """
        根据ratio以及当前标的的结算价确定买入标的的量.
        """

        if sum(data[self.ratio_relative] < 0) > 0:
            raise (ValueError("比例市价单不能含有卖单"))

        # 初始化订单
        order_data = self.new_order(data.shape[0])

        # 编辑订单信息
        order_data[SymbolName.CODE] = data[SymbolName.CODE].values
        order_data[SymbolName.ORDERSEQ] = order_seq
        order_data[SymbolName.DIRECTION] = Direction.LONG.value

        # 市价单用结算价替代目标价格
        info = self.mkt_obj.current_frame.loc[data[SymbolName.CODE]]
        order_data[PriceName.TARGETP] = info[PriceName.SETTLEP].values

        # 按照比例计算目标成交比例
        order_data = self.observer.add_orderid(order_data)
        order_data[self.ratio_relative] = ratio_to_netasset * \
            data[self.ratio_relative].values / sum(data[self.ratio_relative])

        # 参数保存在self.observer.active_order_params中
        order_data['vwap_volume_limit_ratio'] = vwap_volume_limit_ratio
        params = order_data.set_index(SymbolName.ORDERID)[[
            self.ratio_relative, 'vwap_volume_limit_ratio']].to_dict(orient='index')
        self.observer.active_order_params.update(params)

        # 计算目标交易量
        order_data[VolumeName.TARGETVOL] = self.netasset_ratio_to_volume(
            order_data[self.ratio_relative], order_data[PriceName.TARGETP])
        del order_data[self.ratio_relative], order_data['vwap_volume_limit_ratio']

        return(order_data)

    def on_close(self, order_data):
        """
        收盘:
        1. 交易前更新成交量。
        2. 可成交订单走order_to_trade方法执行交易。
        3. 不可成交的订单直接作废。
        """
        tol = 1e-6  # 浮点数之间进行判断的容忍度

        info = self.mkt_obj.current_frame.loc[order_data[SymbolName.CODE]].copy(
        )
        order_data[PriceName.TARGETP] = info[PriceName.VWAPCLOSEADJ].values

        # 读取参数
        params = pd.DataFrame.from_dict(
            {k: self.observer.active_order_params[k] for k in
             order_data[SymbolName.ORDERID]}, orient='index')
        # 虽然字典是有序的，重新排序后会更加保险
        params = params.loc[order_data[SymbolName.ORDERID]]

        # 计算目标成交量（限制）
        # TODO 严格来讲需要根据vwap价格对应的区间成交量限制， 我们这里做一个不严谨的替代， 用全天成交量替代对应区间的成交量
        target_volume_limit = params['vwap_volume_limit_ratio'].values * \
            info[VolumeName.VOLUME].values
        target_volume = self.netasset_ratio_to_volume(
            params[self.ratio_relative], order_data[PriceName.TARGETP])
        order_data[VolumeName.TARGETVOL] = np.minimum(
            target_volume, target_volume_limit)

        # 成交量大于0即为可交易
        l_trade = order_data[VolumeName.TARGETVOL] > tol  # 成交量大于0

        # 执行可交易的订单
        trade_data_raw, unfinished_order = self.order_to_trade(
            order_data.loc[l_trade])

        # 没有执行成功的订单自动取消
        return (order_data.head(0), order_data.loc[~l_trade], trade_data_raw)

    def on_open(self, order_data):
        """
        开盘:
        1. 交易前更新成交量。
        2. 可成交订单走order_to_trade方法执行交易。
        3. 不可成交的订单直接作废。
        """
        tol = 1e-6  # 浮点数之间进行判断的容忍度

        info = self.mkt_obj.current_frame.loc[order_data[SymbolName.CODE]].copy(
        )
        order_data[PriceName.TARGETP] = info[PriceName.VWAPOPENADJ].values

        # 读取参数
        params = pd.DataFrame.from_dict(
            {k: self.observer.active_order_params[k] for k in
             order_data[SymbolName.ORDERID]}, orient='index')
        # 虽然字典是有序的，重新排序后会更加保险
        params = params.loc[order_data[SymbolName.ORDERID]]

        # 计算目标成交量（限制）
        # TODO 严格来讲需要根据vwap价格对应的区间成交量限制， 我们这里做一个不严谨的替代， 用全天成交量替代对应区间的成交量
        target_volume_limit = params['vwap_volume_limit_ratio'].values * \
            info[VolumeName.VOLUME].values
        target_volume = self.ratio_to_volume(
            params[self.ratio_relative], order_data[PriceName.TARGETP]).values
        order_data[VolumeName.TARGETVOL] = np.minimum(
            target_volume, target_volume_limit)

        # 成交量大于0即为可交易
        l_trade = order_data[VolumeName.TARGETVOL] > tol  # 成交量大于0

        # 执行可交易的订单
        trade_data_raw, unfinished_order = self.order_to_trade(
            order_data.loc[l_trade])

        # 没有执行成功的订单自动取消
        return (order_data.head(0), order_data.loc[~l_trade], trade_data_raw)


class StockLimitOrder(OrderStrategyBase):
    pass


class StockStopOrder(OrderStrategyBase):
    pass
