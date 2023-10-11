# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 09:04:38 2023

@author: houjian
"""
import importlib
import pandas as pd
from typing import List
from datetime import datetime, timedelta

from ..constant import EventType
from ..base_api import TradeSessionBase, MarketBase, OrderStrategyBase, \
    Event, event_generator
from .workers import TraderPandas, ObserverPandas, PMPandas


class TradeSession(TradeSessionBase):
    def __init__(self,
                 PortfolioManager: PMPandas = None,  # 基金经理（策略）
                 cash=1e8,  # 默认初始金额为一个小目标
                 start_datetime='2016-01-01',  # 回测的开始日期
                 end_datetime=None,  # 回测的结束日期
                 markets='cb_daily',  # 回测所需的市场对象
                 mkt_startdate='2012-01-01',  # 市场对象实例化的开始日期
                 **kwargs):  # 其它参数

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.kwargs = kwargs

        super().__init__(cash=cash,
                         start_datetime=self.start_datetime,
                         end_datetime=self.end_datetime,
                         Observer=ObserverPandas,
                         PortfolioManager=PortfolioManager,
                         Trader=TraderPandas)

        # 添加市场实例
        self.reset_markets(markets=markets, mkt_startdate=mkt_startdate)

        # 更新记录事件
        self.init_records()

    def reset_markets(self, markets, mkt_startdate: str):

        self.del_all_markets()  # 重置市场列表和事件列表

        if isinstance(markets, str):
            markets = [markets]

        for mkt_str in set(markets):  # 注意去重， 防止重复输入
            self.add_mkt_module(mkt_str, mkt_startdate)

    def add_mkt_module(self, mkt_str, mkt_startdate: str):
        """
        根据模块名称加载市场类和订单策略要求：
        1. 市场类和订单策略类放入同一个脚本中；
        2. 市场类和订单策略类都要实现各自的类变量；
        """
        module = importlib.import_module(
            f".{mkt_str}", package=__name__)
        for method_str in dir(module):
            if not method_str.startswith('_'):
                method = getattr(module, method_str)
                # 获取模块中有市场类型的市场类
                if isinstance(method, type) and issubclass(method, MarketBase):
                    if method.secu_type is not None:
                        for key, value in self.kwargs.items():
                            if key.startswith(mkt_str):
                                name = key[len(mkt_str)+1:]
                                setattr(method, name, value)
                        mkt = method(mkt_startdate)  # 市场数据初始化
                        self.add_market(mkt)

                # 获取模块中有订单类型的订单策略类
                if isinstance(method, type) and issubclass(
                        method, OrderStrategyBase):
                    if method.order_type is not None:
                        self.order_strategys.update([method])

    def init_records(self):
        self.del_events(EventType.RECORD)
        # TODO 按照第一个市场的交易日做记录
        mkt = self.market_list[0]
        trade_dates = mkt.trade_dates

        l1 = trade_dates >= self.start_datetime
        l2 = trade_dates <= self.end_datetime

        events = []
        for trade_date in trade_dates.loc[l1 & l2]:
            day = trade_date.to_pydatetime()
            event = Event(
                # 记录时间为20：00
                datetime=day + timedelta(hours=20),
                event_seq=10,
                is_trigger=False,
                event_type=EventType.RECORD,)
            events.append(event)
        self.add_events(events)

    def reset_records(self, events: List[Event]):
        self.del_events(EventType.RECORD)
        self.add_events(events)

    def reset_events(self):
        super().reset_events()
        self.init_records()

    @property
    def start_datetime(self):
        return (self._start_datetime)

    @start_datetime.setter
    def start_datetime(self, start_datetime):
        if isinstance(start_datetime, str):
            start_datetime = datetime.strptime(start_datetime, "%Y-%m-%d")
        self._start_datetime = start_datetime

    @property
    def end_datetime(self):
        return (self._end_datetime)

    @end_datetime.setter
    def end_datetime(self, end_datetime):
        if end_datetime is None:
            end_datetime = datetime.now()

        if isinstance(end_datetime, str):
            end_datetime = datetime.strptime(end_datetime, "%Y-%m-%d")
            end_datetime = end_datetime + timedelta(days=1)  # 包含今天
        self._end_datetime = end_datetime
