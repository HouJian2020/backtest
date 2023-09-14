# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:53:00 2023

@author: houjian
"""
import data_model
from typing import List
from base_api import Event, MarketBase, OrderStrategyBase
from constant import SecuType, SymbolName, TimeName, PriceName, VolumeName, OtherName


class ETFMarket(MarketBase):
    """
    1. ETF市场， 封装该市场的相关数据
        （交易费率， 行情推动事件， 标的行情数据）。
    2. 标的行情数据占用内存较大，因此数据被所有角色（PM，observer， trader）共享，
        建议只可读取，不建议修改。
    """

    secu_type: SecuType = SecuType.STOCK  # 证券类型
    buy_fee_ratio = 0.0005  # 买入手续费率
    sell_fee_ratio = 0.0005  # 卖出手续费率, ETF没有印花税

    def __init__(self, start_date: str = '2015-01-01'):
        # 市场行情数据
        self.get_data(start_date=start_date)
        self.trigger_event: List[Event] = []  # 行情推动事件集合

    def get_data(self, start_date):
        """
        获市场行情数据， 格式要参考data_model.ETF_MKTDATA.
        初次配置要根据使用者的数据库结构做数据结构转化
        """

        # from findata import get_data
        # import os
        # df_data = get_data.ETFPV()
        # columns = ['etfCode', 'tradeDate', 'open', 'high', 'low', 'close',
        #            'volume', 'amount', 'adjFactor']
        # df_data = df_data[columns].copy()
        # root_path = r'D:\jupyter_notebook\同步盘\backtest'
        # df_data.to_pickle(os.path.join(root_path, 'data_mock', 'etf.pkl'))

        import os
        import pandas as pd
        root_path = os.path.abspath('..')
        data_path = os.path.join(root_path, 'data_mock', 'etf.pkl')
        df_data = pd.read_pickle(data_path)

        self.data = data_model.init_pandas_data(data_model.ETF_MKTDATA)
        self.data[SymbolName.CODE] = df_data['etfCode']
        self.data[TimeName.TDATE] = df_data['tradeDate']
        self.data[PriceName.OPENADJ] = df_data['open'] * df_data['adjFactor']
        self.data[PriceName.HIGHADJ] = df_data['high'] * df_data['adjFactor']
        self.data[PriceName.LOWADJ] = df_data['low'] * df_data['adjFactor']
        self.data[PriceName.CLOSEADJ] = df_data['close'] * df_data['adjFactor']
        self.data[PriceName.VWAPOPENADJ] = df_data['open'] * \
            df_data['adjFactor']
        self.data[PriceName.VWAPCLOSEADJ] = df_data['close'] * \
            df_data['adjFactor']
        # ETF停牌标志是成交额为null
        self.data[OtherName.IFSUSEPEND] = df_data['amount'].isnull()
        self.data[VolumeName.VOLUME] = df_data['volume'].fillna(0)
        self.data[VolumeName.AMOUNT] = df_data['amount'].fillna(0)
        self.data.sort_values([SymbolName.CODE, TimeName.TDATE], inplace=True)
        # 模拟的涨停价、跌停价， 根据本地接口调整
        self.data['pre_close'] = self.data.groupby(SymbolName.CODE)[
            PriceName.CLOSEADJ].shift(1)
        self.data[PriceName.DOWNLIMIT] = self.data['pre_close']*1.10
        self.data[PriceName.UPLIMIT] = self.data['pre_close']*0.90
        del self.data['pre_close']
        self.data = self.data.query(f"{TimeName.TDATE} >= @start_date").copy()

    def init_trigger_event(self):
        """
        初始化.
        """
        pass

    def on_trigger(self, event: Event):
        """
        市场行情，last_frame更新.current_frame更新.
        """

        pass

    def settlement(self, position):
        """
        结算当前持仓.
        """
        pass
