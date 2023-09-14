"""
用于回测平台的基本的数据格式.
基于日频行情数据的回测系统设计大量的向量化操作，使用DataFrame是比较合适的数据类型
"""


import pandas as pd
from ..constant import SymbolName, PriceName, VolumeName, TimeName, OtherName
from types import MappingProxyType


class _PandasDataType:
    """
    以字典格式记录以DataFrame为存储格式的数据格式.
    key代表DataFrame的列， value代表DataFrame的数据类型.
    """
    # 股票市场行情的数据格式
    stock_mktdata = {
        SymbolName.CODE: 'object',
        TimeName.TDATE: 'datetime64[ns]',
        PriceName.OPENADJ: 'float64',
        PriceName.HIGHADJ: 'float64',
        PriceName.LOWADJ: 'float64',
        PriceName.CLOSEADJ: 'float64',
        PriceName.VWAPOPENADJ: 'float64',
        PriceName.VWAPCLOSEADJ: 'float64',
        VolumeName.VOLUME: 'float64',
        VolumeName.AMOUNT: 'float64',
        OtherName.IFSUSEPEND: 'bool',
        PriceName.DOWNLIMIT: 'float64',
        PriceName.UPLIMIT: 'float64',
    }

    # 可转债市场行情的数据格式
    cb_mktdata = {
        SymbolName.CODE: 'object',
        TimeName.TDATE: 'datetime64[ns]',
        PriceName.OPENADJ: 'float64',
        PriceName.HIGHADJ: 'float64',
        PriceName.LOWADJ: 'float64',
        PriceName.CLOSEADJ: 'float64',
        PriceName.VWAPADJ: 'float64',
        VolumeName.VOLUME: 'float64',
        VolumeName.AMOUNT: 'float64',
    }

    # ETF市场行情的数据格式
    etf_mktdata = {
        SymbolName.CODE: 'object',
        TimeName.TDATE: 'datetime64[ns]',
        PriceName.OPENADJ: 'float64',
        PriceName.HIGHADJ: 'float64',
        PriceName.LOWADJ: 'float64',
        PriceName.CLOSEADJ: 'float64',
        PriceName.VWAPOPENADJ: 'float64',
        PriceName.VWAPCLOSEADJ: 'float64',
        VolumeName.VOLUME: 'float64',
        VolumeName.AMOUNT: 'float64',
        OtherName.IFSUSEPEND: 'bool',
        PriceName.DOWNLIMIT: 'float64',
        PriceName.UPLIMIT: 'float64',
    }

    # 订单的数据格式
    order_data = {
        # 同一笔订单是交易员的最小操作单位， 但是可以对应多笔成交（订单不是市场的最小操作单位）
        # 造成ORDERID不唯一的唯一逻辑就是交易
        SymbolName.ORDERID: 'int',
        SymbolName.SECUTYPE: 'object',
        SymbolName.CODE: 'object',
        TimeName.ORDERTIME: 'datetime64[ns]',  # 提交订单的时间
        SymbolName.ORDERTYPE: 'object',  # 订单类型，限价、 市价、 vwap、 stop
        SymbolName.DIRECTION: 'object',  # 方向，多、空、平
        PriceName.TARGETP: 'float64',  # 订单的价格， 部分订单有效，如限价单和stop
        VolumeName.TARGETVOL: 'float64',  # 订单要完成的量
        VolumeName.FILLEDVOL: 'float64',  # 已经完成的成交订单量
        SymbolName.OFFSETID: 'int',  # 平仓对应的交易ID
        SymbolName.STATUS: 'object',  # 订单的状态， 配合订单策略的字段
        SymbolName.ORDERSEQ: 'int',  # 订单的执行优先级
    }

    # 交易信息的数据格式
    trade_data = {
        SymbolName.TRADEID: 'int',
        SymbolName.ORDERID: 'int',
        SymbolName.SECUTYPE: 'object',
        SymbolName.CODE: 'object',
        TimeName.TRANSTIME: 'datetime64[ns]',  # 交易的时刻
        SymbolName.DIRECTION: 'object',  # 方向
        PriceName.TRANSP: 'float64',
        VolumeName.TRANSVOL: 'float64',
        SymbolName.OFFSETID: 'int',  # 平仓对应的交易ID
        VolumeName.TRANSCOST: 'float64',
    }

    # 持仓信息的数据格式
    positon_data = {
        SymbolName.ORDERID: 'int',
        SymbolName.TRADEID: 'int',
        SymbolName.SECUTYPE: 'object',
        SymbolName.CODE: 'object',
        TimeName.TRANSTIME: 'datetime64[ns]',  # 交易的时刻
        PriceName.TRANSP: 'float64',
        PriceName.SETTLEP: 'float64',
        VolumeName.VOLUME: 'float64',   # 当前的持仓量
        VolumeName.FROZENVOL: 'float64',  # 冻结的持仓量（有活跃订单要卖出）
    }

    # 历史持仓信息的数据格式（相对持仓信息多了时间戳）
    positon_records = {
        TimeName.RECORDTIME: 'datetime64[ns]',
        SymbolName.ORDERID: 'int',
        SymbolName.TRADEID: 'int',
        SymbolName.SECUTYPE: 'object',
        SymbolName.CODE: 'object',
        TimeName.TRANSTIME: 'datetime64[ns]',  # 交易的时刻
        PriceName.TRANSP: 'float64',
        PriceName.SETTLEP: 'float64',
        VolumeName.VOLUME: 'float64',
        VolumeName.FROZENVOL: 'float64'
    }

    # 组合净值的数据格式（回测平台中以字典形式存储）
    balance_data = {
        VolumeName.NETASSET: 'float64',  # 净资产
        VolumeName.CASH: 'float64',  # 现金
        VolumeName.FROZENCASH: 'float64',  # 冻结的现金
    }
    # 历史组合净值信息的数据格式（相对组合净值数据多了时间戳）
    balance_records = {
        TimeName.RECORDTIME: 'datetime64[ns]',
        VolumeName.NETASSET: 'float64',
        VolumeName.CASH: 'float64',
        VolumeName.FROZENCASH: 'float64',
    }


STOCK_MKTDATA = MappingProxyType(_PandasDataType.stock_mktdata)
CB_MKTDATA = MappingProxyType(_PandasDataType.cb_mktdata)
ETF_MKTDATA = MappingProxyType(_PandasDataType.etf_mktdata)
ORDER_DATA = MappingProxyType(_PandasDataType.order_data)
TRADE_DATA = MappingProxyType(_PandasDataType.trade_data)
POSITION_DATA = MappingProxyType(_PandasDataType.positon_data)
POSITION_RECORDS = MappingProxyType(_PandasDataType.positon_records)
BALANCE_DATA = MappingProxyType(_PandasDataType.balance_data)
BALANCE_RECORDS = MappingProxyType(_PandasDataType.balance_records)


def init_pandas_data(data_format):
    df = pd.DataFrame(columns=list(data_format.keys()))
    df = df.astype(data_format)
    return (df)


if __name__ == '__main__':
    df = init_pandas_data(CB_MKTDATA)
    print(df.info())
    df = df.T
