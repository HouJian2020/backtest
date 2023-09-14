# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:43:59 2023

@author: houjian
"""
from enum import Enum, unique

# %% ConsName类（数据属性名称）


class ConsName:
    """
    全局变量名在回测平台的作用：
    1. DataFrame格式的数据 列名定义
    2. DataFrame格式的数据 基于列名的数值引用
    3. DataFrame格式的数据 索引的指定
    4. DataFrame格式的数据 排序规则的指定
    5. 因为类变量的值会作为 DataFrame的列名，因此命名的时候注意重名问题
    """
    pass


class SymbolName(ConsName):
    # 代码类
    SECUTYPE = 'secuType'  # 标的代码标志
    CODE = 'code'  # 标的代码标志

    # ID类
    ORDERID = 'orderID'
    TRADEID = 'tradeID'
    OFFSETID = 'offsetID'

    ORDERSEQ = 'orderSeq'  # 同时触发的订单执行顺序
    EVENTSEQ = 'eventSeq'  # 同事触发的事件的执行顺序

    # 其它
    STATUS = 'status'
    ORDERTYPE = 'orderType'
    DIRECTION = 'direction'


class PriceName(ConsName):
    # 价格类
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VWAP = 'vwap'
    UPLIMIT = 'upLimit'  # 涨停价标志
    DOWNLIMIT = 'downLimit'  # 停价标志
    ADJfACTOR = 'adjFactor'
    OPENADJ = 'openAdj'
    HIGHADJ = 'highAdj'
    LOWADJ = 'lowAdj'
    CLOSEADJ = 'closeAdj'
    VWAPADJ = 'vwapAdj'
    VWAPOPENADJ = 'vwapOpenAdj'
    VWAPCLOSEADJ = 'vwapCloseAdj'
    TRANSP = 'transPrice'
    SETTLEP = 'settlePrice'
    TARGETP = 'targetPrice'


class VolumeName(ConsName):
    # 成交量类
    VOLUME = 'volume'
    AMOUNT = 'amount'
    HSL = 'hsl'
    TRANSVOL = 'transVol'  # 成交量
    FILLEDVOL = 'filledVol'
    FROZENVOL = 'frozenVol'
    TARGETVOL = 'targetVol'
    NETASSET = 'netAsset'
    CASH = 'cash'
    FROZENCASH = 'frozenCash'
    TRANSCOST = 'transCost'
    RATIO = 'ratio'
    MKTV = 'mktv'  # 市值


class TimeName(ConsName):
    # 日期与时间类
    DATE = 'date'  # 自然日
    TDATE = 'tradeDate'  # 交易日
    TRANSTIME = 'transTime'  # 订单成交的时刻
    ORDERTIME = 'orderTime'
    RECORDTIME = 'recordTime'


class OtherName(ConsName):
    # # 索引与排序
    # INDEX = 'index'
    # SORTBY = 'sort_by'

    # 其它
    IFSUSEPEND = 'ifSuspend'  # 停牌标志
    TOCANCEL = "to_cancel"  # 需要取消的订单，市场行情更新时调用订单策略的on_cancel方法


# %% 枚举类（属性的值）
"""
ConName类可以看作DataFrame的列名(columns).
枚举类可以认为是DataFrame中的值(value).
"""


@unique
class Status(Enum):
    """
    订单状态.
    配合订单策略，对订单进行自定义的处理.
    """
    UNSUBMIT = "unsubmit"  # 未提交的订单
    ACTIVE = "active"  # 活跃订单
    TOCANCEL = "to_cancel"  # 处于取消状态的订单
    CANCELLED = "cancelled"  # 已经取消的订单


@unique
class SecuType(Enum):
    """
    证券类型.
    """
    STOCK = "stock"
    STOCKDAILY = "stock_daily"  # 日频交易的A股市场
    CBDAILY = "cb_daily"  # 日频交易的可转债市场
    FUTURES = "CTA"
    INDEX = "index"
    ETF = "ETF"
    BOND = "bond"
    FUND = "fund"


@unique
class EventType(Enum):
    """
    事件类型.
    """
    OPEN = "open"
    CLOSE = "close"
    RECORD = 'record'
    MONTHEND = 'month_end'
    WEEKEND = 'week_end'


@unique
class OrderType(Enum):
    """
    订单类型.
    """
    MARKET = "market"
    MARKETFAK = "market_fak"  # 完成不了就立刻取消
    RATIOMARKET = "ratio_market"  # 根据目标 交易金额的比例 确定目标成交量的市价单
    TARGETRATIOMARKET = "target_ratio_market"  # 根据 目标持仓金额的比例 确定目标成交量的市价单
    TARGETVOLUMMARKET = "target_volume_market"  # # 根据 目标持仓量 确定目标成交量的市价单

    VWAP = "vwap"
    VWAPFAK = "vwap_fak"  # 完成不了就立刻取消
    RATIOVWAP = "ratio_vwap"  # 根据目标 交易金额的比例 确定目标成交量的vwap单
    TARGETRATIOVWAP = "target_ratio_vwap"  # 根据 目标持仓金额的比例 确定目标成交量的VWAP单
    TARGETVOLUMEVWAP = "target_volume_vwap"  # 根据 目标持仓量 确定目标成交量的VWAP单

    LIMIT = "limit"
    STOP = "stop"


@unique
class Direction(Enum):
    """
    交易方向.
    """
    LONG = "long"
    SHORT = "short"
    NET = "net"  # 平仓
