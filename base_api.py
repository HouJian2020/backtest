# -*- coding: utf-8 -*-
"""

模块包含了回测框架的角色的接口定义，具体有：
1. Event: 事件数据类， 事件的抽象， 实例化例子：分红事件， 收盘事件， 横盘突破事件；
2. MarketBase: 市场基类， 市场的抽象，实例A股市场、ETF市场；
3. ObserverBase: 观察者基类， 基金会计的抽象， 其作用是记录一个基金的所有行为， 如交易记录、持仓记录、净资产等；
4. TraderBase: 交易员基类， 交易员的抽象， 其作用是执行基金经理的指令；
5. OrderStrategyBase: 订单策略基类， 订单策略的抽象，如市价单， 限价单；
    a. 订单策略的具体实现由市场和订单类型共同决定的， 比如A股限价单、ETF的市价单；
    b. 订单策略是交易员的技能，交易员通过调用订单策略来完成基金经理的指令。
6. PortfolioManagerBase: 基金经理基类， 基金经理的抽象， 基金经理根据各种数据下达交易指令。
7. TradeSessionBase: 回测过程控制；

回测的过程是基于事件的:
1. 事件类按照时间顺序执行。
2. 市场类根据行情推动事件，如开盘、收盘更新数据。
3. 基金经理根据某个事件（分红事件）， 结合其它数据做出交易决策并下达交易指令；
4. 交易员根据行情推动类事件执行交易操作。
5. 观察者（基金会计）根据事件进行结算、记录操作， 也会被动记录（发生了交易， 接收了订单）
6. 所有的事件便利完毕后， 回测也就完成了， 观察者会被返回， 里面封装了所有的回测信息。

Created on Sun Mar 19 11:28:30 2023

@author: HJ
"""

import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Type, Set, Union, Iterable
from .constant import SecuType, EventType, OrderType


# %%事件数据结构定义


@dataclass(frozen=True)
class Event:
    # 事件发生时间
    datetime: datetime.datetime
    event_seq: int = 5  # 事件类的排序， 执行排序靠前的

    # event_type为事件类型，不可以为 None
    # 如果是临时创建的事件，则可以用字符串表示， 行情触发类的事件必须是EventType类型以保证代码统一
    event_type: Union[EventType, str] = None
    """
    1. is_trigger是判断一个事件是不是“行情推动事件”的标志。
    2. 只有“行情推动事件”才会触发订单策略（OrderStrategyBase）的交易行为。
    3. 同一个市场的订单策略（OrderStrategyBase）必须有事件类型（event_type）的接口。
    3. 如果is_trigger = False, 代表非行情事件，如财报数据披露事件；这种事件下
        市场行情不用变更（MarketBase），订单策略（OrderStrategyBase）什么也不做。
    4. 同一个证券市场（secu_type），同一个时刻（datetime）只能有一个事件推动行情
        (is_trigger=True)。
    5. 如果两个事件，a，收到一个分钟K线；b，收盘， 同时发生且都触发is_trigger，
        那么市场（MarketBase）就会更新两次行情， 从而报错。
    """
    is_trigger: bool = False
    secu_type: SecuType = None  # 事件关联的市场， 可以为None，

    def __lt__(self, other):
        """比较两个事件谁先执行"""
        if self.datetime == other.datetime:
            return self.event_seq < other.event_seq
        return self.datetime < other.datetime

    # trade_date: datetime.date = None  # 所属交易日， 可以为None

    # def __hash__(self):
    #     return hash((self.datetime, self.event_seq,
    #                  self.is_trigger, self.secu_type.value,
    #                  self.event_type.value, self.trade_date))


def event_generator(t_series: datetime.datetime,
                    event_type: Union[EventType, str],
                    event_seq: int = 5,
                    is_trigger: bool = False,
                    secu_type: Union[SecuType, str] = None,):
    """
    生成事件类序列:
    1. 如果证券类型是字符串， 则匹配SecuType的成员；
    2. 如果事件类型是字符串， 则尝试匹配EventType的成员（匹配不到则用字符串）；
    """
    if isinstance(secu_type, str):
        secu_type = SecuType(secu_type)
    if isinstance(event_type, str):
        try:
            event_type = EventType(event_type)
        except:
            pass
    events = []
    for t in t_series:
        events.append(
            Event(datetime=t,
                  event_seq=event_seq,
                  event_type=event_type,
                  is_trigger=is_trigger,
                  secu_type=secu_type,
                  ))
    return (events)
# %%市场接口


class MarketBase(ABC):
    """
    1. 市场基类， 主要用于封装特定市场（如股票、可转债）的相关数据
        如（交易费率， 行情推动事件， 标的行情数据）等。
    2. 标的行情数据占用内存较大，因此数据被所有角色（PM，observer， trader）共享，
        建议只可读取，不可以修改。
    """

    secu_type: SecuType = None  # 证券类型
    # 如果不同标的手续费不同，可以在市场子类中进行修改，比如在data属性中维护费率信息
    buy_fee_ratio = 0.0005  # 买入手续费率
    sell_fee_ratio = 0.0015  # 卖出手续费率

    def __init__(self, start_date):
        """
        self.data, self.trigger_events是在回测之前必须要实现的属性，
        但是考虑到行情数据较大，未必在实例化的时候立刻实现。
        self.trade_date则建议实现，方便回测时调用。
        """
        self.data = None  # 市场行情数据
        self.trigger_events: Set[Event] = []  # 行情推动事件集合
        self.trade_date = None  # 市场当前的交易日， 数据格式与self.data中的日期格式一致

    def get_data(self, *args, **kwargs):
        """
        获取行情数据的函数.
        """
        pass

    def init_trigger_event(self):
        """
        初始化行情推动事件.
        """
        pass

    @abstractmethod
    def init(self):
        """
        市场的初始化，比如进行加载数据的操作等.
        """
        pass

    @abstractmethod
    def on_trigger(self, event: Event):
        """
        1. 获取结算价格
        2. 获取当前市场数据供订单策略的使用
        """
        pass

    @abstractmethod
    def settlement(self, position):
        """
        结算当前持仓.
        """
        pass


# %% 观察者接口


class RecordRenew(ABC):
    """
    观察者和交易员（trader）或订单策略之间的通讯函数。
    通过这些函数， 观察者可以被动（不直接依赖事件）记录信息。
    观察者外部修改观察者信息的仅能通过 RecordRenew中的方法。
    """

    @abstractmethod
    def on_trade(self, trade_data):
        """交易员完成交易后观察者需要更新的信息"""
        pass

    @abstractmethod
    def on_cancelled_orders(self, order_data):
        """交易员取消订单后观察者需要更新的信息"""
        pass

    @abstractmethod
    def add_active_orders(self, order_data):
        """
        观察者记录交易员正在完成（尚未完成）的订单，如未成交的限价单，未激活的止损单
        1. 基金经理提交新订单的记录.
        2. 交易员没有完成或部分完成的订单记录.
        """
        pass

    @abstractmethod
    def dispense_active_orders(self, secu_type: List[str]):
        """
        交易员尝试执行某个市场活跃的订单，观察者需要做的事情：
        1. 删除属于secu_type类型的所有活跃订单;
        2. 把删除的订单return给交易员去处理.
        """
        pass

    @abstractmethod
    def cal_frozen(self):
        """
        计算冻结的持仓和冻结的仓位
        提交订单，取消订单，交易完成后都需要进行一次清算。
        """
        pass


class ObserverBase(RecordRenew):
    """
    采用观察者设计模式， 代表基金会计？角色, 负责投资组合所有的信息的记录功能.
    负责记录的内容（属性）：
    1. 当前活跃订单  self.active_orders.
    2. 当前的基金净值 self.balance.
    3. 当前的组合持仓 self.position.
    4. 历史的组合持仓 self.position_records.
    5. 已经完成的交易记录 self.trade_records.
    6. 已经作废的订单记录 self.invalid_orders.
    7. 历史的基金净值数据 self.balance_records.

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

    def __init__(self, cash: float, market_list: List[MarketBase]):

        self.mkt_dict = {data.secu_type.value: data for data in market_list}
        self.init_data(cash=cash)

    def init_data(self, cash: float):
        # 初始化订单与交易编号
        self.order_id = self.trade_id = 0
        # 活跃单包含所有尚未完成的订单， 如未成交的限价单，未激活的止损单
        self.active_orders = self.init_active_orders()
        # 活跃订单对应的一些参数, 如交易比例等
        self.active_order_params = {}

        # 取消的订单记录
        self.invalid_orders = self.init_invalid_orders()

        # 交易记录
        self.trade_records = self.init_trade_records()

        # 当前持仓与历史持仓记录
        self.position = self.init_position()
        self.position_records_ = self.init_position_records()

        # 当前组合净值与历史组合净值记录
        self.balance = self.init_balance(cash)
        self.balance_records_ = self.init_balance_records()

    @abstractmethod
    def init_active_orders(self):
        pass

    @abstractmethod
    def init_position(self):
        pass

    @abstractmethod
    def init_balance(self, cash):
        pass

    @abstractmethod
    def init_position_records(self):
        pass

    @abstractmethod
    def init_balance_records(self):
        pass

    @abstractmethod
    def init_trade_records(self):
        pass

    @abstractmethod
    def init_invalid_orders(self):
        pass

    def on_trade(self, trade_data_raw):
        """
        发生交易后obsever需要更新以下内容：
        1. self.position。
        2. self.balance更新balance。
        3. self.trade_records。
        """
        trade_data = self.add_tradeid(trade_data_raw)
        self.update_position(trade_data)
        self.update_balance(trade_data)
        self.update_trade_records(trade_data)

    def add_active_orders(self, order_data):
        """更新活跃订单， 如果订单没有id则需要添加id"""
        self.update_active_orders(order_data)

    def on_cancelled_orders(self, order_data):
        """更新已经取消的订单的记录"""
        self.update_invalid_orders(order_data)

    @abstractmethod
    def add_tradeid(self, trade_data_raw):
        pass

    @abstractmethod
    def add_orderid(self, order_data_raw):
        pass

    @abstractmethod
    def update_active_orders_params(self):
        pass

    @abstractmethod
    def update_position(self, trade_data):
        pass

    @abstractmethod
    def update_balance(self, trade_data):
        pass

    @abstractmethod
    def update_trade_records(self, trade_data):
        pass

    @abstractmethod
    def update_active_orders(self, order_data):
        pass

    @abstractmethod
    def update_invalid_orders(self, order_data):
        # 删除要删除活动订单的数据
        pass

    @abstractmethod
    def settlement(self, secu_types: [SecuType]):
        """
        结算组合中属于某个市场的持仓.
        """
        pass

    @abstractmethod
    def update_position_records(self, record_time):
        """
        增加持仓记录.
        """
        pass

    @abstractmethod
    def update_balance_records(self, record_time):
        """
        增加组合净值记录.
        """
        pass


# %% 订单策略接口

class OrderStrategyBase(ABC):
    """
    1. 订单策略的两个维度的属性为市场（A股、tick级A股、CTA等）以及订单类型（市价订单， 限价订单等）；
    2. 订单策略的方法包含：
        a. send_orders, 发送订单的方法；
        b. cancel_orders, 取消订单的方法；
        c. on_trigger_event， 转发订单的交易结果给观察者；
        d. 基于事件类型的方法， 如on_open, on_tick等
            1. 这些方法输入值为订单， 输出订单的三个去向
            2. 输出的三个去向由on_trigger_event方法处理
    3. 事件类型包含 tick， bar， open， close等等，根据框架需求扩展
    4. 订单策略执行只会根据RenewRecord的接口修改observer的内容
    """
    secu_type: SecuType = None
    order_type: OrderType = None

    def __init__(self, observer: ObserverBase, market_obj: MarketBase):
        self.mkt_obj = market_obj
        self.observer = observer
        if self.secu_type != self.mkt_obj.secu_type:
            raise (ValueError('订单策略和市场行请数据的证券类型不匹配'))

    @abstractmethod
    def send_orders(self, data, order_seq: int = 5):
        """
        1. pm"发送"订单的函数, 新发送的订单是无法立刻成交的，状态为"unsubmit"。
        2. 两个行情trigger之间发送的订单必须等到下一个行情trigger才可以成交，
            否则会有用到未来数据的可能。
        3. 某只股票一天的开高低收为（8.1, 10, 8.1, 10），
            如果客户在中午提交了限价买单9.8，实际上无法确定是否可以成交的，
            很可能下单后股价就涨到10元导致无法成交。
        """
        pass

    def cancel_orders(self, order_data):
        """
        1. pm提交取消订单申请的函数, 取消的订单可能立刻生效，也可能无法立刻生效。
        2. 对于取消立刻生效的订单，直接调用observer相关方法。
        3. 取对于无法立刻取消的订单， 修改订单状态为"to_cancel", 然后等待"excute_cancel"方法处理。
        4. 因为目前市场订单取消会立刻生效，因此暂不实现"excute_cancel"。
        """
        self.observer.on_cancelled_orders(order_data)
        self.observer.cal_frozen()

    # def excute_cancel(self, order_data):
    #     """
    #     1. 对"to_cancel"状态的订单 尝试取消操作。
    #     2. excute_cancel在市场行情更新之后，在所有的交易操作之前。
    #     """
    #     active_orders, cancelled_orders = self.is_cancelable(order_data)
    #     self.observer.on_cancelled_orders(cancelled_orders)
    #     self.observer.add_active_orders(active_orders)

    def on_trigger_event(self, method, order_data):
        """
        处理行情推动事件发生时活跃订单的交易，订单有三种去向：
        1. active_orders 没有交易但是还要继续交易的， 通过observer.add_active_orders更新活跃订单记录；
        2. trade_data_raw 交易成功的订单， 通过observer.on_trade处理。
        3. cancelled_orders 没有交易且不需要继续交易的，通过observer.on_cancelled_orders处理。
        """
        active_orders, cancelled_orders, trade_data_raw = method(order_data)
        if active_orders is not None:
            self.observer.add_active_orders(active_orders)
        if cancelled_orders is not None:
            self.observer.on_cancelled_orders(cancelled_orders)
        if trade_data_raw is not None:
            self.observer.on_trade(trade_data_raw)

    def on_open(self, order_data):
        """开盘时刻活跃订单交易情况的判断。"""
        pass

    def on_close(self, order_data):
        """收盘时刻活跃订单交易情况的判断。"""
        pass

    def on_tick(self, order_data):
        """新的tick数据推送时活跃订单交易情况的判断。"""
        pass

    def on_bar(self, order_data):
        """新的bar数据推送时活跃订单交易情况的判断。"""
        pass


# %% 交易员接口


class TraderBase(ABC):

    def __init__(self, observer: ObserverBase):
        self.observer = observer
        # 市场类字典, key为secutype， value是MarketBase。
        self.mkt_dict = self.observer.mkt_dict
        # 保存订单策略的字典, key为secutype， value是order_strategy列表。
        self.order_strategys = {}

    def add_order_strategys(self, strategys: Iterable[OrderStrategyBase]):
        """为交易员添加订单策略， 储存为字典self.order_strategys"""
        for strategy in strategys:
            secu_type_str = strategy.secu_type.value
            if secu_type_str in self.mkt_dict:
                # raise (ValueError('找不到对应证券类型的行情数据'))
                if secu_type_str not in self.order_strategys:
                    self.order_strategys[secu_type_str] = []
                self.order_strategys[secu_type_str].append(
                    strategy(self.observer, self.mkt_dict[secu_type_str]))

    def get_order_strategy(self, secu_type: SecuType, order_type: OrderType):
        order_strategys = self.order_strategys[secu_type.value]
        for order_strategy in order_strategys:
            if order_strategy.order_type == order_type:
                return (order_strategy)
        else:
            raise (ValueError('找不到对应的订单策略'))

    @abstractmethod
    def on_cancel(self, order_data_id):
        """
        通过调用对应订单策略的cancel_orders方法取消order_data_id对应的订单。
        """
        pass

    @abstractmethod
    def on_trigger_events(self, events: List[Event]):
        """
        1. 行情推动事件发生后，交易员开始进行交易。
        2. 注意：执行顺序是以订单的顺序执行。
        """
        pass

    # @abstractmethod
    # def excute_cancel(self, event: Event):
    #     """
    #     1. 取消活跃订单中状态为"to_cancel"的订单。
    #     2. 如果取消订单是立刻生效的行为， 那么该方法则没有作用。
    #     3. 因为大部分的情况下订单取消是即刻生效的行为， 因此该方法暂不实现。
    #     """
    #     pass


# %% 基金经理基类


class PortfolioManagerBase(ABC):
    """
    基金经理基类中有三类方法：
    1. 基于事件的钩子函数， 如：
        1. after_close函数就是收盘事件要执行的函数，
        2. after_close_stock函数就是股票市场发生收盘事件时要执行的函数。
    2. init函数，是回测之前基金经理需要做的工作， 比如信号读取；
    3. 其它辅助函数，如self.position， self.get_mkt_obj， 帮助基金经理方便的工作。
    """

    def __init__(self, observer: ObserverBase, trader: TraderBase):
        self.observer = observer
        # 市场类字典, key为secutype， value是MarketBase。
        self.mkt_dict = self.observer.mkt_dict
        self.trader = trader

    def init(self):
        """
        回测前基金经理可以做的事情， 如：
        1. 基于行情数据的计算；
        2. 加载外部的数据，如信号数据。
        """
        pass

    def send_orders(self,
                    *args,
                    order_type: Union[OrderType, str],
                    secu_type:  Union[SecuType, str] = None,
                    order_seq: int = 5,
                    **kwargs):
        """
        调用订单策略的send_orders函数下单。
        1. 如果当前仅有一个市场，那么secu_type可以不用输入。
        2. order_seq代表同时交易的订单执行先后顺序。
        """

        order_type = self.get_order_type(order_type)
        secu_type = self.get_secu_type(secu_type)

        order_strategy = self.trader.get_order_strategy(secu_type, order_type)
        order_data = order_strategy.send_orders(
            *args, order_seq=order_seq, **kwargs)
        # 修改观察者信息
        self.observer.add_active_orders(order_data)  # 提交订单
        self.observer.cal_frozen()  # 计算冻结的资金和冻结的持仓

    def cancel_orders(self, order_data):
        """
        订单取消函数
        订单中要包含order_id
        """
        self.trader.on_cancel(order_data)

    @property
    def position(self):
        """获取观察者记录的持仓信息"""
        return (self.observer.position.copy())

    @property
    def active_orders(self):
        """获取观察者记录的活跃订单信息"""
        return (self.observer.active_orders.copy())

    @property
    def balance(self):
        return (self.observer.balance.copy())

    def get_mkt_obj(self, secu_type=None):
        """
        获取对应市场对象.
        """
        secu_type = self.get_secu_type(secu_type)
        return (self.mkt_dict[secu_type.value])

    def get_mkt_data(self, secu_type=None):
        """
        获取对应市场行情数据.
        """
        return (self.get_mkt_obj(secu_type).data)

    def get_secu_type(self, secu_type:  Union[SecuType, str]):
        """从字符串获取证券类型, 如果仅有一个市场，那么市场类型就唯一确定了"""
        if len(self.mkt_dict) == 1 and secu_type is None:
            secu_type = list(self.mkt_dict.keys())[0]
        if isinstance(secu_type, SecuType):
            return (secu_type)
        elif isinstance(secu_type, str):
            return (SecuType(secu_type))
        else:
            raise (TypeError('证券类型必须是Product枚举类或者是字符串'))

    def get_order_type(self, order_type: Union[OrderType, str]):
        """从字符串获取证券类型"""
        if isinstance(order_type, OrderType):
            return (order_type)
        elif isinstance(order_type, str):
            return (OrderType(order_type))
        else:
            raise (TypeError('订单策略类型必须是Product枚举类或者是字符串'))

# %% 回测框架基类


class TradeSessionBase(ABC):
    """
    回测框架基类。

    1. 主要的方法为self.run, 在该方法为回测程序的过程有两个重要步骤：
        a. 角色的初始化
            1. 实例化观察者、交易员、基金经理；
            2. 市场初始化、基金经理初始化、 交易员学习订单策略。
        b. 通过对事件的循环
            1. 根据事件的集合生成事件字典（以事件为key）；
            2. 根据事件字典的key，按顺序执行事件，并根据事件调用相应的处理函数。
    2. 其它的方法主要为回测过程需要的数据以及参数的管理：
        a. 市场对象的添加以及市场数据的加载
        b. 回测开始与结束时间
        c. 观察者、交易员、基金经理的更换
    """

    def __init__(self,
                 cash: float,  # 回测默认的初始资金
                 start_datetime: datetime.datetime,  # 回测开始时间
                 end_datetime: datetime.datetime,  # 回测结束时间
                 Observer: Type[ObserverBase],  # 观察者类
                 Trader: Type[TraderBase],  # 交易员类
                 PortfolioManager: Type[PortfolioManagerBase]  # 基金经理类
                 ):
        self.start_datetime = start_datetime  # 回测开始日期
        self.end_datetime = end_datetime  # 回测结束日期
        self.cash = cash
        self.Observer = Observer
        self.Trader = Trader
        self.PortfolioManager = PortfolioManager
        # 需要补充的数据
        self.market_list = []  # 市场对象列表
        self.events: Set[Event] = set()  # 事件集合
        self.order_strategys = set()  # 订单策略集合

    def get_event_dict(self):
        """事件集合转化为事件字典， 以事件的时间为键值"""
        event_dict = {}
        for event in self.events:
            dt = event.datetime
            if (dt >= self.start_datetime) and (dt <= self.end_datetime):
                if dt not in event_dict.keys():
                    event_dict[dt] = [event]
                else:
                    event_dict[dt].append(event)
        return (event_dict)

    def init_backtest(self):
        # 策略运行前，初始化观察者、交易员
        self.observer = self.Observer(self.cash, self.market_list)  # 观察者初始化
        self.trader = self.Trader(self.observer)  # 交易员初始化
        self.pm = self.PortfolioManager(self.observer, self.trader)  # 基金经理初始化
        # 基金经理在回测前的准备工作
        self.pm.init()
        # 市场的初始化
        for mkt in self.market_list:
            mkt.init(self.start_datetime)
        # 交易员学习交易策略
        self.trader.add_order_strategys(self.order_strategys)

    def run(self):
        """
        回测函数， 在运行回测之前必须：
        1. 通过add_market添加市场；
        2. 通过add_events添加事件；
        3. 完善add_order_strategys方法，为交易员添加订单策略；
        4. 对start_datetime， end_datetime，cash， Observer，Trader，
            PortfolioManager 等属性进行必要的修改。
        """
        # 初始化观察者，交易员与基金经理， 交易员添加订单策略， 市场初始化
        self.init_backtest()
        # 按时间把事件类代码转化为字典
        event_dict = self.get_event_dict()
        # 按时间顺序对事件进行操作
        for t in sorted(event_dict.keys()):
            events = event_dict[t]
            self.observer.time = t  # 观察者记录当前时间

            # 优先处理行情推送事件， 并保存在列表当中
            triggers = []
            for event in events:
                if event.is_trigger:  # 市场行情推进类事件
                    # 更新市场的最新状态
                    market = self.observer.mkt_dict[event.secu_type.value]
                    market.on_trigger(event)
                    triggers.append(event)

            # 行情推送事件触发后的工作流
            if len(triggers) > 0:
                # 交易员完成对应市场的交易工作
                self.trader.on_trigger_events(triggers)
                # 观察者对应市场的持仓标的进行结算操作
                self.observer.settlement(
                    [event.secu_type for event in triggers])
                # 删除已经无用的活跃订单参数
                self.observer.update_active_orders_params()
                # 观察者更新冻结的资金和仓位
                self.observer.cal_frozen()

            # 按照事件的顺序基金经理和观察者开始工作
            for event in sorted(events):  # 按事件的顺序执行命令
                if event.event_type == EventType.RECORD:  # 记录类事件
                    # 记录历史持仓数据
                    self.observer.update_position_records(event.datetime)
                    # 记录历史净值数据
                    self.observer.update_balance_records(event.datetime)

                # pm基金经理开始计算信号，假设事件类型是'open',
                # pm会先执行'after_open'方法，如果没有对应的方法则什么都不做
                if isinstance(event.event_type, EventType):
                    method_name = 'after_' + event.event_type.value
                elif isinstance(event.event_type, str):
                    method_name = 'after_' + event.event_type
                method = getattr(self.pm, method_name, None)
                if method is not None:
                    method()
                # 如果事件类型有对应的市场，如股票市场（stock),
                # pm会执行'after_open_stock'方法, 如果没有对应的方法则什么都不做
                if event.secu_type is not None:
                    method_name = method_name + '_' + event.secu_type.value
                    method = getattr(self.pm, method_name, None)
                    if method is not None:
                        method()
        return self.observer

    # 市场管理： 添加市场
    def add_market(self, market: MarketBase):
        # 添加市场前线删除已经存在的同类型的市场
        for mkt in self.market_list:
            if mkt.secu_type == market.secu_type:
                self.del_market(mkt)
                break
        self.market_list.append(market)
        self.add_events(market.trigger_events)

    # 市场管理： 删除市场
    def del_market(self, market: MarketBase):
        self.market_list.remove(market)
        self.events = {event for event in self.events
                       if event not in market.trigger_events}

    # 市场管理： 重置市场
    def del_all_markets(self):
        self.market_list = []
        self.events = {event for event in self.events if not event.is_trigger}

    # 事件管理： 增加事件
    def add_events(self, events: List[Event]):
        """新增事件"""
        self.events.update(events)

    # 事件管理： 删除事件
    def del_events(self, event_type: Union[EventType, str], is_trigger=False):
        """根据事件类型以及是否行情推动事件判断是否删除"""
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except:
                pass
        self.events = {event for event in self.events
                       if not ((event.event_type == event_type)  # 删除事件类型相同的
                               and (event.is_trigger == is_trigger))}  # 删除is_trigger相同的

    # 事件管理：重置事件
    def reset_events(self):
        """重置事件"""
        self.events: Set[Event] = set()
        for mkt in self.market_list:
            self.events.update(mkt.trigger_events)
