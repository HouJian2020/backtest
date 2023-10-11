import pandas as pd
from backtest.base_api import event_generator
from backtest.dataframe import PMPandas, TradeSession

class PMSimple(PMPandas):
    """基金经理（策略）的具体实现"""

    def init(self):

        # 获取对象
        obs = self.observer  # 获取观察者对象
        print(f'init方法回测前被调用：')
        mkt = self.get_mkt_obj()  # 获取市场对象

        # 修改市场对象的属性
        mkt.sell_fee_ratio = 0.002

        # 获取数据，获取外部数据
        print('数据加载：')
        self.signal = SIGNAL

        # 获取数据： 获取市场数据
        data = self.get_mkt_data()

        # 数据预处理： 粗略计算距离退市的时间
        data['ndelist'] = data.groupby('code', group_keys=False)['tradeDate'].apply(
            lambda x: pd.Series(range(len(x), 0, -1), index=x.index))

        # 初始化一个计数的变量
        self.open_count = 0
        print('-' * 10 + '初始化工作完成' + '-' * 10)

    def after_open(self):
        if self.open_count % 5 == 0:
            print(f'{self.observer.time}开盘了，且self.open_count是5的整数倍，执行买入操作')
            data = self.get_mkt_frame()
            data = data.sort_values('settlePrice').head(5).query("ndelist > 30")
            data['ratio_relative'] = data['settlePrice']  # 以价格为成交比例
            data['code'] = data.index

            # 与交易员的交互，下单
            self.send_orders(
                data[['code', 'ratio_relative']],
                ratio_to_netasset=0.5,
                order_type='ratio_market',
                order_seq=1)  # 交易优先级靠后， 保证有足够的资金释放
        self.open_count += 1

    def after_close(self):
        self.cancel_orders(self.active_orders)
        if (self.open_count + 1) % 5 == 0:
            print(f'{self.observer.time}收盘了，触发卖出条件，执行平仓操作')
            # 与交易员的交互 平仓
            self.sell_position(self.position)

    # 通过定义事件与环境交互
    def after_any(self):
        print(f'{self.observer.time}after_any被调用')

    # 通过定义事件与环境交互
    def after_any_cb_daily(self):
        print(f'{self.observer.time}after_any_cb_daily被调用')
        print('---' * 15)
        print("获取数据汇总：")
        print(f"当前时间:{self.observer.time}")
        print(f"当前持仓:{self.position}")
        print(f"当前权益:{self.balance}")
        print(f"当前活跃订单:{self.active_orders}")
        print(f"市场行情:{self.get_mkt_data().tail()}")
        print(f"市场行情切片:{self.get_mkt_frame().tail()}")
        print(f"外部输入的信号:{self.signal}")
        print('---' * 15)

    def __str__(self):
        return ("PM例")
t_series = pd.date_range('2021-01-01', '2021-03-01', freq='w')
events = event_generator(t_series.to_pydatetime(), event_type='any')
ts = TradeSession()
ts.reset_markets('cb_daily', '2012-01-01')
ts.start_datetime = '2018-01-01'# 更改回测开始时间
ts.end_datetime = '2021-12-31' # 更改回测结束时间
ts.cash = 1e10 # 更改初始资金大小
ts.add_events(events)
ts.PortfolioManager = PMSimple
SIGNAL = '某外部数据'
obs = ts.run()