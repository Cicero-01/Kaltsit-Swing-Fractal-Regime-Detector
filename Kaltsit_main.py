import pandas as pd
import numpy as np

def swing_regime_detector(df_raw, k=2):

    df = df_raw.copy()

    # Step1: 几何结构点识别 (定义理论上的Swing点)
    # Define Swing
    is_swing_high = pd.Series(True, index=df.index)
    is_swing_low = pd.Series(True, index=df.index)

    for i in range(1, k + 1):
        is_swing_high &= (df['High'] > df['High'].shift(i)) & (df['High'] > df['High'].shift(-i))
        is_swing_low &= (df['Low'] < df['Low'].shift(i)) & (df['Low'] < df['Low'].shift(-i))

    df['Raw_Swing_High'] = np.where(is_swing_high, df['High'], np.nan)
    df['Raw_Swing_Low'] = np.where(is_swing_low, df['Low'], np.nan)

    # Step2: 消除未来函数 (Confirmation Shift)
    # 第t天的Swing点，要在第t+k天才能被右侧K线确认
    df['Confirmed_SH_Price'] = df['Raw_Swing_High'].shift(k)
    df['Confirmed_SL_Price'] = df['Raw_Swing_Low'].shift(k)

    # Step3: 提取并更新最新的两个结构高点 (SH1, SH2) 和低点 (SL1, SL2)
    # 前向填充最新确认的结构位
    df['Latest_SH'] = df['Confirmed_SH_Price'].ffill()
    df['Latest_SL'] = df['Confirmed_SL_Price'].ffill()

    # 提取上上个结构点 (用于对比HH/LH/HL/LL
    sh_changed = df['Latest_SH'] != df['Latest_SH'].shift(1)
    sl_changed = df['Latest_SL'] != df['Latest_SL'].shift(1)

    df['Prev_SH'] = np.where(sh_changed, df['Latest_SH'].shift(1), np.nan)
    df['Prev_SH'] = df['Prev_SH'].ffill()

    df['Prev_SL'] = np.where(sl_changed, df['Latest_SL'].shift(1), np.nan)
    df['Prev_SL'] = df['Prev_SL'].ffill()

    # Step4: 结构状态分类 Dow Theory Regime Classification
    # HH: Higher High (高点抬高), LH: Lower High (高点降低)
    # HL: Higher Low (低点抬高), LL: Lower Low (低点降低)
    is_hh = df['Latest_SH'] > df['Prev_SH']
    is_lh = df['Latest_SH'] < df['Prev_SH']
    is_hl = df['Latest_SL'] > df['Prev_SL']
    is_ll = df['Latest_SL'] < df['Prev_SL']

    # 1: Trend Up   (HH & HL)
    # 2: Trend Down (LH & LL)
    # 0: Range/Chaos (高低点收敛、扩张或交叉，如HH+LL或LH+HL)
    conditions = [
        (is_hh & is_hl),  # 上升趋势 Trend Up
        (is_lh & is_ll)   # 下降趋势 Trend Down
    ]
    choices = [1, 2]

    # 默认状态为0 (震荡/混沌区间)
    # Default: 0 (Range/Chaos)
    df['Dow_Regime'] = np.select(conditions, choices, default=0)

    # 计算该状态已连续维持的时间
    # Duration
    regime_changed = df['Dow_Regime'] != df['Dow_Regime'].shift(1)
    df['Dow_Session_Group'] = regime_changed.cumsum()
    df['Dow_Duration'] = df.groupby('Dow_Session_Group').cumcount() + 1

    # 再做一次shift(1)，确保交易日只使用上条K线确认的状态，对应实盘中只能最晚看到到上一条K线的数据，消除未来函数
    # By using another shift(1) to avoid Look-Ahead Bias
    df['Swing_State'] = df['Dow_Regime'].shift(1)
    df['Swing_Duration'] = df['Dow_Duration'].shift(1)
    
    df['State_3']=df['Swing_State'] # 可视化接口 For Visualization
    
    return df


if __name__ == "__main__":
    df_raw = pd.read_csv("backtesting_daily.csv")
    df = swing_regime_detector(df_raw,k=2)
    df.to_csv("example_kline.csv")


