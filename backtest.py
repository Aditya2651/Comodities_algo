## LOADING LIBRARIES
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import ta 
import datetime
pd.options.mode.chained_assignment = None  

## LOADING GLOBAL 
path = "D:/SWING/COMMODITIES_DATA/ohlc_5min/"
cmpDat =['SILVERM25FEB','SILVER25MAR','NATGASMINI24DEC','CRUDEOIL24DEC','NATURALGAS24DEC','SILVERMIC25FEB','CRUDEOILM24DEC']
prod= ['SILVERM','SILVER','NATGASMINI','CRUDEOIL','NATURALGAS','SILVERMIC','CRUDEOILM']
mth_code =['G25','H25','Z24','Z24','Z24','G25','Z24']

#### FUNCTIONS FOR BACKTESTING ####

def backtest_trading_strategy(df):
    """
    Main function to backtest the trading strategy.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing OHLCV data and necessary columns.
    
    Returns:
        pd.DataFrame: DataFrame with positions and trade details.
    """
    ## ADDING TECHNICAL INDICATORS AND ZONES
    def add_technical_indicators_and_zones(df):
        """
        Adds technical indicators (EMA20, EMA50, ATR, BarSize) and creates Buy and Sell zones.

        Args:
            df: A Pandas DataFrame containing 'Close', 'High', 'Low', and 'Timestamp' columns.

        Returns:
            A Pandas DataFrame with added technical indicators and Buy/Sell zones.
        """
        # Add EMA20 and EMA50
        df['EMA20'] = ta.trend.ema_indicator(df['Close'], 20)
        df['EMA50'] = ta.trend.ema_indicator(df['Close'], 50)
        
        # Add ATR
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Add BarSize (50-period SMA of the range)
        df['BSize'] = ta.trend.sma_indicator(df['High'] - df['Low'], 50)
        
        # Add Date (from Timestamp)
        df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        # Create BuyZone and SellZone
        df['BuyZone'] = df['EMA20'] > df['EMA50']
        df['SellZone'] = df['EMA20'] < df['EMA50']
        
        return df

    ## ADDING SWING HIGH AND LOWS
    def add_swing_high_low(df, atrMlt=0.25):
        """
        Adds swing high and swing low columns to the DataFrame based on certain conditions.

        Args:
            df: A Pandas DataFrame with columns 'High', 'Low', 'Close', 'Open', 'ATR', 'BuyZone', 'SellZone'.
            atrMlt: Multiplier for ATR when determining the swing high and low.

        Returns:
            A Pandas DataFrame with 'swing_high' and 'swing_low' columns added.
        """
        # Unpacking columns for readability
        highs, lows, closes, opens = df['High'], df['Low'], df['Close'], df['Open']
        atr, bzn, szn = df['ATR'], df['BuyZone'], df['SellZone']

        # Shifting columns for window-based comparisons
        shifts = {
            'close': [closes.shift(i) for i in range(1, 3)] + [closes.shift(i) for i in range(-1, -3, -1)],
            'open': [opens.shift(i) for i in range(1, 3)] + [opens.shift(i) for i in range(-1, -3, -1)],
        }

        # High and low condition checks
        high_condition = (highs > pd.concat(shifts['close'] + shifts['open'], axis=1).max(axis=1)) & bzn
        low_condition = (lows < pd.concat(shifts['close'] + shifts['open'], axis=1).min(axis=1)) & szn

        # Calculate swing highs and swing lows
        df['swing_high'] = highs.where(high_condition) + atr.shift(1) * atrMlt
        df['swing_low'] = lows.where(low_condition) - atr.shift(1) * atrMlt

        return df

    ## IDENTIFYING BODY TOUCHES AS A COLUMN
    def add_body_touch(df):
        """
        Adds a 'Bdy' column to indicate if the price body touches key EMAs.

        Args:
            df (pd.DataFrame): A DataFrame containing 'Open', 'Close', 'EMA20', and 'EMA50' columns.

        Returns:
            pd.DataFrame: The modified DataFrame with the 'Bdy' column added.
        """
        # Determine the minimum and maximum of 'Open' and 'Close' for each row
        price_min = df[['Open', 'Close']].min(axis=1)
        price_max = df[['Open', 'Close']].max(axis=1)

        # Evaluate conditions for body touching the EMAs
        conditions = (
            (df['EMA20'].between(price_min, price_max)) |  # EMA20 between body range
            (df['EMA50'].between(price_min, price_max)) |  # EMA50 between body range
            (price_min.between(df['EMA20'], df['EMA50'])) |  # Body range between EMAs
            (price_max.between(df['EMA20'], df['EMA50']))
        )

        # Assign the result to the 'Bdy' column
        df['Bdy'] = conditions

        return df

    def consecutive_vec(data, stepsize=0):
        """
        Returns the indices where the difference between consecutive elements 
        is not equal to the specified stepsize.

        Args:
            data (array-like): The input data (e.g., list or NumPy array).
            stepsize (int or float, optional): The value to compare the differences with (default is 0).

        Returns:
            list: List of indices where the consecutive difference is not equal to stepsize,
                or [0] if no such indices are found.
        """
        # Ensure that the data is a NumPy array for efficient computation
        data = np.asarray(data)

        # Compute the indices where the difference is not equal to stepsize
        cntVec = np.flatnonzero(np.diff(data) != stepsize) + 1

        # Return the list of indices, or [0] if no such indices are found
        return cntVec.tolist() if cntVec.size > 0 else [0]

    def isBetween(lower_bound, upper_bound, value):
        """
        Check if the value is between the lower and upper bounds, inclusive.

        Args:
            lower_bound (int or float): The lower bound for the comparison.
            upper_bound (int or float): The upper bound for the comparison.
            value (int or float): The value to check if it's between the bounds.

        Returns:
            bool: True if the value is between the bounds (inclusive), False otherwise.
        """
        return lower_bound <= value <= upper_bound

    def appendDict(dict1,dict2):
        # Use dictionary comprehension for clarity and efficiency
        return {
            key: dict1.get(key, []) + dict2.get(key, [])
            for key in set(dict1) | set(dict2)  # Union of keys from both dictionaries
        }

    def retPosDct():
        buyPos = {
            "Date":[None],
            "Time":[None],
            "Entry":[None],
            "SL1":[None],
            "SL2":[None],
            "SL3":[None],
            "TP1":[None],
            "TP2":[None],
            "TP3":[None],
            "TP4":[None],
            "ExitSL":[None],
            "ExitTP1":[None],
            "ExitTP2":[None],
            "ExitTP3":[None],
            "ExitTP4":[None]
        }
        return buyPos

    ## EXIT FUNCTION
    def exitfunc(buyPos, tmpDat, i):
        """
        Updates the buyPos dictionary with exit stop-loss (SL) or take-profit (TP) levels based on conditions.

        Args:
            buyPos (dict): Dictionary containing trade position details.
            tmpDat (DataFrame): DataFrame containing price data with 'Low' and 'High' columns.
            i (int): Index of the current row in tmpDat.

        Returns:
            dict: Updated buyPos dictionary.
        """
        lowPrc = tmpDat['Low'][i]
        higPrc = tmpDat['High'][i]

        # Check for Stop Loss (SL) levels
        for sl_key in ['SL1', 'SL2', 'SL3']:
            if buyPos[sl_key][0] is not None and isBetween(lowPrc, higPrc, buyPos[sl_key][0]):
                buyPos['ExitSL'][0] = buyPos[sl_key][0]
                break

        # Check for Take Profit (TP) levels only if no SL has been hit
        if buyPos['ExitSL'][0] is None:
            for tp_key, next_sl_key, next_sl_value in [
                ('TP1', None, None),
                ('TP2', 'SL2', 'Entry'),
                ('TP3', 'SL3', 'TP1'),
                ('TP4', None, None),
            ]:
                if isBetween(lowPrc, higPrc, buyPos[tp_key][0]):
                    buyPos[f'Exit{tp_key}'][0] = buyPos[tp_key][0]
                    if next_sl_key:
                        buyPos[next_sl_key][0] = buyPos[next_sl_value][0]
                    break

        return buyPos

    ## ENTRY FUNCTION
    def entryfunc(buyPos, tmpDat, i, zone='BuyZone', swing='swing_high', buyorsell=True):
        """
        Determines entry conditions and updates the buy position.

        Args:
            buyPos (dict): Dictionary to store trade entry details.
            tmpDat (DataFrame): DataFrame containing price data and indicators.
            i (int): Current row index in tmpDat.
            zone (str): Column name for zone conditions.
            swing (str): Column name for swing high/low levels.
            buyorsell (bool): True for buy conditions, False for sell.

        Returns:
            dict: Updated buyPos dictionary with entry details.
        """
        # Get the last valid swing high/low
        lstSwgHgh = tmpDat[swing][:(i-2)].last_valid_index()
        if not bool(lstSwgHgh):  
            return buyPos  # Exit early if no valid swing point

        # Get consecutive zones
        iscnt_bzn = consecutive_vec(tmpDat[zone][:(i-1)])[-1]
        iscnt_bdy = consecutive_vec(tmpDat['Bdy'][:(i-1)])[-1]

        # Common variables for both buy and sell
        swing_level = tmpDat[swing][lstSwgHgh]
        atr_adjustment = tmpDat['ATR'][lstSwgHgh] * 0.25
        ema20_level = tmpDat['EMA20'][lstSwgHgh]
        price_range = tmpDat['Low'][lstSwgHgh:(i)], tmpDat['High'][lstSwgHgh:(i)]

        # Define buy and sell logic
        if buyorsell:  # Buy conditions
            sl_value = min(price_range[0]) - atr_adjustment
            sl_condition = tmpDat['Low'][lstSwgHgh] > ema20_level
        else:  # Sell conditions
            sl_value = max(price_range[1]) + atr_adjustment
            sl_condition = tmpDat['High'][lstSwgHgh] < ema20_level

        if (
            sl_condition and
            (lstSwgHgh >= iscnt_bzn and lstSwgHgh >= iscnt_bdy) and
            all(tmpDat[zone][lstSwgHgh:(i)]) and
            not any(tmpDat['Bdy'][lstSwgHgh:(i)]) and
            isBetween(tmpDat['Low'][i], tmpDat['High'][i], swing_level)
        ):
            # Populate buy position details
            buyPos['Date'] = [tmpDat['Date'][i]]
            buyPos['Time'] = [tmpDat['Timestamp'][i]]
            buyPos['Entry'] = [swing_level]
            buyPos['SL1'] = [sl_value]
            risk_reward = buyPos['Entry'][0] - buyPos['SL1'][0]
            for tp in range(1, 5):
                buyPos[f'TP{tp}'] = [buyPos['Entry'][0] + (risk_reward * tp)]

        return buyPos

    ## MAIN BACKTEST LOOP
    df = add_technical_indicators_and_zones(df)
    df = add_swing_high_low(df)
    df = add_body_touch(df)
    unqDat = df['Date'].unique()

    final_df = retPosDct()  # Initialize once outside the loop
    ## Order for final_df
    desired_order = [
        "Date", "Time", "Entry", "SL1","TP1","ExitTP1", "SL2", "TP2","ExitTP2",
        "SL3", "TP3","ExitTP3", "TP4", "ExitTP4","ExitSL"
    ]

    

    for k in unqDat:
        # Subset data for the current date
        tmpDat = df[df['Date'] == k].reset_index(drop=True)  # reset the index for convenience
        tmpDat['swing_high'].iloc[:4] = None  # set initial values to None
        tmpDat['swing_low'].iloc[:4] = None

        buyPos = retPosDct()  # Initialize positions
        sellPos = retPosDct()

        tmpDat['Position'] = None
        tmpDat['PosPrc'] = None

        # Entry conditions loop
        for i in range(7, len(tmpDat) - 5):
            # Handle Buy Zone
            if buyPos['Entry'][0] is not None:
                buyPos = exitfunc(buyPos, tmpDat, i)
            elif tmpDat['BuyZone'][i-1]:
                buyPos = entryfunc(buyPos, tmpDat, i, zone="BuyZone", swing="swing_high", buyorsell=True)

            # Handle Sell Zone
            if sellPos['Entry'][0] is not None:
                sellPos = exitfunc(sellPos, tmpDat, i)
            elif tmpDat['SellZone'][i-1]:
                sellPos = entryfunc(sellPos, tmpDat, i, zone="SellZone", swing="swing_low", buyorsell=False)

        # Append results to the final result dictionary
        final_df = appendDict(final_df, buyPos)
        final_df = appendDict(final_df, sellPos)

    # Convert the result dictionary to DataFrame and clean it
    final_df = pd.DataFrame(final_df)
    final_df = final_df.dropna(subset=["Time"])  # Remove rows with no time entry
    #final_df = final_df.drop(columns=final_df.columns[4:10])  # Drop unnecessary columns
    # Reorder the columns in the DataFrame
    final_df = final_df[desired_order]
    return final_df


## CALLING BACKTEST FUNCTION 
data =  pd.read_parquet(path+'SILVERMG25.parquet')
backtest_final_df = backtest_trading_strategy(data)
