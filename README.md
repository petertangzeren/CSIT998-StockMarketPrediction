# CSIT998-StockMarketPrediction
stock_predict(model, df_close, days) is the function to use 'model' to predict future market 'close' price in 'df_close', then get the change rate in a time 'days'.
'df_close' -- All the close data for all opening days.

确保model已经被从路径中提取出来了，然后再放到stock_predict里面去。关于df_close的获取步骤，Stock Price.ipynb中第4-5个cells里有。

Range of 'days': 0 <= 'days' <= 60. 可以使用范围中任意天数

