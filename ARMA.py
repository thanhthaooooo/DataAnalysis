#%% - Nap thu vien
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller,kpss
#from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA


import warnings
warnings.filterwarnings("ignore")

#%% DPI : mật độ điểm ảnh
plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 14

#%%
df = pd.read_csv("./Data/ACG.csv", index_col="Date", parse_dates=True)
df.info

#%% - giá đóng của của cổ phiếu -> nhận định ban đầu có xu hướng tăng
plt.plot(df['Close'])
plt.xlabel("Date")
plt.ylabel("Close price")
plt.show()

#%% - Chia dữ liệu train test để phân tích dữ liệu trend
# có đặc điểm gì rồi mới kiểm tra lại bằng dữ liệu test
df_close = np.log(df["Close"])
#log cho dữ liệu nhỏ để tiện phân tích và giảm một cách đồng nhất
train_data, test_data = df_close[:int(len(df_close)*0.9)], df_close[int(len(df_close)*0.9):]
plt.plot(train_data,'blue',label='Train Data')
plt.plot(test_data,'red',label='Test Data')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

#%% - Phân rã dữ liệu xem xét tính mùa vụ ,
# Cần làm bước trung gian vẽ biểu đồ ss giá trung bình và độ lệch chuẩn
# Biểu đồ so sánh giá đóng cửa với giá trị trung bình và độ lệch chuẩn của 12 kỳ trước đó
rolmean = train_data.rolling(12).mean()
rolstd = train_data.rolling(12).std()
#std sự biến thiên dao động cuar dữ liênuj có bất thường hay không
plt.plot(train_data,'blue', label="Original")
plt.plot(rolmean,'red',label="Rolling Mean")
plt.plot(rolstd,'Orange',label="Rolling STD")
plt.legend()
plt.show()

# Std có một chỗ 2008 - 2010 có vài chỗ dao động lớn

#%%
# Biểu đồ phân rã chuỗi thời gian (decompose) để xem tính xu hướng và mùa vụ
decompose_results = seasonal_decompose(train_data,model="multiplication", period=30)
decompose_results.plot()
plt.show()


#%% - Kiểm định tính dứng của dữ liệu dùng ADF hoặc KPSS:
def adf_test(data):
    indices = ["ADF: Test statistic","p value","# of Lags","# of Observations"]
    test = adfuller(data, autolag="AIC")
    results = pd.Series(test[:4], index=indices)
    for key, value in test[4].items():
        results[f"Critical value ({key})"] = value
    if results[1] <=0.05:
        print('Reject the null hyposthesis (H0), \n the data is stationary')
    else:
        print('Fail to reject the null hyposthesis (H0), \n the data is non-stationary')
    return results

def kpss_test(data):
    indices = ["KPSS: Test statistic", "p value", "# of Lags"]
    test = kpss(data)
    results = pd.Series(test[:3], index=indices)
    for key, value in test[3].items():
        results[f"Critical value ({key})"] = value
    if results[1] <= 0.05:
        print('Reject the null hyposthesis (H0), \n the data is stationary')
    else:
        print('Fail to reject the null hyposthesis (H0), \n the data is non-stationary')
    return results


# Kiểm định ADF
#test = adfuller(train_data, autolag="AIC")

print(adf_test(train_data))
# có thể bác bỏ khác không lấy trị tuyệ đối Crit mà lớn hớn stati thì không thir bác bỏ chuỗi dữ liệu không dừng


#Kieemr ddinhj KPSS
#test = kpss(train_data)
print("------------"*5)
print(kpss_test(train_data))

#%% - Kiểm định tự tương quan (Auto Correlation) khi nó không dừng
# Kỳ trước nó có liên quan đến kỳ sau hay không lag - độ trễ
pd.plotting.lag_plot(train_data)
plt.show()

#%%
plot_pacf(train_data)
plt.show()

#%% dường sin cos có sự tương quan các kỳ - và giảm dần cũng vậy
plot_acf(train_data)
plt.show()

#%% Chuyển đổi dữ liệu sang chuỗi dừng -> sai phân bậc 1 rồi kiểm tra dừng nếu không thif sai phân bậc 2
diff =train_data.diff(1).dropna()
# Biểu đồ thể hiện dữ liệu ban đầu và sau khi lấy sai phân
fig, ax = plt.subplots(2, sharex="all")
train_data.plot(ax=ax[0],title="Gía đóng cửa")
diff.plot(ax=ax[1], title="Gía đóng cửa")
plt.show()

#xet nó có phải nhiễu trắng hay không

#%% - Kiểm tra lại tính dừng dữ liệu sau khi lấy sai phân
print(adf_test(diff))
print("-----------"*5 )
print(kpss_test(diff))
#%%
plot_pacf(diff)
plt.show() # Xác định tham số p cho ARIMA p=1 hoặc 2

#%%
plot_acf(diff)
plt.show() #q = 1,2

#%% - Xác định tham số p,d,q cho mô hình ARIMA
stepwise_fit = auto_arima(train_data, trace=True, suppress_warnings=True)
print(stepwise_fit.summary()) #Chon AIC thấp nhất
stepwise_fit.plot_diagnostics(figsize=(15,8))
plt.show()

#%% - Tao mo hinh
#model = ARIMA(train_data, order=(1,1,2))
#fitted = model.fit()

model = ARIMA(train_data, order=(1,1,2), trend="t")
fitted = model.fit()
print(fitted.summary())

#%% - Dự báo ̣forecast - conf hiện thi cận trên dưới đọ tin cập
#fc, se, conf  = fitted.forecase(len(test_data),alpha=0.05)
#fc_series = pd.Series(fc, index=test_data.index)
#lower_series = pd.Series(conf[:,0], index=test_data.index)
#upper_series = pd.Series(conf[:,1], index=test_data.index)

fc = fitted.get_forecast(len(test_data))
fc_values = fc.predicted_mean
fc_values.index = test_data.index
conf = fc.conf_int(alpha=0.05)  # 95% conf

lower_series = conf["lower Close"]
lower_series.index = test_data.index
upper_series = conf["upper Close"]
upper_series.index = test_data.index

#%% Plot actual and predicted values
plt.figure(figsize=(16, 10), dpi=150)
plt.plot(train_data, label="Training data")
plt.plot(test_data, color="orange", label="Actual stock price")
plt.plot(fc_values, color="red", label="Predicted stock price")
plt.fill_between(lower_series.index, lower_series, upper_series, color="b", alpha=.10)
plt.title("Stock price prediction")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend(loc="upper left", fontsize=16)
plt.show()


#%% Đánh giá hiệu suất mô hình
mse=mean_squared_error(test_data,fc_values)
print('Test MSE: %.3f' % mse)
rmse=np.sqrt(mse)
print('Test RMSE: {:.3f}'.format(rmse))

#%%
baseline_prediction = np.full_like(test_data,train_data.mean())
baseline_rmse = np.sqrt(mean_squared_error(test_data,baseline_prediction))

#%% Visualize RMSE comparison
plt.figure(figsize=(16,10))
plt.bar(['ARIMA Model','Baseline'],[rmse, baseline_rmse], color=['blue','green'])
plt.title('Root Mean Squared Error (RMSE) Comparison')
plt.ylabel('RMSE')
plt.show()

print('ARIMA Model RMSE: {:.2f}'.format(rmse))
print('Baseline RMSE : {:.2f}'.format(baseline_rmse))