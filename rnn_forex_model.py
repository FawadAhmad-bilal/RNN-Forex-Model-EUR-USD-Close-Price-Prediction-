
import yfinance as yf

data = yf.download("EURUSD=X", start="2010-01-01")

print(data.head())

data.head()

data.shape

data.info()

data.describe()

from sklearn.preprocessing import StandardScaler,MinMaxScaler

data.columns

data.columns = data.columns.droplevel(1)

data.duplicated().sum()
df=data.drop(columns=['Volume'])

X=df.drop(columns=['Close'])
y=df['Close']

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
split = int(len(df)*0.8)
X_train_raw = X.iloc[:split]
X_test_raw = X.iloc[split:]

y_train_raw = y.iloc[:split]
y_test_raw = y.iloc[split:]

# scaler=StandardScaler()
# x_train_scaled=scaler.fit_transform(x_train)
# x_test_scaled=scaler.transform(x_test)
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

def create_sequences(X_data, y_data, time_step):
    X = []
    y = []

    for i in range(len(X_data)-time_step):
        X.append(X_data[i:i+time_step])
        y.append(y_data.iloc[i+time_step])

    return np.array(X), np.array(y)

X_train, Y_train = create_sequences(
    X_train_scaled,
    y_train_raw.reset_index(drop=True),
    20
)
X_test, Y_test = create_sequences(
    X_test_scaled,
    y_test_raw.reset_index(drop=True),
    20
)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

model=Sequential()
model.add(SimpleRNN(16,return_sequences=True,input_shape=(20,3)))

model.add(SimpleRNN(16,return_sequences=False))

model.add(Dense(40,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mse',optimizer='adam',metrics=['r2_score'])

history = model.fit(
    X_train,
    Y_train,
    epochs=25,
    validation_data=(X_test,Y_test)
)

plt.plot(history.history['r2_score'])
plt.plot(history.history['val_r2_score'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

pred=model.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

r2_score(pred,Y_test)

mean_absolute_error(pred,Y_test)

mean_squared_error(pred,Y_test)

