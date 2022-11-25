import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers


def create_df():
    while True:
        file_name = input('File name with stock data: ')
        try:
            df = pd.read_csv(file_name)
            break
        except:
            continue
    return df

def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])

    return datetime.datetime(year=year, month=month, day=day)

df = create_df()
df = df.drop(['Adj Close'], axis=1)

df['Date'] = df['Date'].apply(str_to_datetime)
df.index = df.pop('Date')
#print(df.head())

def stripped_dataframe(start_date, end_date, df, n=3):
    stripped_df = df[['Close']].copy()
    stripped_df['Date'] = df.index
    for _ in range(n):
        stripped_df = stripped_df.drop([stripped_df.index[0]])
    stripped_df.reset_index(drop=True, inplace=True)
    col_names = ['Date', 'Close']
    for i in range(n):
        col_names.insert(len(col_names)-1, f"Close-{n-i}")
        i += 1
        column = list(df['Close'])
        for _ in range(n-i):
            column.pop(0)
        for _ in range(i):
            column.pop()
        stripped_df[f"Close-{i}"] = column
    col_names
    stripped_df = stripped_df.loc[:, col_names]

    days_between = []
    delta = end_date - start_date
    for i in range(delta.days + 1):
        day = start_date + datetime.timedelta(days=i)
        days_between.append(day)
    stripped_df = stripped_df[stripped_df.Date.isin(days_between)]

    return stripped_df

def stripped_df_to_date_X_y(stripped_df):
    df_as_np = stripped_df.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)

def get_dates(df):
    while True:
        start_date = input('First date (YYYY-MM-DD): ')
        try:
            start_date = str_to_datetime(start_date)
        except:
            continue
        else:
            if start_date in df.index:
                break
    while True:
        end_date = input('Last date (YYYY-MM-DD): ')
        try:
            end_date = str_to_datetime(end_date)
        except:
            continue
        else:
            if end_date in df.index:
                break
    return start_date, end_date

start_date, end_date = get_dates(df)
stripped_df = stripped_dataframe(start_date, end_date, df)
dates, X, y = stripped_df_to_date_X_y(stripped_df)

q_80 = int(len(dates) * 0.8)
q_90 = int(len(dates) * 0.9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_value, X_value, y_value = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]


#print(stripped_df.head())

model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

model.fit(X_train, y_train, validation_data=(X_value, y_value), epochs=100)

train_predictions = model.predict(X_train).flatten()
value_predictions = model.predict(X_value).flatten()
test_predictions = model.predict(X_test).flatten()


plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.plot(dates_value, value_predictions)
plt.plot(dates_value, y_value)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(['Training Predictions',
           'Training Observations',
           'Validation Predictions',
           'Validation Observations',
           'Testing Predictions',
           'Testing Observations'])

plt.show()

