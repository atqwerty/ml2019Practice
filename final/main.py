import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import cv2

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from datetime import datetime, date


def split_date(date_string):
    return date_string.split(' ')


def calculate_days(date_string_1, date_string_2):
    date_1 = datetime.strptime(date_string_1, "%Y-%m-%d")
    date_2 = datetime.strptime(date_string_1, "%Y-%m-%d")

    d1 = date(year=date_1.year, month=date_1.month, day=date_1.day)
    d2 = date(year=date_2.year, month=date_2.month, day=date_2.day)

    return (d1 - d2).days


def time_in_hours(date_string):
    date_num = datetime.strptime(date_string, "%H:%M:%S")

    hr = ((date_num.hour * 60) + date_num.minute) / (24 * 60)

    return hr


def pct_more_than_one(pct):
    return ('%1.f%%' % pct) if pct > 1 else ''


def risk(latitude, longitude):
    west = 25.67
    east = 44.81
    south = 35.81
    north = 42.10

    real_width = east - west
    real_height = north - south

    map_width = np.shape(risk_map)[1]
    map_height = np.shape(risk_map)[0]

    width_ratio = map_width/(real_width*100)
    height_ratio = map_height/(real_height*100)

    # Calculating pixels to look up for the grade
    easting = longitude-west
    northing = latitude-south

    pixel_x = int(round(easting*100*width_ratio))
    pixel_y = map_height-int(round(northing*100*height_ratio))

    # Correction of the error caused by floating points
    if pixel_x >= map_width:
        pixel_x = map_width-1

    if pixel_y >= map_height:
        pixel_y = map_height-1

    grade = risk_map[pixel_y, pixel_x]

    return grade


df = pd.read_csv('./earthquakes/Earthquakes.csv')
df['Date(UTC)'][0].split(' ')

dates = []
times = []

for i in range(len(df)):
    cur_date, cur_time = split_date(df.loc[i].iloc[1])
    dates.append(cur_date)
    times.append(cur_time)
dates = pd.Series(dates)
times = pd.Series(times)

df.insert(1, 'Date', dates, allow_duplicates=True)
df.insert(1, 'Time', times, allow_duplicates=True)
df.drop(columns='Date(UTC)', inplace=True)

df = df.sort_values(['Date', 'Time'], ascending=[True, True])
df = df.reset_index(drop=True)
begin_date = '1900-01-01'
df['Time gap'] = ""
df['Converted Date'] = ""
df['Converted Time'] = ""
df['Converted Date'].loc[0] = calculate_days(df['Date'][0], begin_date)
df['Time gap'].loc[0] = 0
df['Converted Time'].loc[0] = time_in_hours(df['Time'].loc[0])
for i in range(1, len(df['Date'])):
    df['Time gap'].loc[i] = calculate_days(df['Date'][i], df['Date'][i - 1])
    df['Converted Date'].loc[i] = calculate_days(df['Date'][i], begin_date)
    df['Converted Time'].loc[i] = time_in_hours(df['Time'][i])
df.drop(columns=['Date', 'Time'], inplace=True)

df['Converted Time'] = pd.to_numeric(df['Converted Time'])
df['Constant Deg.'] = df['Constant Deg.'].replace({'No': 0, 'Yes': 1})

df = df.rename(columns={"Constant Deg.": "ConstatnDeg"})

cat_cols = list(df.dtypes[df.dtypes == 'object'].index)

for i in cat_cols:
    print(i)
    print("=====")
    print(df[i].unique())
    print()

for col in list(df.dtypes[df.dtypes == 'object'].index):
    df[col].value_counts().plot(kind='pie', autopct=pct_more_than_one,
                                labels=None, figsize=(15, 15), title=col)
    plt.legend(df[col].unique())
    plt.show()

df.drop(columns=['No', 'Ref1', 'Source Description 1', 'Source No 2',
                 'Source Description 2', 'Source No 3', 'Source Description 3',
                 'Type'], inplace=True)

img = cv2.imread("./earthquakes/risk_map_clean.jpg", cv2.IMREAD_GRAYSCALE)
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)

blur = cv2.GaussianBlur(img, (7, 7), 0)
for i in range(105):
    blur = cv2.GaussianBlur(blur, (7, 7), 0)
# plt.imshow(blur, cmap='gray', vmin=0, vmax=255), plt.title('Blurred')

size = 6
increment = 2
epoch = 4

recovered_img = img.copy()

for i in range(0, epoch):
    width_step = np.shape(recovered_img)[1] / size
    height_step = np.shape(recovered_img)[0] / size

    for h in range(0, int(height_step)):
        for w in range(0, int(width_step)):
            window = recovered_img[h * size:(h + 1) * size, w * size:(w + 1) * size]

            if i == 0:
                window = window.max()
            else:
                window = window.min()

            recovered_img[h * size:(h + 1) * size, w * size:(w + 1) * size] = window

    size += increment
# plt.imshow(recovered_img, cmap='gray', vmin=0, vmax=255)

risk_map = recovered_img.copy()

high = 75
medium = 150
low = 225
no_data = 250
default = 5

risk_map = np.where(risk_map <= high, 4, risk_map)
risk_map = np.where(((risk_map > high) & (risk_map <= medium)), 3, risk_map)
risk_map = np.where(((risk_map > medium) & (risk_map <= low)), 2, risk_map)
risk_map = np.where(((risk_map > low) & (risk_map <= no_data)), 1, risk_map)
risk_map = np.where(risk_map > no_data, default, risk_map)
plt.imshow(risk_map, cmap='gray', vmin=1, vmax=5)

print(risk_map.shape)
grades = []
for i in range(len(df)):
    grades.append(risk(df.loc[i].iloc[0], df.loc[i].iloc[1]))
df['Risk Grade'] = pd.Series(grades)

x = df.drop(columns=['Latitude', 'Longitude', 'Magnitude',
                     'Converted Date']).to_numpy()
y = df[['Latitude', 'Longitude', 'Magnitude', 'Converted Date']].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.2,
                                                  random_state=1)

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

x_train = scaler_x.fit_transform(x_train)
x_val = scaler_x.fit_transform(x_val)
x_test = scaler_x.fit_transform(x_test)

y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.fit_transform(y_val)
y_test = scaler_y.fit_transform(y_test)

print(y_train)

model1 = tree.DecisionTreeRegressor(max_depth=4)
model2 = tree.DecisionTreeRegressor(max_depth=5)
model3 = tree.DecisionTreeRegressor(max_depth=6)
model4 = tree.DecisionTreeRegressor(max_depth=20, splitter="random")

model4 = model4.fit(x_train, y_train)
prediction = model4.predict(x_test)

# print(prediction)
# print(x_test)
# print(y_test)

prediction = scaler_y.inverse_transform(prediction)
answers = scaler_y.inverse_transform(y_test)

plt.figure(figsize=(20, 6))
plt.scatter(answers.T[1], answers.T[0], label='Answers')
plt.scatter(prediction.T[1], prediction.T[0], label='Dense Prediction')
plt.title("Coordinates Prediction")
plt.legend()
plt.show()
