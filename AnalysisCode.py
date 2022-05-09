#For displaying plotly graphs in google colab
def enable_plotly_in_cell():
  import IPython
  from plotly.offline import init_notebook_mode
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
  '''))
  init_notebook_mode(connected=False)
 
get_ipython().events.register('pre_run_cell', enable_plotly_in_cell)

#Importing the necessary libraries for analysis
import numpy as np 
import pandas as pd 
import os
import plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as FF
from plotly import tools
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.features import DivIcon
from folium.plugins import HeatMap
import warnings
import datetime
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
import gc

#Loading google drive in order to load the dataset
from google.colab import drive
drive.mount('/content/drive/')

#Run in a seperate cell only if directory not found issue arises
cd drive/My\ Drive/

#Loading the data
#dv will be used strictly for data visualization purposes 
df = pd.read_csv('./data.csv')
dv = df.copy()

df.head()

#Checking the number of missing values for each column
df.isnull().sum().sort_values(ascending=False)

#Quick stats on the data
df.info()

####################### START OF DATA CLEANING/FEATURE ENGINEERING + DATA VISUALIZATION #######################

#Replacing the missing values in 'Bathroom' with the median of the 'Bathroom' count in houses
#with similar number of rooms
median_bath = df.groupby(['Rooms'])['Bathroom'].median()
def fillna_bath(row, median_bath):
        bath = median_bath.loc[row['Rooms']]
        return bath
df['Bathroom'].fillna(df['Bathroom'].median(), inplace=True)

#Replacing missing values in 'Car' with the median of the 'Car' count in houses
#with similar number of rooms
median_car = df.groupby(['Rooms'])['Car'].median()
def fillna_bath(row, median_car):
        car = median_car.loc[row['Rooms']]
        return car
df['Car'].fillna(df['Car'].median(), inplace=True)

#Since 'Postcode' only has 1 missing value, replacing it with the mode
most_common = df['Postcode'].value_counts().index[0]
df['Postcode'].fillna(most_common, inplace=True)

#Changing the datatypes of 'Postcode', 'Car' and 'Rooms'
df['Postcode'] = df['Postcode'].astype('object')
df['Bathroom'] = df['Bathroom'].astype('int')
df['Car'] = df['Car'].astype('int')

#For the data analysis part, dropping the rows where Price is null
dv = dv[dv['Price'].notnull()]

#Visualizing the Property Locations on a map
#Creating our folium map
#Prefer canvas forces leaflet.js to use the canvas backend(if available) for vector layers instead of svg
folium_map = folium.Map(prefer_canvas=True)

#Creating a temporary dataframe to make sure location values are not null
temp = dv[dv['Lattitude'].notnull() & dv['Longtitude'].notnull()]

#Function to help us iterate through the rows of Latitude and Longitude values and display it on a map with circles
def plotDot(point):
    folium.CircleMarker(location=[point.Lattitude, point.Longtitude],
                        radius=1,
                        weight=3).add_to(folium_map)

#Iterating through every row of our temp dataframe
temp.apply(plotDot, axis=1)

#Set the zoom to the maximum possible
folium_map.fit_bounds(folium_map.get_bounds())

#Displaying the map
folium_map

#Price Distribution Histograms
#Plotting the price distribution for the different regions
all = dv['Price'].values
north = dv['Price'].loc[dv['Regionname'] == 'Northern Metropolitan'].values
south = dv['Price'].loc[dv['Regionname'] == 'Southern Metropolitan'].values
east = dv['Price'].loc[dv['Regionname'] == 'Eastern Metropolitan'].values
west = dv['Price'].loc[dv['Regionname'] == 'Western Metropolitan'].values
southeast = dv['Price'].loc[dv['Regionname'] == 'South-Eastern Metropolitan'].values
northern_v = dv['Price'].loc[dv['Regionname'] == 'Northern Victoria'].values
eastern_v = dv['Price'].loc[dv['Regionname'] == 'Eastern Victoria'].values
western_v = dv['Price'].loc[dv['Regionname'] == 'Western Victoria'].values

#Plotting the histograms for each individual region
overall_price = go.Histogram(
    x=all,
    histnorm='', 
    name='All Regions',
    marker=dict(
        color='#6E6E6E'
    )
)

northern_metropolitan = go.Histogram(
    x=north,
    histnorm='', 
    name='Northern Metropolitan',
    marker=dict(
        color='#2E9AFE'
    )
)

southern_metropolitan = go.Histogram(
    x=south,
    histnorm='', 
    name='Southern Metropolitan',
    marker=dict(
        color='#FA5858'
    )
)

eastern_metropolitan = go.Histogram(
    x=east,
    histnorm='', 
    name='Eastern Metropolitan',
    marker=dict(
        color='#81F781'
    )
)

western_metropolitan = go.Histogram(
    x=west,
    histnorm='', 
    name='Western Metropolitan',
    marker=dict(
        color='#BE81F7'
    )
)

southeastern_metropolitan = go.Histogram(
    x=southeast,
    histnorm='', 
    name='SouthEastern Metropolitan',
    marker=dict(
        color='#FE9A2E'
    )
)

northern_victoria = go.Histogram(
    x=northern_v,
    histnorm='', 
    name='Northern Victoria',
    marker=dict(
        color='#04B4AE'
    )
)

eastern_victoria = go.Histogram(
    x=eastern_v,
    histnorm='', 
    name='Eastern Victoria',
    marker=dict(
        color='#088A08'
    )
)


western_victoria = go.Histogram(
    x=western_v,
    histnorm='', 
    name='Western Victoria',
    marker=dict(
        color='#8A0886'
    )
)

#Creating the subplots for each individual histogram
fig = tools.make_subplots(rows=5, cols=2, print_grid=False, specs=[[{'colspan': 2}, None], [{}, {}], [{}, {}], [{}, {}], [{}, {}]],
                         subplot_titles=(
                             'Overall Price Distribution',
                             'Northern Metropolitan',
                             'Southern Metropolitan',
                             'Eastern Metropolitan',
                             'Western Metropolitan',
                             'SouthEastern Metropolitan',
                             'Northern Victoria',
                             'Eastern Victoria',
                             'Western Victoria'
                             ))

#Attaching the histograms to our figure subplot at the desired locations
fig.append_trace(overall_price, 1, 1)
fig.append_trace(northern_metropolitan, 2, 1)
fig.append_trace(southern_metropolitan, 2, 2)
fig.append_trace(eastern_metropolitan, 3, 1)
fig.append_trace(western_metropolitan, 3, 2)
fig.append_trace(southeastern_metropolitan, 4, 1)
fig.append_trace(northern_victoria, 4, 2)
fig.append_trace(eastern_victoria, 5, 1)
fig.append_trace(western_victoria, 5, 2)

fig['layout'].update(showlegend=False, title="Distribution of Prices by Region",
                    height=1200, width=800)
fig.show()

df.describe()

#Replacing the missing values in some of the columns with the median
#since median is less prone to getting affected by the outliers instead of mean
df['Landsize'].fillna(df['Landsize'].median(), inplace=True)
df['BuildingArea'].fillna(df['BuildingArea'].median(), inplace=True)

#Replacing landsize=0 with the median
df['Landsize'] = df['Landsize'].replace(0, df['Landsize'].median())

#Filling in some of the categorical columns with the mode
most_common = df['Postcode'].value_counts().index[0]
df['Postcode'].fillna(most_common, inplace=True)

most_common = df['Regionname'].value_counts().index[0]
df['Regionname'].fillna(most_common, inplace=True)

most_common = df['CouncilArea'].value_counts().index[0]
df['CouncilArea'].fillna(most_common, inplace=True)

#Since property_count also contains only 3 missing values, so I'm filling it up
#with the mode
most_common = df['Propertycount'].value_counts().index[0]
df['Propertycount'].fillna(most_common, inplace=True)

#Since Distance contains only 1 missing value, replacing it with the median
df['Distance'].fillna(df['Distance'].median(), inplace=True)

#Filling latitude with median of clusters
median_lat = df.groupby(['Suburb', 'Postcode', 'CouncilArea'])["Lattitude"].median()
def fillna_lat(row, median_lat):
        lat = median_lat.loc[row["Suburb"], row['Postcode'], row["CouncilArea"]]
        return lat
df["Lattitude"] = df.apply(lambda row : fillna_lat(row, median_lat) if np.isnan(row['Lattitude']) else row['Lattitude'], axis=1)

#Filling the longitude with the median of clusters
median_long = df.groupby(['Suburb', 'Postcode', 'CouncilArea'])["Longtitude"].median()
def fillna_long(row, median_long):
        long = median_long.loc[row["Suburb"], row['Postcode'], row["CouncilArea"]]
        return long
df["Longtitude"] = df.apply(lambda row : fillna_long(row, median_long) if np.isnan(row['Longtitude']) else row['Longtitude'], axis=1)

#Filling the remaining missing 89 values in both lat and long with the median
#Used df.isnull().sum().sort_values(ascending=False) to find that out
df['Lattitude'].fillna(df['Lattitude'].median(), inplace=True)
df['Longtitude'].fillna(df['Longtitude'].median(), inplace=True)

#Only the missing values in YearBuilt and Price are remaining
#I can't seem to figure out any other way of dealing with missings in this case,
#so I'm just going to replace them with the median value.
df['Price'].fillna(df['Price'].median(), inplace=True)
df['YearBuilt'].fillna(df['YearBuilt'].median(), inplace=True)

#Changing datatype of YearBuilt to int
df['YearBuilt'] = df['YearBuilt'].astype('int')

#Replacing the possible data entry errors in YearBuilt
df['YearBuilt'] = df['YearBuilt'].replace(2106, 2016)
df['YearBuilt'] = df['YearBuilt'].replace(1196, 1916)

#Converting Date to Pandas date time format and generating some new features
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")

#Creating two new columns 'Month' and 'Year'
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

#Checking properties with Building Area < 1
df.loc[df['BuildingArea'] < 1].head()

df['BuildingArea'].loc[df['BuildingArea'] < 1].value_counts()

#Checking properties with more than 6 bathrooms
df.loc[df['Bathroom'] > 6].head(3)

df['Bathroom'].loc[df['Bathroom'] > 6].value_counts()

#Exchanging Landsize and BuildingArea values where BuildingArea > LandSize
df.Landsize, df.BuildingArea = np.where(df.Landsize < df.BuildingArea, [df.BuildingArea, df.Landsize], [df.Landsize, df.BuildingArea])

#Creating a new variable called LawnSpace
df['LawnSpace'] = df['Landsize'] - df['BuildingArea']

#Visualizing correlation between Lawn Space and Price
#Adding traces to our figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['LawnSpace'], y=df['Price'],
                    mode='markers',
                    marker_color=dv['Price']))

#Setting the title and displaying our graph
fig.update_layout(title="Lawn Space vs Price correlation",
                  xaxis_title="Lawnspace (in squared. meters)")
fig.show()

df.loc[df['LawnSpace'] > 74963].head()

#Age denotes how old the property is
df['Age'] = df['Year'] - df['YearBuilt']

#Creating a temporary list for storing the seasons
temp = []

#According to the website visitmelbourne.com
#Summer months are Dec to Feb
#Winter months are June to Aug
#Spring months are Sept to Nov
#Autumn months are March to May
spring = [9, 10, 11]
summer = [12, 1, 2]
autumn = [3, 4, 5]
winter = [6, 7, 8]

for i in df['Month']:
  if i in spring:
    temp.append('Spring')
  elif i in summer:
    temp.append('Summer')
  elif i in autumn:
    temp.append('Autumn')
  else:
    temp.append('Winter')

#Creating a new feature called 'Season' from the temp list
df['Season'] = pd.Series(temp)

#Since the Season feature has now been added, let us visualize trends in Prices due to both Season as well as Property Type
#Selecting Price subset according to season
summer = df.Price[df['Season'] == 'Summer']
winter = df.Price[df['Season'] == 'Winter']
autumn = df.Price[df['Season'] == 'Autumn']
spring = df.Price[df['Season'] == 'Spring']

#Selecting Price subset according to Property type
h_ = df.Price[df['Type'] == 'h']
u_ = df.Price[df['Type'] == 'u']
t_ = df.Price[df['Type'] == 't']

#Adding the traces for each of the individual season boxplots
trace1 = go.Box(y=summer, name='Summer')
trace2 = go.Box(y=winter, name='Winter')
trace3 = go.Box(y=autumn, name='Autumn')
trace4 = go.Box(y=spring, name='Spring')

#Adding the traces for the individual property type boxplots
trace5 = go.Box(y=h_, name='Type h')
trace6 = go.Box(y=u_, name='Type u')
trace7 = go.Box(y=t_, name='Type t')

#Creating the subplots for each individual boxplot
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, specs=[[{}, {}]],
                         subplot_titles=(
                             'Price Distribution with Season',
                             'Price Distribution with Property Type'
                             ))

#Creating two seperate lists for looping through the traces of each individual
#parameter
data_season = [trace1, trace2, trace3, trace4]
data_type = [trace5, trace6, trace7]

#Looping through the data for Season as well as Property Type
for i in range(len(data_season)):
  fig.append_trace(data_season[i], 1, 1)
for i in range(len(data_type)):
  fig.append_trace(data_type[i], 1, 2)

#Hiding the legend and displaying the graph
fig['layout'].update(showlegend=False)
fig.show()

#Converting 'Date' to the correct date-time format using Pandas in visualization
#dataset as well
dv['Date'] = pd.to_datetime(dv['Date'], format="%d/%m/%Y")

#Creating two new columns 'Month' and 'Year'
dv['Month'] = dv['Date'].dt.month
dv['Year'] = dv['Date'].dt.year

#Sales Per Month for (2016, 2017 & 2018)
#Sales per month for the years 2016, 2017 and 2018
def month_year_sales(df, month, year):
    monthly_sales = df['Price'].loc[(df['Month'] == month) & (df['Year'] == year)].sum()
    return monthly_sales

#Defining a list of labels for the months
labels = ['January', 'February', 'March', 'April',
          'May', 'June', 'July', 'August', 'September', 
          'October', 'November', 'December']

#Intializing empty lists in order to store the values for each year
list_2016 = []
list_2017 = []
list_2018 = []

#Sales per month in 2016
for i in range(1, 13):
  value_2016 = month_year_sales(dv, i, 2016)
  list_2016.append(value_2016)

#Sales per month in 2017
for i in range(1, 13):
  value_2017 = month_year_sales(dv, i, 2017)
  list_2017.append(value_2017)

#Sales per month in 2018 (until March)
for i in range(1, 4):
  value_2018 = month_year_sales(dv, i, 2018)
  list_2018.append(value_2018)

#Plotting the scatter plots for each of the individual years
plot_2016 = go.Scatter(
    x=list_2016,
    y=labels,
    xaxis='x2',
    yaxis='y2',
    mode='markers',
    name='2016',
    marker=dict(
        color='rgba(0, 128, 128, 0.95)',
        line=dict(
            color='rgba(56, 56, 56, 1)',
            width=1.5,
        ),
        symbol='circle',
        size=16,
    )
)

plot_2017 = go.Scatter(
    x=list_2017,
    y=labels,
    xaxis='x2',
    yaxis='y2',
    mode='markers',
    name='2017',
    marker=dict(
        color='rgba(255, 72, 72, 0.95)',
        line=dict(
            color='rgba(56, 56, 56, 1)',
            width=1.5,
        ),
        symbol='circle',
        size=16,
    )
)

plot_2018 = go.Scatter(
    x=list_2018,
    y=labels,
    xaxis='x2',
    yaxis='y2',
    mode='markers',
    name='2018',
    marker=dict(
        color='rgba(72, 255, 72, 0.95)',
        line=dict(
            color='rgba(56, 56, 56, 1)',
            width=1.5,
        ),
        symbol='circle',
        size=16,
    )
)

#Generating the data for our layout
data = [plot_2016, plot_2017, plot_2018]

#Displaying the scatterplot
fig = go.Figure(data=data)
fig.update_layout(
    title="Sales Per Month for the Years (2016, 2017 & 2018)",
    xaxis_title="Sales in Billions"
)
fig.show()

#Most Expensive Regions?
#Finding out the most expensive regions
filtering_cond = dv.groupby('Regionname').count() > 1
sorted_region_prices = dv.groupby('Regionname').mean()[filtering_cond]['Price'].sort_values(ascending=False)
x = sorted_region_prices.index
y = sorted_region_prices
fig = go.Figure(data=[go.Bar(x=x, y=y)])

#Adding some customization
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='Average Price of Properties by Region')
fig.show()

#Visualizing Correlation using Heatmaps
#Visualizing the correlations among the various features in our dataset
fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(dv.corr(), linewidths=0.5, cmap="coolwarm")

#Dropping 'Bedroom2' from our original dataframe
df = df.drop(['Bedroom2'], axis=1)

#Most Expensive Council Areas?
#Finding out the most expensive Council Areas
filtering_cond = dv.groupby('CouncilArea').count() > 1
sorted_council_prices = dv.groupby('CouncilArea').mean()[filtering_cond]['Price'].sort_values(ascending=False)
x = sorted_council_prices.index
y = sorted_council_prices
fig = go.Figure(data=[go.Bar(x=x, y=y)])

#Adding some customization
fig.update_traces(marker_color='#EEF4ED', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
fig.update_layout(title_text='Average Price of Properties by Council Area')
fig.show()

#Visualizing the relation between Rooms available and Price
#Generating data points for individual number of rooms
one_room = dv.Price[dv['Rooms'] == 1]
two_room = dv.Price[dv['Rooms'] == 2]
three_room = dv.Price[dv['Rooms'] == 3]
four_room = dv.Price[dv['Rooms'] == 4]
five_room = dv.Price[dv['Rooms'] == 5]
six_room = dv.Price[dv['Rooms'] == 6]
seven_room = dv.Price[dv['Rooms'] == 7]
eight_room = dv.Price[dv['Rooms'] == 8]

#Adding the traces to our Box Plot
#Data points beyond room count = 8 were discarded due to very less amount of data
trace1 = go.Box(y=one_room, name='1 Room')
trace2 = go.Box(y=two_room, name='2 Rooms')
trace3 = go.Box(y=three_room, name='3 Rooms')
trace4 = go.Box(y=four_room, name='4 Rooms')
trace5 = go.Box(y=five_room, name='5 Rooms')
trace6 = go.Box(y=six_room, name='6 Rooms')
trace7 = go.Box(y=seven_room, name='7 Rooms')
trace8 = go.Box(y=eight_room, name='8 Rooms')

#Generating the data and the boxplot layout
data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]
layout = go.Layout(title='Correlation between Number of Rooms and Price')

#Displaying the figure
fig = go.Figure(data=data, layout=layout)
fig.show()

#Visualizing the relation between Distance and Price
#Adding traces to our figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=dv['Distance'], y=dv['Price'],
                    mode='markers',
                    marker_color=dv['Price']))

#Setting the title and displaying our graph
fig.update_layout(title="Distance vs Price correlation",
                  xaxis_title="Distance (in KM)")
fig.show()

#Investigating the 11.2M dollar property
df.loc[df['Price'] > 11000000].head()

####################### END OF DATA CLEANING/FEATURE ENGINEERING + DATA VISUALIZATION #########################

####################### USING MULTIPLE MACHINE LEARNING MODELS FOR PREDICTIONS ################################

#Making the necessary imports before building our models
import sklearn

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
#One hot encoding categorical variables
df_ = pd.get_dummies(df, columns=['Type', 'Regionname', 'Season', 'CouncilArea'])

#Dropping some features
df_ = df_.drop(['Method', 'SellerG', 'Address', 'Suburb', 'Date'], axis=1)

#First sorting the dataset according to the year in ascending order
df_ = df_.sort_values(by=['Year'], ascending=True)

#Checking value_counts() of years
df_['Year'].value_counts()

#Since almost 14% of the data is from the year 2018,
#I will be doing a 80-20% split in the training set in order to make
#predictions for late 2017 as well as 2018 based on previous trends
df_train, df_test = np.split(df_, [int(.80*len(df_))])

#Applying the Robust Scaler to our dataset
scaler = RobustScaler()

#Dropping Price from both train and test set
target = df_test['Price']
df_test.drop('Price', axis=1, inplace=True)

y_train = df_train['Price']
df_train.drop('Price', axis=1, inplace=True)

#Applying robust scaler to both train and test
df_train = scaler.fit_transform(df_train)
df_test = scaler.fit_transform(df_test)

#Applying the SVM Regressor
svr_regressor = SVR()
svr_regressor.fit(df_train, y_train)

#Necessary imports for RMSLE (Root Mean Square Log Error) loss function
import math
from math import sqrt
from sklearn.metrics import mean_squared_error

#Making predictions on the test set
preds = svr_regressor.predict(df_test)

#Printing the rmsle score
print('RMSLE of our baseline SVR model is: ', sqrt(mean_squared_error(np.log(target), np.log(preds))))

#Doing some hyperparameter tuning using GridSearchCV for our baseline model
#Define the parameter range first
param_grid = {'C': [0.1, 1],   
              'kernel': ['rbf']} 
grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3) 
grid.fit(df_train, y_train)

#Printing out the best parameters after tuning
print(grid.best_params_)

#Generating the accuracy score for our SVR model
#Since the best parameters turned out to be the default params itself, so
#there's no point in running the algorithm again.
models = [('SVR', svr_regressor)]

for i, model in models:
    predictions = model.predict(df_test)
    errors = abs(predictions - target)

    #Calculating the mean_absolute_percentage_error
    mape = np.mean(100 * (errors / target))
    #Generating the accuracy from MAPE
    accuracy = 100 - mape    

    msg = "%s = %.2f"% (i, round(accuracy, 2))
    print('Accuracy of', msg,'%')

#Since tree based models do not require feature scaling, I'm going to use
#the original dataframes instead of the feature scaled ones
df_train, df_test = np.split(df_, [int(.80*len(df_))])

target = df_test['Price']
df_test.drop('Price', axis=1, inplace=True)

y_train = df_train['Price']
df_train.drop('Price', axis=1, inplace=True)

#Fitting the random forest model to our training and test set
random_forest = RandomForestRegressor(n_jobs=-1)
random_forest.fit(df_train, y_train)

#Fitting a decision tree model to our training and test set
decision_tree = DecisionTreeRegressor()
decision_tree.fit(df_train, y_train)

#Generating predictions for both models and comparing
models = [('Decision Tree', decision_tree), ('Random Forest', random_forest)]

#Looping through each of the models
for name, model in models:
  preds = model.predict(df_test)
  rmsle = sqrt(mean_squared_error(np.log(target), np.log(preds)))
  popup = "%s = %.2f" % (name, round(rmsle, 2))
  print("Root Mean Square Logarithmic Error (RMSLE) for", popup)

#To visualize the results in terms of the accuracy metric we can use MAPE
for i, model in models:
    predictions = model.predict(df_test)
    errors = abs(predictions - target)

    #Calculating the mean_absolute_percentage_error
    mape = np.mean(100 * (errors / target))
    #Generating the accuracy from MAPE
    accuracy = 100 - mape    

    info = "%s = %.2f"% (i, round(accuracy, 2))
    print('Accuracy of', info,'%')

#Hyperparamter optimization of Decision Trees using GridSearchCV
#The depth of a tree is one of the most important parameters in order to prevent overfitting
param_grid = {'max_depth': np.arange(3, 10)}

#Feeding our model to GridSearchCV
grid_search_tree = GridSearchCV(decision_tree, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search_tree.fit(df_train, y_train)

#Printing out the best parameters
print(grid_search_tree.best_params_)

#Testing the performance of the best parameters
grid_best = grid_search_tree.best_estimator_.predict(df_test)

#Calculating the mean_absolute_percentage_error (MAPE)
errors = abs(grid_best - target)
mape = np.mean(100 * (errors / target))

#Generating the accuracy score and displaying it
accuracy = 100 - mape   

#Generating the RMSLE score as well
rmsle = sqrt(mean_squared_error(np.log(target), np.log(grid_best)))

print('The best Decision Tree model from grid-search has a RMSLE of', round(rmsle, 2),'%')
print('The best Decision Tree model from grid-search has an accuracy of', round(accuracy, 2),'%')

#Hyperparameter optimization of RF using GridSearchCV
#Takes about 40 mins to run
param_grid = [
{'n_estimators': [50, 100, 200], 'max_features': [5, 10], 
 'max_depth': [10, 50, None], 'bootstrap': [True, False]}
]

grid_search_forest = GridSearchCV(random_forest, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search_forest.fit(df_train, y_train)

#Printing out the best parameters
print(grid_search_forest.best_params_)

#Testing the performance of the best parameters
grid_best = grid_search_forest.best_estimator_.predict(df_test)

#Calculating the mean_absolute_percentage_error (MAPE)
errors = abs(grid_best - target)
mape = np.mean(100 * (errors / target))

#Generating the accuracy score and displaying it
accuracy = 100 - mape   

#Generating the RMSLE score as well
rmsle = sqrt(mean_squared_error(np.log(target), np.log(grid_best)))

print('The best RF model from grid-search has a RMSLE of', round(rmsle, 2),'%')
print('The best RF model from grid-search has an accuracy of', round(accuracy, 2),'%')

#For Random Forest
importances = grid_search_forest.best_estimator_.feature_importances_
features = list(df_train.columns)

#Creating a list of tuples
#Credits for the use of zip: https://realpython.com/python-zip-function/
feature_importances = sorted(zip(importances, features), reverse=True)

#Create a dataframe and store the tuple as two lists in the dataframe
imp_df = pd.DataFrame(feature_importances, columns=['importance', 'feature'])
importance= list(imp_df['importance'])
feature= list(imp_df['feature'])

imp_df.head()

#Plotting it on a graph
plt.style.use('fivethirtyeight')

x_vals = list(range(len(feature_importances)))

#Plotting a horizontal bar chart
plt.figure(figsize=(15,10))
plt.bar(x_vals, importance, orientation = 'vertical')

#Setting x axis tick labels
plt.xticks(x_vals, feature, rotation='vertical')

#Setting the title and axes labels
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.title('Feature Importance by Random Forest')

#For Decision Trees
importances = grid_search_tree.best_estimator_.feature_importances_
features = list(df_train.columns)

#Creating a list of tuples
#Credits for the use of zip: https://realpython.com/python-zip-function/
feature_importances = sorted(zip(importances, features), reverse=True)

#Create a dataframe and store the tuple as two lists in the dataframe
imp_df = pd.DataFrame(feature_importances, columns=['importance', 'feature'])
importance= list(imp_df['importance'])
feature= list(imp_df['feature'])

imp_df.head()

#Plotting it on a graph
plt.style.use('fivethirtyeight')

x_vals = list(range(len(feature_importances)))

#Plotting a horizontal bar chart
plt.figure(figsize=(15,10))
plt.bar(x_vals, importance, orientation = 'vertical')

#Setting x axis tick labels
plt.xticks(x_vals, feature, rotation='vertical')

#Setting the title and axes labels
plt.ylabel('Importance Score')
plt.xlabel('Features')
plt.title('Feature Importance by Decision Tree')

