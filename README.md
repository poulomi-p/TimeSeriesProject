# TimeSeriesProject
#Time Series Forecasting: Stock price prediction

Time series forecasting is a supervised learning problem where we try to predict the current stock price of a company/business by looking into the stock prices of the past few years. Here we re-frame our data in a way so that we are able to use standard machine learning algorithms on it. Since we are trying to predict a real value here, it is a regression problem. I used LSTM model architecture (deep learning) for doing this project.

##Data Collection and Visualization

*I used a data set named Lumber-futures.csv
*The data set has 11378 observations and 9 features
*The data set started with the most recent stock price data and ended with the oldest
*After reading the csv file as a pandas dataframe, I reversed the order of the observations so that they start from the oldest and finish with the latest stock price, the dates range from 11/16/1972 to 2/2/2018
*The column that is important for this kind of project is the closing stock price for a particular day, in this data set that column is called 'Last'
*I plotted the data using just the 'Days' feature and the closing price 'Last' feature using matplotlib

##Data pre-processing
*Here the data 'Last' column is separated out as a new data frame and is converted to a numpy array
*The data is then scaled (I have used MinMaxScaler, so the values will be within 0 and 1) (required step before feeding the data into an LSTM model)
*The data is then divided into the training and test data sets (I used a 0.7 - 0.3 split)
*Within the training data we need a x_train (features) and y_train (label), here we have the window size that determines how many previous days observations do we want the current stock price to look up to, I used a window size of 100, thus the observations, indices 0 to 99, will be put into our x_train and observation 100 will be put into y_train
*The same procedure is performed on the test data as well

##Model building, training and testing
*I used a stacked LSTM (Long Short Term Memory) model architecture
*Model is compiled using the default values and mean squared error loss is calculated
*The model is trained with the data and the label with a batch_size of 64 and 100 epochs
*Then the testing data is fed into the model for prediction 
*The root mean squared error is calculated on the differrence between what the model predicted and what the actual labels were
*The error I got is 1.866, which is really good (<2%)

##Plotting the output
*Finally, the training data, test data and the model's predictions are plotted using matplotlib 
