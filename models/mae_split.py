import pandas as pd

melbourne_file_path = 'melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data  = melbourne_data.dropna(axis=0)
y =  melbourne_data.Price

# print(y)
#  print(y.shape)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                      'YearBuilt', 'Lattitude', 'Longtitude']



X = melbourne_data[melbourne_features]
#  print(X)
# print(X.shape)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2)

melbourne_model = DecisionTreeRegressor(random_state=1)

# fit model
#print('trainning ...')
melbourne_model.fit(train_X, train_y)
#print('done ...')

from sklearn.metrics import mean_absolute_error

val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
