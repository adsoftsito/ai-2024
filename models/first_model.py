import pandas as pd

melbourne_file_path = 'melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data  = melbourne_data.dropna(axis=0)
y =  melbourne_data.Price

print(y)
print(y.shape)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 
                      'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]
print(X)
print(X.shape)

from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1)

# fit model
print('trainning ...')
melbourne_model.fit(X, y)
print('done ...')

print('Making predictions for the following 5 houses')
print(X.head())
print('The predictions are')
print(melbourne_model.predict(X.head()))
print('Real prices')
print(y.head())
