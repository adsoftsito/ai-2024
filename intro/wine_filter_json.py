import pandas as pd

wine_reviews = pd.read_json("winemag-data-130k-v2.json")
# print(wine_reviews.loc[wine_reviews.country=='Mexico', ['country', 'points']])
#print(wine_reviews.points.describe())
print(wine_reviews.country.unique())
