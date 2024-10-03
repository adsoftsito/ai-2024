import pandas as pd

wine_reviews = pd.read_csv("winemag-data-130k-v2.csv")
#print(wine_reviews.country)
#print(wine_reviews.loc[:, ['country', 'points']])
print(wine_reviews.loc[wine_reviews.country=='Mexico', ['country', 'points']])

