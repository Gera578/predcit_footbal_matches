#IA Machine Learning Project
#Predicting Football Match Outcomes with Machine Learning in Python using the Random Forest Classifier
#The goal of this project is to predict the outcome of football matches using machine learning in Python. We will use the Random Forest Classifier to train a model on historical match data and make predictions on future matches. We will also calculate the precision score of the model to evaluate its performance.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.metrics import precision_score  # type: ignore
matches = pd.read_csv('matches.csv', index_col=0)


pd.set_option('display.max_columns', None)  # Show all columns

#print(matches[matches["team"] == "Liverpool"]) <- to get all matches played by Liverpool

#print(matches["round"].value_counts()) <- to get the number of matches played in each round

#print(matches.dtypes) #<- to get the data types of each column   

matches["date"] = pd.to_datetime(matches["date"]) #<- to convert the date column to datetime


matches["venue_code"] = matches["venue"].astype("category").cat.codes

matches["opp_code"] = matches["opponent"].astype("category").cat.codes

matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

matches["day_code"] = matches["date"].dt.dayofweek

matches["target"] = (matches["result"] == "W").astype("int")

#train the model

# Initialize a RandomForestClassifier with 50 trees, a minimum of 10 samples required to split an internal node, and a random state for reproducibility
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# Split the data into training and test sets based on the date
train = matches[matches["date"] < '2022-01-01']  # Training data: matches before 2022-01-01
test = matches[matches["date"] > '2022-01-01']  # Test data: matches after 2022-01-01

# Define the predictors (features) to be used for training the model
predictors = ["venue_code", "opp_code", "hour", "day_code"]

# Train the RandomForestClassifier using the training data
rf.fit(train[predictors], train["target"])

# Make predictions on the test data
preds = rf.predict(test[predictors])

# Calculate the accuracy of the model on the test data
acc = accuracy_score(test["target"], preds)

# Create a DataFrame to compare the actual target values with the predicted values
combined = pd.DataFrame(dict(actual=test["target"], predictors=preds))

# Create a cross-tabulation of the actual vs predicted values
pd.crosstab(index=combined["actual"], columns=combined["predictors"])

# Calculate the precision score of the model on the test data
precision_score(test["target"], preds)
# Group the matches by team
grouped_matches = matches.groupby("team")

# Get the matches for Manchester City
group = grouped_matches.get_group("Manchester City")

# Define a function to calculate the rolling average of specified columns
def rolling_avg(group, colms, new_col):
    group = group.sort_values("date")  # Sort the group by date
    rolling_stats = group[colms].rolling(3, closed='left').mean()  # Calculate the rolling mean
    group[new_col] = rolling_stats  # Add the rolling mean to a new column
    group = group.dropna(subset=new_col)  # Drop rows with NaN values in the new column
    return group

# List of columns to calculate the rolling average for
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]

# Create new column names for the rolling averages
new_cols = [f"{c}_cols" for c in cols]

# Calculate the rolling averages for the Manchester City group
n = rolling_avg(group, cols, new_cols)

# Apply the rolling average function to each team in the matches DataFrame
matches_rolling = matches.groupby("team").apply(lambda x: rolling_avg(x, cols, new_cols))

# Remove the team level from the index
matches_rolling = matches_rolling.droplevel('team')

# Reset the index of the DataFrame
matches_rolling.index = range(matches_rolling.shape[0])

# Define a function to make predictions and calculate precision
def make_prediction(data, predictors):
    train = data[data["date"] < '2022-01-01']  # Training data: matches before 2022-01-01
    test = data[data["date"] > '2022-01-01']  # Test data: matches after 2022-01-01
    
    rf.fit(train[predictors], train["target"])  # Train the RandomForestClassifier
    preds = rf.predict(test[predictors])  # Make predictions on the test data
    
    # Create a DataFrame to compare the actual target values with the predicted values
    combined = pd.DataFrame(dict(actual=test["target"], predictors=preds))
    precision = precision_score(test["target"], preds)  # Calculate the precision score
    
    return combined, precision

# Make predictions and calculate precision using the rolling averages and predictors
combined, precision = make_prediction(matches_rolling, predictors + new_cols)

# Merge the combined DataFrame with the matches DataFrame to include additional columns
combined = combined.merge(matches_rolling[["team", "date", "opponent", "result"]], left_index=True, right_index=True)

class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Newcastle United": "Newcastle Utd",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves" 
}

mapping = MissingDict(**map_values)

combined["new_team"] = combined["team"].map(mapping)

merged = combined.merge(combined, left_on = ["date","new_team"], right_on = ["date","opponent"])

merged[(merged["predictors_x"]==1) & (merged["predictors_y"] == 0)]["actual_x"].value_counts()
