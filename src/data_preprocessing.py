from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    # Load California housing dataset
    housing = fetch_california_housing(as_frame=True)
    data = housing.frame
    data.columns = housing.feature_names + ['PRICE']  # Rename the target column to 'PRICE'
    return data

def preprocess_data(data):
    # Split data into features and target variable
    X = data.drop('PRICE', axis=1)
    y = data['PRICE']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test




#import pandas as pd
#from sklearn.model_selection import train_test_split
#
#def load_data():
    # Load dataset (example with Boston dataset)
 #   from sklearn.datasets import load_boston
  #  boston = load_boston()
   # data = pd.DataFrame(boston.data, columns=boston.feature_names)
#    data['PRICE'] = boston.target
 #   return data

#def preprocess_data(data):
    # Split data into features and target variable
#    X = data.drop('PRICE', axis=1)
 #   y = data['PRICE']

  #  # Split into train and test sets
  #  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   # return X_train, X_test, y_train, y_test

