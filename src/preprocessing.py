import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Drop duplicate rows after disregarding booking ID
def drop_duplicates(dataframe): 
    print(f"There were {dataframe.shape[0]} rows and there were {dataframe['booking_id'].nunique()} unique booking IDs in df. Hence, no booking ID was repeated twice.")
    df_no_id = dataframe.drop('booking_id', axis = 1)
    df = df_no_id.drop_duplicates()
    print(f"There were {df.shape[0]} rows after the removal of duplicate booking records.")
    return df 

# Converts prices in USD to SGD
def currency_converter(dataframe): 
    dataframe['currency'] = dataframe['price'].astype(str).str[0:3]
    dataframe['price'] = dataframe['price'].astype(str).str[4:]
    dataframe['price'] = pd.to_numeric(dataframe['price'], errors = 'coerce')
    dataframe['price'] = dataframe.apply(lambda x: x['price'] * 1.35 if x['currency'] == 'USD' else x['price'], axis = 1)
    dataframe.drop(['currency'], axis = 1)
    return dataframe

def standardize_month(df): 
    df['arrival_month'] = df['arrival_month'].str.lower()
    df['booking_month'] = df['booking_month'].str.lower()
    df['checkout_month'] = df['checkout_month'].str.lower()
    
    month_mapping = {
    'january': 1,
    'february': 2,
    'march': 3,
    'april': 4,
    'may': 5,
    'june': 6,
    'july': 7,
    'august': 8,
    'september': 9,
    'october': 10,
    'november': 11,
    'december': 12}
    
    df['booking_month'] = df['booking_month'].map(month_mapping)
    df['arrival_month'] = df['arrival_month'].map(month_mapping)
    df['checkout_month'] = df['checkout_month'].map(month_mapping)

    df['lag_month'] = df.apply(lambda row: row['arrival_month'] - row['booking_month'] if row['arrival_month'] >= row['booking_month'] else (row['arrival_month'] + 12 - row['booking_month']), axis=1)
    return df 


def standardize_num_adults(df): 
    adult_mapping = {
    'one': 1, 
    'two': 2, 
    '1':1, 
    '2':2}
    
    df['num_adults'] = df['num_adults'].map(adult_mapping)
    return df 

class PricePredictor:
    def __init__(self, dataframe):
        self.df = dataframe
        self.rf_regressor = RandomForestRegressor()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df_test = None

    def train_test_split(self):
        df_train = self.df[self.df['price'].notna()][['branch', 'booking_month', 'lag_month', 'num_adults', 'num_children', 'platform', 'room', 'price']]
        self.df_test = self.df[self.df['price'].isna()]
        
        X = df_train.drop('price', axis=1)
        y = df_train['price']
        
        X = pd.get_dummies(X, columns=['branch', 'platform', 'room'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    def missing_price_predictor(self):
        self.rf_regressor.fit(self.X_train, self.y_train)
        predicted_price = self.rf_regressor.predict(self.X_test)
        difference = root_mean_squared_error(predicted_price, self.y_test)
        print('Difference between predicted and actual price based on RMSE:', difference)

    def impute_missing_price(self):
        df_predict_encoded = pd.get_dummies(self.df_test, columns=['branch', 'platform', 'room'])
        df_predict_encoded = df_predict_encoded.reindex(columns=self.X_train.columns, fill_value=0)
        predicted_prices = self.rf_regressor.predict(df_predict_encoded)
        self.df.loc[self.df['price'].isna(), 'price'] = predicted_prices
        return self.df
    

# Dealing with features with null values 
def deal_with_null(dataframe): 
    df = dataframe[dataframe['no_show'].isna() == False]
    df = currency_converter(df)

    # Dealing with missing room records 
    conditions_changi = [
    (df['branch'] == 'Changi') & (df['price'] <= 600) & df['room'].isna(),
    (df['branch'] == 'Changi') & (df['price'] > 600) & (df['price'] <= 800) & df['room'].isna(),
    (df['branch'] == 'Changi') & (df['price'] > 800) & (df['price'] <= 1200) & df['room'].isna(),
    (df['branch'] == 'Changi') & (df['price'] > 1200) & df['room'].isna()]
    
    choices_changi = ['Single', 'Queen', 'King', 'President Suite']
    conditions_orchard = [
    (df['branch'] == 'Orchard') & (df['price'] <= 900) & df['room'].isna(),
    (df['branch'] == 'Orchard') & (df['price'] > 900) & (df['price'] <= 1200) & df['room'].isna(),
    (df['branch'] == 'Orchard') & (df['price'] > 1200) & (df['price'] < 1800) & df['room'].isna(),
    (df['branch'] == 'Orchard') & (df['price'] >= 1800) & df['room'].isna()]
    
    choices_orchard = ['Single', 'Queen', 'King', 'President Suite']
    conditions = conditions_changi + conditions_orchard
    choices = choices_changi + choices_orchard
    df['room'] = np.select(conditions, choices, default=df['room'])
    print(df.isna().sum()) 

    # Dealing with missing price records
    df = standardize_month(df)
    df = standardize_num_adults(df)
    predictor = PricePredictor(df)
    predictor.train_test_split()
    predictor.missing_price_predictor()
    df = predictor.impute_missing_price()
    print(df.isna().sum())
    return df

def preprocessing(df): 
    df = drop_duplicates(df)
    df = deal_with_null(df)
    print(df.head())
    return df 

























    
