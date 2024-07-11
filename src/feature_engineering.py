import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def engineer_numerical_features(df): 
    df['num_occupants'] = df['num_adults'] + df['num_children']
    df['arrival_day'] = df['arrival_day'].astype(int)
    df['checkout_day'] = abs(df['checkout_day'].astype(int))
    
    current_year = 2020
    df['arrival_date'] = pd.to_datetime(df.apply(lambda row: f"{current_year}-{row['arrival_month']}-{row['arrival_day']}", axis=1))
    df['checkout_date'] = pd.to_datetime(df.apply(lambda row: f"{current_year}-{row['checkout_month']}-{row['checkout_day']}", axis=1))
    
    df['stay_duration'] = (df['checkout_date'] - df['arrival_date']).dt.days 
    df['stay_duration'] = df['stay_duration'].apply(lambda x: x + 366 if x<0 else x)
    return df 

def filter_features(df): 
    df = engineer_numerical_features(df)
    df = df[['branch', 'booking_month', 'arrival_month', 'country', 'first_time', 'room', 'price', 'num_occupants', 'stay_duration', 'lag_month', 'no_show']]
    return df 


class EncodingScaling:
    def __init__(self, dataframe): 
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.df = dataframe
        self.X_train = None 
        self.X_test = None 
        self.y_train = None
        self.y_test = None 

    def train_test_split(self): 
        X = self.df.drop('no_show', axis=1)
        y = self.df['no_show']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
        print(f"Shape of X_train: {self.X_train.shape}")
        print(f"Shape of X_test: {self.X_test.shape}")

    def categorical_encoder(self): 
        categorical = ['branch', 'country', 'first_time', 'room']
        encoder_dict = {}

        for feature in categorical: 
            self.encoder.fit(self.X_train[[feature]])
            encoder_dict[feature] = self.encoder
            
            encoded_train = self.encoder.transform(self.X_train[[feature]])
            encoded_test = self.encoder.transform(self.X_test[[feature]])
            encoded_feature_names = self.encoder.get_feature_names_out([feature])
            
            encoded_train = pd.DataFrame(encoded_train, columns=encoded_feature_names, index=self.X_train.index)
            encoded_test = pd.DataFrame(encoded_test, columns=encoded_feature_names, index=self.X_test.index)
            
            self.X_train = self.X_train.drop(columns=[feature])
            self.X_test = self.X_test.drop(columns=[feature])
            
            self.X_train = pd.concat([self.X_train, encoded_train], axis=1)
            self.X_test = pd.concat([self.X_test, encoded_test], axis=1)
    
    def scaling(self): 
        self.scaler.fit(self.X_train)
        
        X_train_scaled = self.scaler.transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        self.X_train = pd.DataFrame(X_train_scaled, columns=self.X_train.columns, index=self.X_train.index)
        self.X_test = pd.DataFrame(X_test_scaled, columns=self.X_test.columns, index=self.X_test.index)

def feature_engineering(df): 
    df = engineer_numerical_features(df)
    df = filter_features(df)
    encode_and_scale = EncodingScaling(df)
    encode_and_scale.train_test_split()  
    encode_and_scale.categorical_encoder()
    encode_and_scale.scaling()
    print(encode_and_scale.X_train.head())
    return encode_and_scale.X_train, encode_and_scale.X_test, encode_and_scale.y_train,encode_and_scale.y_test


