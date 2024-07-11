import sqlite3 
import pandas as pd 
import preprocessing
import feature_engineering 
import models

def main():
    conn = sqlite3.connect(r"/Users/kelvinfoo/Desktop/AISG Technical Assignments/Hotel Noshow Prediction/Data/noshow.db")
    df = pd.read_sql_query("SELECT * FROM noshow", conn)
    df = preprocessing.preprocessing(df)  
    X_train, X_test, y_train, y_test = feature_engineering.feature_engineering(df)
    models.run_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()