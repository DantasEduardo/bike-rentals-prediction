import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(data_path: str) -> None:
    logger.info("Read data...")
    df = pd.read_csv(data_path)[['day','mnth','year','season','holiday', 'weekday','workingday','weathersit','temp',
                                 'atemp','hum','windspeed','rentals']]
    
    x = df[['day','mnth','year','season','holiday','weekday','workingday','weathersit','temp', 'atemp','hum','windspeed']]
    y = df[['rentals']]

    logger.info("Train model...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators= 150, max_depth=10, min_samples_split=20, min_samples_leaf=25)
    model.fit(x_train, y_train)

    logger.info("Evaluate model...")
    y_pred = model.predict(x_test)
    nrmse = np.sqrt(mean_squared_error(y_test, y_pred)) / (np.max(y_train) - np.min(y_train))
    logger.info(f"NRMSE: {nrmse}")

    logger.info("Saving model...")
    joblib.dump(model, 'model.joblib')


if __name__ == '__main__':
    main(
        data_path=r"C:\Users\danta\Desktop\projetos\bike-rentals-prediction\data\bike_rentals.csv"
    )