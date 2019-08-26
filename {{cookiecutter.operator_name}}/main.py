import os
import datetime as dt

import pandas as pd

from fuga.gcs import (
    get_exported_table_df,
    save_df
)


def run():
    train_date = dt.datetime.strptime(
        os.environ['BATCH_DATE'], '%Y-%m-%d')
    dataset_df = get_exported_table_df('my_train_data', date=train_date)
    print(dataset_df.head())

    # Train model
    # model = lgb.train(...

    # Generate predictions
    predictions = dataset_df > 0

    # save it
    save_df(
        predictions,
        name='my_experiment_predictions')


if __name__ == '__main__':
    run()
