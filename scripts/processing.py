import pandas as pd
from datetime import datetime

from scripts.tp_col_groups import TP_REGION_COLS, TP_INJURY_COLS, TP_TYPE_COLS

# Confirmed processing steps
def clean_data(data):
    """
    Cleans the data by removing irrelevant columns and rows with certain negative values.
    Also converts date_of_loss to days_since_loss.
    """

    # Immediate drop
    data = data.drop(['Loss_code', 'Loss_description', 'Capped Incurred'], axis=1)

    # Want to convert date of loss to difference between time and now
    data['date_of_loss'] = pd.to_datetime(data['date_of_loss'])
    data['days_since_loss'] = (datetime.now() - data['date_of_loss']).dt.days
    data = data.drop('date_of_loss', axis=1)

    # Removing rows with negative Notificatio_period and Incurred
    data = data[data['Notification_period'] >= 0]
    data = data[data['Incurred'] >= 0]

    # Present in nearly all cases - doesn't seem relevant when looking at Incurred
    data = data.drop('Vechile_registration_present', axis=1)

    data['Location_of_incident'] = data['Location_of_incident'].apply(lambda x: x if x in ['Minor Road', 'Main Road', 'Car Park', 'Other', 'Home Address', 'Motorway'] else None)
    data['Weather_conditions'] = data['Weather_conditions'].apply(lambda x: x if x in ['NORMAL', 'WET', 'SNOW,ICE,FOG'] else None)
    data['Vehicle_mobile'] = data['Vehicle_mobile'].apply(lambda x: x if x in ['Y', 'N'] else None)
    data['Main_driver'] = data['Main_driver'].apply(lambda x: x if x == 'Y' else 'N')
    data['PH_considered_TP_at_fault'] = data['PH_considered_TP_at_fault'].apply(lambda x: x if x in ['Y', 'N'] else None)

    return data.set_index('Claim Number')


def pre_process_data(data):
    """
    Further preprocessing of the data based on exploration.
    Dropping columns with too many missing values, filling other missing values.
    Feature Engineering
    """

    # Too many missing values
    data = data.copy()
    data.drop(columns=['PH_considered_TP_at_fault'], inplace=True)

    # Filling other missing values with mode
    data['Location_of_incident'] = data['Location_of_incident'].fillna(data['Location_of_incident'].mode()[0])
    data['Weather_conditions'] = data['Weather_conditions'].fillna(data['Weather_conditions'].mode()[0])
    data['Vehicle_mobile'] = data['Vehicle_mobile'].fillna(data['Vehicle_mobile'].mode()[0])

    # Removing 'Incurred' outliers
    data = data[(data['Incurred'] - data['Incurred'].mean()) / data['Incurred'].std() < 2]

    # Changing two cat columns into binary
    data.loc[:, 'Vehicle_mobile'] = data['Vehicle_mobile'].apply(lambda x: 1 if x == 'Y' else 0)
    data.loc[:,'Main_driver'] = data['Main_driver'].apply(lambda x: 1 if x == 'Y' else 0)

    # Convert time to is night
    data.loc[:, 'is_night'] = data['Time_hour'].apply(lambda x: 1 if x >= 20 or x <= 5 else 0)
    data = data.drop(columns=['Time_hour'])

    # Summing TP_TYPE columns
    data['total_from_regions'] = data[TP_TYPE_COLS].sum(axis=1)
    data = data.drop(columns=TP_REGION_COLS)

    data['TP_severe_injuries'] = data['TP_injury_traumatic'] + data['TP_injury_fatality'] + data['TP_injury_whiplash']
    data = data.drop(columns=TP_INJURY_COLS)

    return data