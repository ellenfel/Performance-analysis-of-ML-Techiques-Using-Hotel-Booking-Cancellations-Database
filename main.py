import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

pd.set_option('display.max_columns', None)

df = pd.read_csv("data/hotel_bookings.csv")

#df2 = pd.read_csv("data/hotel_booking.csv") personal info

df.head(10)

df.info()

"""

**Data types:**

* Categorical - hotel, is_canceled, customer_type, is_repeated_guest, meal, * 
* country, market_segment, distribution_channel, reserved_room_type, * 
* assigned_room_type, deposit_type, agent, company, reservation_status, *

* Numerical - lead_time, stays_in_weekend_nights, stays_in_week_nights, * 
* adults, children, babies, previous_cancellations, booking_changes, * 
* previous_bookings_not_canceled, days_in_waiting_list, adr, * 
* required_car_parking_spaces, total_of_special_requests, * 

* Ordinal - arrival_date_year, arrival_date_month, arrival_date_week_number, * 
* arrival_date_day_of_month,  reservation_status_date

"""

df.describe()

"""

* The following columns; *
* previous_cancellations, previous_bookings_not_canceled, booking_changes, *
* days_in_waiting_list, required_car_parking_spaces, total_of_special_requests * 
* have only a maximum value. This shows that these features contribute to the *
* decision of cancellation only in very few cases. *

"""

"""
**Assumptions about impact of features:**
* High:
hotel, lead_time, arrival_date_year, arrival_date_month, 
stays_in_weekend_nights, stays_in_week_nights, is_repeated_guest, 
previous_cancellations, previous_bookings_not_canceled, reserved_room_type, 
assigned_room_type, deposit_type, days_in_waiting_list, customer_type

* Medium:
children, babies, distribution_channel, booking_changes, adr

* Low:
arrival_date_week_number, arrival_date_day_of_month,country, meal, adults, 
market_segment, agent, company, required_car_parking_spaces, 
total_of_special_requests, reservation_status, reservation_status_date
"""