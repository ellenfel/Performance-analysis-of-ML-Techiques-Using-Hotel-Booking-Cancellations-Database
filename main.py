# DATA ACQUISITION

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

"""
**Assumptions about cancellation:**
1. The type of hotel decides the cancelation rate with higher cancellations 
in city hotels as compared to resort hotels due to variety of facilities 
available in resort hotels.

2. The earlier the booking made, higher the chances of cancellation.

3. Customers who have bookings for longer durations have lesser chance of 
cancelling their booking. 

4. As more children or babies are involved in the booking, higher chances of 
cancellation.

5. Old guest (is_repeated_guest=1) is less likely to cancel current booking.

6. If there are high previous cancellations, possibility of cancellation of 
current booking is also high.

7. If room assigned is not the same as reserved room type, 
customer might positively cancel the booking.

8. Higher the number of changes made to the booking, lesser is the chance of 
cancellation due to the investment of time in curating the booking as per 
one's requirement.

9. Bookings that are refundable or for which deposits were not made at the 
time of booking stand a high chance of cancelation.

10. If the number of days in waiting list is significant, customer might make 
some other booking due to uncertainty of confirmation of current booking.
"""

"""**Target variable: 
    is_canceled"""
    
    
# EXPLORATORY DATA ANALYSIS

""" ***UNIVARIATE ANALYSIS (Checking the validity of assumptions)***"""
    
is_can = len(df[df['is_canceled']==1])
print("Percentage cancelation= ", is_can/len(df))
df['reservation_status'].value_counts(normalize=True)*100


# highest positive correlations : lead_time followed by previous_cancellations
# highest negative correlations : total_of_special_requests, required_car_parking_spaces
corr= df.corr(method='pearson')['is_canceled'][:]
corr

 
sns.countplot(data=df, x='hotel', hue='is_canceled')
resort_canceled = df[(df['hotel']=='Resort Hotel') & (df['is_canceled']==1)]
city_canceled = df[(df['hotel']=='City Hotel') & (df['is_canceled']==1)]
print('Cancelations in resort hotel= ', (len(resort_canceled))/(len(df[df['hotel']=='Resort Hotel'])))
print('Cancelations in city hotel= ', (len(city_canceled))/(len(df[df['hotel']=='City Hotel'])))   