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
 

#df['is_canceled'] = df['is_canceled'].astype(str)
sns.countplot(data=df, x='hotel', hue='is_canceled')
resort_canceled = df[(df['hotel']=='Resort Hotel') & (df['is_canceled']==1)]
city_canceled = df[(df['hotel']=='City Hotel') & (df['is_canceled']==1)]
print('Cancelations in resort hotel= ', (len(resort_canceled))/(len(df[df['hotel']=='Resort Hotel'])))
print('Cancelations in city hotel= ', (len(city_canceled))/(len(df[df['hotel']=='City Hotel'])))  
#Our 1st assumption, city hotels have higher cancelation rate than resort hotels, is valid.
# what is coralation?
# cant quite make heads and tails of the above code block


grid = sns.FacetGrid(df, col='is_canceled')
grid.map(plt.hist, 'lead_time', width=50)
grid.add_legend()
"""
Maximum cancelations occur if the booking is made 60-70 days before the checkin 
 date. Longer the lead_time, lower is the cancelation. This invalidates our 
 2nd assumption. 
"""


print(len(df[(df['stays_in_weekend_nights']==0) & (df['stays_in_week_nights']==0)])) 
#715
"""715 bookings don't have both weekday or weekend nights which could be an 
 error in the data as this is not possible in real life scenario. Therefore 
 these rows can be eliminated from the dataset."""


((len(df.loc[(df['children']!=0) | (df['babies']!=0)]))/(len(df))) * 100
"""The number of customers having children or babies or both are only 8% of the
 total population. Therefore this information can be ignored as it will not play 
 a significatn role in deciding whether to cancel the booking or not. 
 Assumption 4 can be discarded."""


sns.countplot(data=df, x='is_repeated_guest', hue='is_canceled')

new_guest = df[(df['is_repeated_guest']==0) & (df['is_canceled']==1)]
old_guest = df[(df['is_repeated_guest']==1) & (df['is_canceled']==1)]
print('Cancelations among new guests= ', (len(new_guest))/(len(df[df['is_repeated_guest']==0])))
print('Cancelations among old guests= ', (len(old_guest))/(len(df[df['is_repeated_guest']==1])))
"""As seen in the correlation table, the above graph bolsters the evidence that
 maximum customers are new comers and they are less likely to cancel their 
 current booking. Old guests are less likely to cancel the booking (14%).
 Assumption 5 holds true."""


sns.countplot(data=df, x='previous_cancellations', hue='is_canceled')
"""Maximum customers have 0 previous cancellations. They are less likely to
 cancel the current booking. However, customers who have cancelled once 
 earlier are more likely to cancel the current booking. This also matches with
 the positive correlation between previous_cancellations and is_cancelled and
 supports Assumption 6.
"""


temp = df.loc[df['reserved_room_type']!=df['assigned_room_type']]
temp['is_canceled'].value_counts(normalize=True)*100
"""Assumption 7 that there more cancellations when assigned room type is 
 different from reserved room type is not valid. There are only 5% 
 cancellations in such a case.
"""
















