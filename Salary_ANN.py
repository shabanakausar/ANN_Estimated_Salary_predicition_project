import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

data = pd.read_csv('Churn_Modelling.csv')
data

data = data.drop(['RowNumber','CustomerId','Surname'],axis=1)
data.head()

##Encode categorical variables

label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
data.head()

## One hot encode 'Geography'
from sklearn.preprocessing import OneHotEncoder

onehot_encoder_geo = OneHotEncoder(handle_unknown='ignore')
geo_encoded = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()
geo_encoded

geo_encoded_df = pd.DataFrame(geo_encoded,columns = onehot_encoder_geo.get_feature_names_out(['Geography']))
geo_encoded_df

# Combine one hot encoder columns with the orignal data
data= pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)
data.head()

X = data.drop(['EstimatedSalary'], axis=1)
y = data['EstimatedSalary']

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


## save encoder and scalar
with open('label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(label_encoder_gender, f)

with open('onehot_encoder_geo.pkl', 'wb') as f:
    pickle.dump(onehot_encoder_geo, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

#ANN Regression Problem Implementation

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

# Build the model

model = Sequential({
    Dense(64, activation='relu',input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
})

#compile the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

log_dir = 'regressionlogs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test,y_test),
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping_callback, tensorboard_callback])

#%load_ext tensorboard

#%tensorboard --logdir regressionlogs/fit

model.save('regression_model.keras')


