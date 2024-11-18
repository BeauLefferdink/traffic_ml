# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# import CSVs
default_df = pd.read_csv('Traffic_Volume.csv')
sample_df = pd.read_csv('traffic_data_user.csv')


#reading pickle file
model_pickle = open('reg_traffic.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()


# converts month and weekday formats
def convert_month_weekday(df):
    # Define mappings for month and weekday
    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    weekday_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    # Convert 'month' and 'weekday' columns
    df['month'] = df['month'].map(month_mapping)
    df['weekday'] = df['weekday'].map(weekday_mapping)

    return df

# headers 
st.title("Traffic Volume Predictor")
st.write("Utilize our advanced Machine Learning application to predict traffic volume.")
st.image("traffic_image.gif")

alpha = st.slider('Alpha Value for Confidence Interval:', min_value=0.01, max_value=.5, value=.1, step=0.01,)

# sidebar
st.sidebar.image('traffic_sidebar.jpg', 'Traffic Volume Predictor')
st.sidebar.write('You can either upload your data file or manually enter diamond features')
with st.sidebar.expander("Option 1: Upload A CSV", expanded= False):
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])
    st.write("Sample Data Format for Upload")
    st.dataframe(sample_df)
    st.warning('Ensure your data uploaded file has the same column names and data types as shown above.', icon="⚠️")

# Sidebar for user inputs with an expander
with st.sidebar.expander("Option 2: Fill out a Form", expanded=False):
    with st.form("Input Form"):
        # Streamlit Sidebar Form for Traffic Volume Prediction
        st.header("Traffic Volume Prediction Inputs")
        # Input form
        holiday = st.selectbox(
            'Holiday', 
            options=[
                'None', 'Christmas Day', 'Columbus Day', 'Independence Day',
                'Labor Day', 'Martin Luther King Jr Day', 'Memorial Day',
                'New Years Day', 'State Fair', 'Thanksgiving Day',
                'Veterans Day', 'Washingtons Birthday'
            ]
        )
        temp = st.slider('Temperature (Kelvin)', 250.0, 320.0, step=0.1)
        rain_1h = st.slider('Rain in the last hour (mm)', 0.0, 20.0, step=0.1)
        snow_1h = st.slider('Snow in the last hour (mm)', 0.0, 10.0, step=0.1)
        clouds_all = st.slider('Cloud Cover (%)', 0, 100, step=1)
        weather_main = st.selectbox('Weather Condition', options=['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze',
                'Mist', 'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm'])
        month = st.selectbox('Month', options=list(range(1, 13)), format_func=lambda x: f"{x} ({pd.to_datetime(f'2023-{x}-01').strftime('%B')})")
        weekday = st.selectbox('Day of the Week', options=list(range(7)), format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
        hour = st.slider('Hour of the Day', 0, 23, step=1)
        submitted = st.form_submit_button("Predict")


if uploaded_file:
        user_data = pd.read_csv(uploaded_file)
        temp = convert_month_weekday(user_data)
        features = temp[ ['holiday','temp', 'rain_1h', 'snow_1h', 'clouds_all',
                               'weather_main', 'month', 'weekday', 'hour']]
        

        # One-hot encoding to handle categorical variables
        cat_var = ['holiday', 'weather_main']
        df = pd.get_dummies(features, columns = cat_var)



        expected_columns = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'month', 'weekday', 'hour',
       'holiday_Christmas Day', 'holiday_Columbus Day',
       'holiday_Independence Day', 'holiday_Labor Day',
       'holiday_Martin Luther King Jr Day', 'holiday_Memorial Day',
       'holiday_New Years Day', 'holiday_State Fair',
       'holiday_Thanksgiving Day', 'holiday_Veterans Day',
       'holiday_Washingtons Birthday', 'weather_main_Clear',
       'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog',
       'weather_main_Haze', 'weather_main_Mist', 'weather_main_Rain',
       'weather_main_Smoke', 'weather_main_Snow', 'weather_main_Squall',
       'weather_main_Thunderstorm']
        


        
        for col in expected_columns:
           if col not in df.columns:
              df[col] = 0 
        x = df[expected_columns]
      
        # Make batch predictions
        predictions, intervals = reg_model.predict(x, alpha=alpha)


        
        # Process predictions and intervals
        predicted_traffics = predictions.flatten().tolist() 
        lower_limits = intervals[:, 0].clip(min=0).tolist()  
        upper_limits = intervals[:, 1].tolist()             


        # Assign back to the DataFrame
        user_data['Predicted Traffic'] = predicted_traffics
        user_data['Lower Limit'] = lower_limits
        user_data['Upper Limit'] = upper_limits

        user_data['Predicted Traffic'] = user_data['Predicted Traffic'].apply(lambda x: f"{int(x)}")
        user_data['Lower Limit'] = user_data['Lower Limit'].apply(lambda x: f"{int(x[0])}")
        user_data['Upper Limit'] = user_data['Upper Limit'].apply(lambda x: f"{int(x[0])}")

        # Display the updated DataFrame

        st.write(f"## Prediction Results with {100 -alpha * 100:.0f}% Confidence Interval")
        st.dataframe(user_data)
else:
     # Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns=['traffic_volume'])

    # Ensure 'date_time' column is in datetime format
    encode_df['date_time'] = pd.to_datetime(encode_df['date_time'])

    # Extract relevant features
    encode_df['month'] = encode_df['date_time'].dt.month       
    encode_df['weekday'] = encode_df['date_time'].dt.dayofweek 
    encode_df['hour'] = encode_df['date_time'].dt.hour         

    # Drop the original datetime column
    encode_df = encode_df.drop(columns=['date_time'])
    # Combine the list of user data as a row to default_df
    encode_df.loc[len(encode_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all,
                               weather_main, month, weekday, hour]

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)


    expected_columns = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'month', 'weekday', 'hour',
        'holiday_Christmas Day', 'holiday_Columbus Day',
        'holiday_Independence Day', 'holiday_Labor Day',
        'holiday_Martin Luther King Jr Day', 'holiday_Memorial Day',
        'holiday_New Years Day', 'holiday_State Fair',
        'holiday_Thanksgiving Day', 'holiday_Veterans Day',
        'holiday_Washingtons Birthday', 'weather_main_Clear',
        'weather_main_Clouds', 'weather_main_Drizzle', 'weather_main_Fog',
        'weather_main_Haze', 'weather_main_Mist', 'weather_main_Rain',
        'weather_main_Smoke', 'weather_main_Snow', 'weather_main_Squall',
        'weather_main_Thunderstorm']
            


            
    for col in expected_columns:
        if col not in user_encoded_df.columns:
            user_encoded_df[col] = 0 
        x = user_encoded_df[expected_columns]

    # Get the prediction with its intervals

    prediction, intervals = reg_model.predict(x, alpha = alpha)
    pred_value = prediction[0]
    lower_limit = intervals[:, 0]
    upper_limit = intervals[:, 1]

    # Ensure limits are within [0, 1]
    lower_limit = max(0, lower_limit[0][0])
    upper_limit = upper_limit[0][0]


    # Show the prediction on the app
    st.write("## Predicting Traffic Volume...")

    # Display results using metric card
    st.metric(label = "Predicted Traffic Volume", value = f"{pred_value:.0f}")
    st.write("With the given confidence interval:")
    st.write(f"**Confidence Interval**: [{lower_limit:.0f}, {upper_limit:.0f}]")


# Additional tabs for DT model performance
st.subheader("Model Insights")  
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")



