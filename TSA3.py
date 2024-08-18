# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load the data
data = pd.read_excel("testfile.xlsx")

# Preprocess the data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features and target
X = data.drop('Course', axis=1)
y = data['Course']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Function to decode course label
def decode_course_label(encoded_label):
    return label_encoders['Course'].inverse_transform([encoded_label])[0]

# Streamlit app
st.title("Tech Studio Course Prediction App")
st.image("tsa_image.jpeg")

st.sidebar.header("Please take the test")

# Collect user input features into a dataframe
def user_input_features():
    how_good_working_with_numbers = st.sidebar.slider(
        'Do you have the tenacity to solve complex mathematical problems?', 0, 10, 0
    )
    interest_in_cybersecurity = st.sidebar.selectbox(
        "Imagine you work for an IT firm and an unusual activity got detected in their network. Will you be motivated to learn the skills needed to investigate and secure the system against potential attacks/threats?", ['Yes', 'Not really', 'No']
    )
    math_skill = st.sidebar.slider(
        'Think about a time when you had to use mathematical skills to solve a problem or complete a project. How successful were you in applying those skills?', 0, 10, 0
    )
    curious_about_protection = st.sidebar.selectbox(
        'Picture yourself in a company or organization and they need to manage sensitive information. Are you interested are you in exploring ways to ensure that this data remains secure and protected?', ['Yes', 'No']
    )
    studied_numbers_related = st.sidebar.selectbox(
        'Did you study statistics, accounting, or any course related to numbers?', ['Yes', 'No']
    )
    studied_cs_related = st.sidebar.selectbox(
        'Did you study computer science, computer/system engineering, or related?', ['Yes', 'No']
    )
    passion_for_design = st.sidebar.slider(
        'How passionate are you about design?', 0, 10, 0
    )
    communication_skill = st.sidebar.slider(
        'Rate your communication skill', 0, 10, 0
    )
    project_work = st.sidebar.selectbox(
        'Do you prefer working on projects alone or collaborating with others',
        ['prefer working on project alone, then ask questions when I am stuck', 'collaborate with others and do it together']
    )
    basic_sketching_skills = st.sidebar.selectbox(
        'Do you have basic sketching skills?', ['Yes', 'No']
    )
    
    interest_area = st.sidebar.selectbox(
        'Which area interests you the most?', 
        ['Password security', 'Working with numbers', 'Illustrations, animations, and video games', 'Useful platforms, apps, and websites']
    )
    user_empathy = st.sidebar.slider(
        'Rate your user empathy (understanding needs and experiences of users)', 0, 10, 0
    )

    data = {
        'How good are you working with numbers and solving complex mathematical problems': how_good_working_with_numbers,
        'Are you interested in understanding how computer networks and systems are secured from cyber threats?': interest_in_cybersecurity,
        'How good are you in maths': math_skill,
        'Are you curious about how to protect data and information from unauthorized access?': curious_about_protection,
        'Did you study statistics, economics or any related course that deals with numbers': studied_numbers_related,
        'Did you study computer science, software engineering or related': studied_cs_related,
        'How passionate are you  about creating visually appealing designs and user interfaces': passion_for_design,
        'How good are you with communicating ideas or insights to a large audience': communication_skill,
        'Do you prefer working on projects alone or collaborating with others': project_work,
        'Which of these areas captures your interest the most': interest_area,
        'Do you have basic sketching skills to visualise ideas and concepts': basic_sketching_skills,
        'Do you have user empathy, understanding needs and experiences of users (rate)': user_empathy
    }
    features = pd.DataFrame(data, index=[0])
    for column in features.select_dtypes(include=['object']).columns:
        features[column] = label_encoders[column].transform(features[column])
    return features

# Main panel

input_df = user_input_features()


# Add a predict button
if st.button('Predict', key='predict_button'):
    # Make prediction when button is pressed
    prediction = model.predict(input_df)
    predicted_course = decode_course_label(prediction[0])

    # Display the prediction
    st.write('Predicted Course')
    st.subheader(predicted_course)
    feedback_message = f"Based on the prediction, you will likely excel in {predicted_course}."
    st.write(feedback_message)

    # Plotting a radar chart
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    
    # Prepare the radar chart data
    categories = X.columns.tolist()
    values = input_df.values.flatten().tolist()
    max_values = [10] * len(categories)  # assuming a 0-10 scale for sliders
    min_values = [0] * len(categories)

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='User Input'
    ))

    fig.add_trace(go.Scatterpolar(
        r=max_values,
        theta=categories,
        fill='toself',
        name='Max Capability',
        line=dict(color='rgba(0,0,0,0)')  # Hide the line
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True
    )

    st.plotly_chart(fig)

else:
    st.write("Press the Predict button to see the prediction.")