import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np



def get_clean_data():
    data = pd.read_csv("model/data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    return data

def add_sidebar():
    st.sidebar.header("Feature Adjustment")
    data = get_clean_data()

    slider_labels = [
            ("model used", "model")
            ("Feature", "example")
        ]
    input_dict = {}

    for label, key in slider_labels:
        # give mean, max and min
        if label == "Model Used":
            # is string or object feature, so skip it
            input_dict[key] = st.sidebar.selectbox('Model',('Logistic Regression', 'KNN', 'Support Vector Machine'))
        else:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )
    return input_dict
    

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}    
    for key, value in input_dict.items():
        if key == "model":
            pass
        else:
            max_val = X[key].max()
            min_val = X[key].min()
            scaled_value = (value - min_val) / (max_val - min_val)
            scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']


    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    #  Do the same for worst and the tested value

    # show your radar Chart
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig





def add_predictions(input_data):
    model = pickle.load(open("model/model1.pkl", "rb"))
    model2 = pickle.load(open("model/model2.pkl", "rb"))
    model3 = pickle.load(open("model/model3.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    if input_data['opt'] == 'Logistic Regression':
        model = model
    elif input_data['opt'] == 'KNN':
        model = model2
    elif input_data['opt'] == 'Support Vector Machine' :
        model = model3
    del input_data['opt']
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Prediction")
    st.write("The cell cluster is:")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
        
    
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    



def main():
    st.set_page_config(page_title="Breast Cancer Wisconsin (Diagnostic)",
                        page_icon="M", 
                        layout="wide", 
                        initial_sidebar_state="expanded")
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Wisconsin (Diagnostic)")
        st.write("Breast Cancer Diagnosis Prediction is a web application based on machine learning and Streamlit. It allows users to adjust feature values and select from three algorithms (KNN, Logistic Regression, and Support Vector Machine) to predict breast cancer diagnosis. The Radar Chart provides a visual representation of how feature modifications impact predictions")
    col1, col2 = st.columns([4, 1])
    with col1:
        radarChart = get_radar_chart(input_data)
        st.plotly_chart(radarChart)
    with col2:
        add_predictions(input_data)
    with st.container():
        data = get_clean_data()
        st.dataframe(data)
   





if __name__ == '__main__':
    main()
