"""
app.py - Streamlit web application for anemia prediction

This script creates a web interface for the anemia prediction model,
allowing users to input patient data and receive a diagnosis prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.feature_config import NORMAL_RANGES, FEATURES
from src.utils.preprocessing import process_user_input
from src.visualization import (
    create_anemia_distribution_pie, create_gender_distribution_plots,
    create_age_distribution_plots, create_hematological_parameter_plots,
    create_correlation_heatmap, create_prediction_gauge,
    create_feature_comparison_radar
)

# Load the model
@st.cache_resource
def load_model(model_path="models/anemia_prediction_model.pkl"):
    """
    Load the anemia prediction model
    
    Args:
        model_path: Path to the model file
        
    Returns:
        model: Loaded model
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure the model is trained and saved properly.")
        return None

# Load the dataset for visualization and statistics
@st.cache_data
def load_data(data_path="data/Anemia Dataset.xlsx"):
    """
    Load the anemia dataset
    
    Args:
        data_path: Path to the dataset
        
    Returns:
        df: Loaded DataFrame
    """
    try:
        df = pd.read_excel(data_path)
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the file path.")
        return None

def make_prediction(user_input, model):
    """
    Make prediction based on user input with enhanced error handling
    
    Args:
        user_input: Dictionary of user input values
        model: Trained model
        
    Returns:
        prediction: Predicted class (0 or 1)
        probability: Prediction probability
        error: Error message (if any)
    """
    try:
        # Process user input
        input_df = process_user_input(user_input)
        
        # Check for required columns
        required_columns = set(['Age', 'Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'Gender_Encoded'])
        missing_columns = required_columns - set(input_df.columns)
        
        if missing_columns:
            return None, None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Input validation
        for feature, value in input_df.iloc[0].items():
            if feature in FEATURES and feature != 'Gender_Encoded':
                min_val = FEATURES[feature].get('min_value')
                max_val = FEATURES[feature].get('max_value')
                
                if min_val is not None and value < min_val:
                    st.warning(f"{feature} value ({value}) is below typical minimum ({min_val})")
                
                if max_val is not None and value > max_val:
                    st.warning(f"{feature} value ({value}) is above typical maximum ({max_val})")
        
        # Get prediction
        prediction = model.predict(input_df)[0]
        
        # Get probability if available
        probability = None
        try:
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_df)[0][1]
            elif hasattr(model, 'named_steps') and 'model' in model.named_steps:
                if hasattr(model.named_steps['model'], 'predict_proba'):
                    probability = model.named_steps['model'].predict_proba(input_df)[0][1]
        except Exception as e:
            return prediction, None, f"Could not calculate probability: {str(e)}"
        
        return prediction, probability, None
        
    except KeyError as e:
        return None, None, f"Column error: {str(e)}"
    except ValueError as e:
        return None, None, f"Value error: {str(e)}"
    except Exception as e:
        return None, None, f"Unexpected error: {str(e)}"

def display_prediction_result(prediction, probability):
    """
    Display the prediction result with appropriate formatting
    
    Args:
        prediction: Predicted class (0 or 1)
        probability: Prediction probability
    """
    if prediction is None:
        return
    
    # Create columns for the result display
    result_col1, result_col2 = st.columns([1, 2])
    
    with result_col1:
        if prediction == 1:
            st.error("Diagnosis: **Anemic**")
        else:
            st.success("Diagnosis: **Non-Anemic**")
        
        if probability is not None:
            st.write(f"Confidence: **{probability:.2%}**")
    
    with result_col2:
        # Display result with gauge chart
        if probability is not None:
            fig = create_prediction_gauge(probability, prediction)
            st.plotly_chart(fig)

def reset_input_values():
    """
    Reset all input values in the session state to their default/empty state.
    
    This function will clear all input fields and set them to a state 
    similar to when a random test data is generated.
    """
    # Define the keys to reset
    reset_keys = [
        'age', 'hb', 'rbc', 'pcv', 'mcv', 'mch', 'mchc', 'gender'
    ]
    
    # Reset each key
    for key in reset_keys:
        # Remove the key from session state if it exists
        if key in st.session_state:
            del st.session_state[key]
    
    # Explicitly set gender to its initial state
    st.session_state['gender'] = 'Select gender'

def validate_input(user_input):
    """
    Validate user input to ensure all required fields are filled.
    
    Args:
        user_input (dict): Dictionary of user input values
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check for missing or zero values
    missing_fields = []
    
    # Check each required field (excluding Gender_Encoded)
    required_fields = ['Age', 'Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC']
    for field in required_fields:
        if field not in user_input or user_input[field] is None or user_input[field] == 0:
            missing_fields.append(field)
    
    # Check gender specifically
    if user_input.get('Gender_Encoded') is None:
        missing_fields.append('Gender')
    
    # Determine validation result
    if missing_fields:
        error_message = "Please fill in all input fields. Missing or invalid values in: " + ", ".join(missing_fields)
        return False, error_message
    
    return True, ""

# Update the display_prediction_interface function
def display_prediction_interface():
    """
    Display the user interface for anemia prediction with enhanced validation
    
    Returns:
        user_input: Dictionary of user input values
        submitted: Boolean indicating if form was submitted
    """
    st.subheader("Anemia Prediction")
    st.write("Enter patient's hematological parameters to predict anemia status.")
    
    # Random test data generation button
    if st.button("Generate Random Test Data", help="Fill the form with random sample data for testing"):
        # Generate realistic test values
        import random
        
        # Set session state values for random data
        st.session_state.gender = random.choice(["Female", "Male"])
        st.session_state.age = random.randint(18, 80)
        
        # Generate realistic values based on gender
        if st.session_state.gender == "Female":
            st.session_state.hb = round(random.uniform(9.0, 15.0), 1)
        else:
            st.session_state.hb = round(random.uniform(10.0, 16.5), 1)
            
        st.session_state.rbc = round(random.uniform(3.5, 5.5), 2)
        st.session_state.pcv = round(random.uniform(30.0, 45.0), 1)
        st.session_state.mcv = round(random.uniform(75.0, 95.0), 1)
        st.session_state.mch = round(random.uniform(25.0, 33.0), 1)
        st.session_state.mchc = round(random.uniform(31.0, 36.0), 1)
        
        st.success("Random test data generated! Click 'Predict Anemia Status' to analyze.")
    
    # Reset button with explicit reset function
    if st.button("Reset Form"):
        reset_input_values()
        st.rerun()  # Rerun the app to refresh the form
    
    # Create form for user input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox(
                "Gender", 
                options=["Select gender","Female", "Male"],
                index=0 if "gender" not in st.session_state else 
                      (0 if st.session_state.gender == "Select gender" else 
                       (1 if st.session_state.gender == "Female" else 2)),
                key="gender",
                help="Select the patient's gender - important as normal ranges differ by gender"
            )
            
            age = st.number_input(
                "Age", 
                min_value=1, 
                max_value=100, 
                value=st.session_state.get("age", None),
                step=1,
                help="Enter patient's age in years - anemia risk and normal ranges can vary by age"
            )
            
            hb = st.number_input(
                "Hemoglobin (Hb) in g/dL", 
                min_value=5.0, 
                max_value=20.0, 
                value=st.session_state.get("hb", None),
                step=0.1,
                help="Hemoglobin is the protein in red blood cells that carries oxygen. Normal: Males 13.5-17.5, Females 12.0-15.5 g/dL"
            )
            
            rbc = st.number_input(
                "Red Blood Cell Count (RBC) in million/μL", 
                min_value=1.0, 
                max_value=8.0, 
                value=st.session_state.get("rbc", None),
                step=0.1,
                help="Total number of red blood cells per microliter. Normal: Males 4.5-5.9, Females 4.0-5.2 million/μL"
            )
        
        with col2:
            pcv = st.number_input(
                "Packed Cell Volume (PCV) in %", 
                min_value=10.0, 
                max_value=60.0, 
                value=st.session_state.get("pcv", None),
                step=0.1,
                help="Percentage of blood volume occupied by red blood cells. Normal: Males 40-52%, Females 37-47%"
            )
            
            mcv = st.number_input(
                "Mean Corpuscular Volume (MCV) in fL", 
                min_value=20.0, 
                max_value=120.0, 
                value=st.session_state.get("mcv", None),
                step=0.1,
                help="Average size of red blood cells. Normal: 80-100 fL. Low = microcytic, High = macrocytic"
            )
            
            mch = st.number_input(
                "Mean Corpuscular Hemoglobin (MCH) in pg", 
                min_value=10.0, 
                max_value=40.0, 
                value=st.session_state.get("mch", None),
                step=0.1,
                help="Average amount of hemoglobin per red blood cell. Normal: 27-33 pg"
            )
            
            mchc = st.number_input(
                "Mean Corpuscular Hemoglobin Concentration (MCHC) in g/dL", 
                min_value=25.0, 
                max_value=40.0, 
                value=st.session_state.get("mchc", None),
                step=0.1,
                help="Average concentration of hemoglobin in a given volume of red blood cells. Normal: 32-36 g/dL"
            )
        
        # Convert gender to binary
        gender_encoded = 1 if gender == "Female" else (0 if gender == "Male" else None)
        
        # Create user input dictionary
        user_input = {
            'Age': age,
            'Hb': hb,
            'RBC': rbc,
            'PCV': pcv,
            'MCV': mcv,
            'MCH': mch,
            'MCHC': mchc,
            'Gender_Encoded': gender_encoded
        }
        
        # Submit button
        submitted = st.form_submit_button("Predict Anemia Status")
        
        # Validation on submission
        if submitted:
            # Validate input
            is_valid, error_message = validate_input(user_input)
            
            if not is_valid:
                # Display warning
                st.warning(error_message)
                submitted = False
    
    return user_input, submitted

def display_exploratory_analysis(df):
    """
    Display exploratory analysis of the dataset
    
    Args:
        df: DataFrame with anemia data
    """
    st.subheader("Exploratory Data Analysis")
    
    # Display data sample
    with st.expander("View dataset sample"):
        st.dataframe(df.head())
    
    # Overall statistics
    st.write("### Dataset Overview")
    overview_col1, overview_col2 = st.columns(2)
    
    with overview_col1:
        st.write(f"Total records: {df.shape[0]}")
        st.write(f"Anemic cases: {df[df['Decision_Class'] == 1].shape[0]} ({df[df['Decision_Class'] == 1].shape[0]/df.shape[0]*100:.1f}%)")
        st.write(f"Non-anemic cases: {df[df['Decision_Class'] == 0].shape[0]} ({df[df['Decision_Class'] == 0].shape[0]/df.shape[0]*100:.1f}%)")
    
    with overview_col2:
        # Create diagnosis pie chart
        fig_diagnosis = create_anemia_distribution_pie(df)
        st.plotly_chart(fig_diagnosis)
    
    # Create copy for visualization
    df_viz = df.copy()
    df_viz['Gender'] = df_viz['Gender'].replace({'f': 'Female', 'm': 'Male'})
    df_viz['Diagnosis'] = df_viz['Decision_Class'].replace({1: 'Anemic', 0: 'Non-Anemic'})
    
    # Gender distribution
    st.write("### Gender Distribution")
    gender_col1, gender_col2 = st.columns(2)
    
    with gender_col1:
        gender_figs = create_gender_distribution_plots(df)
        st.plotly_chart(gender_figs['gender_pie'])
    
    with gender_col2:
        st.plotly_chart(gender_figs['gender_diagnosis_bar'])
    
    # Age distribution
    st.write("### Age Distribution")
    age_col1, age_col2 = st.columns(2)
    
    with age_col1:
        age_figs = create_age_distribution_plots(df)
        st.plotly_chart(age_figs['age_histogram'])
    
    with age_col2:
        st.plotly_chart(age_figs['age_anemia_bar'])
    
    # Hematological parameters
    st.write("### Hematological Parameters")
    
    # Let user select which parameter to visualize
    param_options = ['Hb', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC']
    selected_param = st.selectbox("Select hematological parameter to visualize:", param_options)
    
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        param_figs = create_hematological_parameter_plots(df, selected_param)
        st.plotly_chart(param_figs['histogram'])
    
    with param_col2:
        st.plotly_chart(param_figs['boxplot'])
    
    # Display normal ranges and statistics
    st.write(f"### {selected_param} Analysis")
    unit = FEATURES[selected_param]['unit'] if selected_param in FEATURES else ''
    
    st.write(f"**Normal Range for {selected_param}:**")
    
    if selected_param in NORMAL_RANGES:
        if 'all' in NORMAL_RANGES[selected_param]:
            min_val, max_val = NORMAL_RANGES[selected_param]['all']
            st.write(f"* General: {min_val} - {max_val} {unit}")
        else:
            for gender, range_vals in NORMAL_RANGES[selected_param].items():
                if gender != 'unit':
                    min_val, max_val = range_vals
                    st.write(f"* {gender.upper()}: {min_val} - {max_val} {unit}")
    
    # Statistics by gender and diagnosis
    st.write(f"**{selected_param} Statistics by Gender and Diagnosis:**")
    stats_df = df_viz.groupby(['Gender', 'Diagnosis'])[selected_param].agg(['mean', 'std', 'min', 'max']).round(2)
    st.dataframe(stats_df)
    
    # Correlation heatmap
    st.write("### Correlation Between Parameters")
    
    corr_fig = create_correlation_heatmap(df)
    st.plotly_chart(corr_fig)
    
    # Research findings from the paper
    st.write("### Research Findings")
    st.info("""
    According to the research by Mojumdar et al. (2025), the Chi-square test yielded a p-value of 4.1929 × 10^-29, 
    indicating no significant association between gender and diagnostic outcomes. However, both Z-test and T-test 
    revealed significant gender differences in hemoglobin levels, with p-values of 3.4789 × 10^-33 and 4.1586 × 10^-24, 
    respectively. This underscores the importance of gender when analyzing hemoglobin variations in anemia diagnosis.
    """)

def display_interpretation_guide():
    """Display guide for interpreting hematological parameters"""
    st.subheader("Interpretation Guide")
    
    # Create tabs for different parameters
    tabs = st.tabs(["Hemoglobin", "RBC", "PCV", "MCV", "MCH", "MCHC"])
    
    with tabs[0]:
        st.write("### Hemoglobin (Hb)")
        st.write("""
        **What it is:** Hemoglobin is the protein in red blood cells that carries oxygen.
        
        **Normal ranges:**
        - Adult males: 13.5 to 17.5 g/dL
        - Adult females: 12.0 to 15.5 g/dL
        
        **In anemia:**
        - Low hemoglobin levels are the primary indicator of anemia
        - Values below normal range suggest insufficient oxygen-carrying capacity
        
        **Clinical significance:**
        - Severe anemia: Hb < 8 g/dL
        - Moderate anemia: Hb 8-10 g/dL
        - Mild anemia: Hb 10-12 g/dL (females) or 10-13 g/dL (males)
        """)
    
    with tabs[1]:
        st.write("### Red Blood Cell Count (RBC)")
        st.write("""
        **What it is:** The total number of red blood cells per volume of blood.
        
        **Normal ranges:**
        - Adult males: 4.5 to 5.9 million cells/μL
        - Adult females: 4.0 to 5.2 million cells/μL
        
        **In anemia:**
        - Low RBC count often accompanies low hemoglobin
        - Can help distinguish between different types of anemia
        
        **Clinical significance:**
        - Low RBC with normal MCV: Normocytic anemia (e.g., chronic disease, acute blood loss)
        - Low RBC with high MCV: Macrocytic anemia (e.g., vitamin B12 or folate deficiency)
        - Low RBC with low MCV: Microcytic anemia (e.g., iron deficiency, thalassemia)
        """)
    
    with tabs[2]:
        st.write("### Packed Cell Volume (PCV) / Hematocrit")
        st.write("""
        **What it is:** The percentage of blood volume that consists of red blood cells.
        
        **Normal ranges:**
        - Adult males: 40% to 52%
        - Adult females: 37% to 47%
        
        **In anemia:**
        - Reduced PCV indicates decreased red cell mass
        - Generally follows hemoglobin trends
        
        **Clinical significance:**
        - Used to monitor hydration status along with anemia
        - Helps assess response to treatment
        - Important for determining severity of anemia
        """)
    
    with tabs[3]:
        st.write("### Mean Corpuscular Volume (MCV)")
        st.write("""
        **What it is:** The average size of red blood cells.
        
        **Normal range:** 80 to 100 femtoliters (fL)
        
        **In anemia:**
        - Low MCV (<80 fL): Microcytic anemia (small RBCs)
        - Normal MCV (80-100 fL): Normocytic anemia (normal-sized RBCs)
        - High MCV (>100 fL): Macrocytic anemia (large RBCs)
        
        **Clinical significance:**
        - Critical for classifying anemia type:
          * Microcytic: Often iron deficiency, thalassemia
          * Normocytic: Chronic disease, acute blood loss, kidney disease
          * Macrocytic: Vitamin B12/folate deficiency, liver disease, alcoholism
        """)
    
    with tabs[4]:
        st.write("### Mean Corpuscular Hemoglobin (MCH)")
        st.write("""
        **What it is:** The average amount of hemoglobin per red blood cell.
        
        **Normal range:** 27 to 33 picograms (pg)
        
        **In anemia:**
        - Low MCH: Hypochromic anemia (less hemoglobin per cell)
        - Normal/high MCH: Normochromic or hyperchromic anemia
        
        **Clinical significance:**
        - Often correlates with MCV
        - Low in iron deficiency anemia and thalassemia
        - Helps distinguish between different causes of microcytic anemia
        """)
    
    with tabs[5]:
        st.write("### Mean Corpuscular Hemoglobin Concentration (MCHC)")
        st.write("""
        **What it is:** The average concentration of hemoglobin in a given volume of red blood cells.
        
        **Normal range:** 32 to 36 g/dL
        
        **In anemia:**
        - Low MCHC: Hypochromic anemia (less hemoglobin concentration)
        - Normal MCHC: Normochromic anemia
        
        **Clinical significance:**
        - Provides information about hemoglobin synthesis
        - Decreased in iron deficiency anemia
        - Helps in differential diagnosis of microcytic anemias
        """)

def display_clinical_interpretation(prediction, user_input):
    """
    Display clinical interpretation of the prediction result
    
    Args:
        prediction: Predicted class (0 or 1)
        user_input: Dictionary of user input values
    """
    st.subheader("Clinical Interpretation")
    
    # Count abnormal parameters
    abnormal_params = 0
    gender = 'Female' if user_input['Gender_Encoded'] == 1 else 'Male'
    gender_code = 'f' if gender == 'Female' else 'm'
    
    for param, ranges in NORMAL_RANGES.items():
        if param in user_input:
            if 'all' in ranges:
                min_val, max_val = ranges['all']
            else:
                min_val, max_val = ranges[gender_code]
                
            if user_input[param] < min_val or user_input[param] > max_val:
                abnormal_params += 1
    
    # Basic interpretation based on key parameters
    if prediction == 1:  # If anemia is predicted
        st.write("### Suggested Anemia Classification:")
        
        # MCV-based classification
        if user_input['MCV'] < 80:
            anemia_type = "Microcytic Anemia"
            st.write(f"**{anemia_type}** (MCV < 80 fL)")
            st.write("""
            **Possible causes:**
            - Iron deficiency anemia
            - Thalassemia
            - Anemia of chronic disease (some cases)
            - Lead poisoning
            
            **Recommended additional tests:**
            - Serum ferritin
            - Iron studies (serum iron, TIBC, transferrin saturation)
            - Hemoglobin electrophoresis (for thalassemia)
            """)
            
        elif user_input['MCV'] > 100:
            anemia_type = "Macrocytic Anemia"
            st.write(f"**{anemia_type}** (MCV > 100 fL)")
            st.write("""
            **Possible causes:**
            - Vitamin B12 deficiency
            - Folate deficiency
            - Liver disease
            - Alcoholism
            - Myelodysplastic syndromes
            
            **Recommended additional tests:**
            - Serum B12 and folate levels
            - Liver function tests
            - Reticulocyte count
            """)
            
        else:
            anemia_type = "Normocytic Anemia"
            st.write(f"**{anemia_type}** (MCV 80-100 fL)")
            st.write("""
            **Possible causes:**
            - Anemia of chronic disease
            - Kidney disease
            - Hemolytic anemia
            - Acute blood loss
            - Mixed nutritional deficiencies
            
            **Recommended additional tests:**
            - CRP and ESR (inflammation markers)
            - Kidney function tests
            - Reticulocyte count
            - Bilirubin (for hemolysis)
            """)
        
        # Check for hypochromic anemia
        if user_input['MCHC'] < 32:
            st.write("**Features of Hypochromic Anemia** (MCHC < 32 g/dL)")
            
        # Severity assessment based on hemoglobin
        if gender == 'Female':
            if user_input['Hb'] < 8:
                severity = "Severe"
            elif user_input['Hb'] < 10:
                severity = "Moderate"
            else:
                severity = "Mild"
        else:  # Male
            if user_input['Hb'] < 8:
                severity = "Severe"
            elif user_input['Hb'] < 11:
                severity = "Moderate"
            else:
                severity = "Mild"
                
        st.write(f"**Severity: {severity}** based on hemoglobin level")
        
    else:  # If not anemic
        if abnormal_params > 0:
            st.write(f"""
            The analysis suggests a **non-anemic** status, despite {abnormal_params} 
            parameter(s) outside the reference range. This could indicate:
            
            - Early changes not yet manifesting as clinical anemia
            - Compensated anemia
            - Recent recovery from anemia
            - Individual variations in baseline values
            
            Consider monitoring if other clinical symptoms are present.
            """)
        else:
            st.write("""
            All hematological parameters are within normal range, consistent with the 
            **non-anemic** prediction. No further hematological evaluation is indicated 
            based on these results alone.
            """)

def display_reference_table(user_input):
    """
    Display reference ranges table with user input values
    
    Args:
        user_input: Dictionary of user input values
    """
    st.subheader("Parameter Reference Ranges")
    
    # Define reference ranges
    reference_df = []
    gender = 'Female' if user_input['Gender_Encoded'] == 1 else 'Male'
    gender_code = 'f' if gender == 'Female' else 'm'
    
    for param, config in FEATURES.items():
        if param != 'Gender' and param != 'Gender_Encoded' and param in user_input:
            unit = config.get('unit', '')
            
            if param in NORMAL_RANGES:
                if 'all' in NORMAL_RANGES[param]:
                    reference_min, reference_max = NORMAL_RANGES[param]['all']
                    gender_specific = 'No'
                else:
                    reference_min, reference_max = NORMAL_RANGES[param][gender_code]
                    gender_specific = 'Yes'
                
                status = 'Normal'
                if user_input[param] < reference_min:
                    status = 'Low'
                elif user_input[param] > reference_max:
                    status = 'High'
                
                reference_df.append({
                    'Parameter': param,
                    'User Value': f"{user_input[param]:.1f} {unit}",
                    'Reference Range': f"{reference_min:.1f} - {reference_max:.1f} {unit}",
                    'Gender Specific': gender_specific,
                    'Status': status
                })
    
    # Create DataFrame
    if reference_df:
        reference_df = pd.DataFrame(reference_df)
        
        # Apply color styling based on status
        def color_status(val):
            if val == 'Low':
                return 'background-color: #8B0000'  # Red
            elif val == 'High':
                return 'background-color: #000080'  # Blue
            else:
                return 'background-color: #008000'  # Green
        
        st.dataframe(reference_df.style.applymap(color_status, subset=['Status']))
    else:
        st.write("No reference ranges available for the provided parameters.")

def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="Anemia Prediction App",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and introduction
    st.title("🩸 Anemia Diagnostic Assistant")
    
    
    # Sidebar navigation
    st.sidebar.title("Menu")
    app_mode = st.sidebar.radio("Go to", ["About", "Prediction Tool", "Exploratory Analysis", "Interpretation Guide"])
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    if app_mode == "About":
        st.header("Introduction")
        st.write("""
        This project implements a web application for predicting anemia risk using machine learning. The app allows users to input their hematological parameters and receive an anemia diagnosis based on a trained predictive model. 
        It leverages Streamlit for an interactive interface and uses advanced machine learning techniques to assess anemia status.


        This project builds upon the foundational work of [Mojumdar et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11699093/), which
        used traditional statistical methods like Chi-Square tests and T-tests to explore associations between biological factors and anemia.
        Building on these insights, our goal is to apply machine learning techniques to uncover more complex, non-linear relationships in
        the data. By doing so, we aim to enhance diagnostic accuracy and provide deeper insights into the factors influencing anemia.
   

        What makes this project particularly unique is the collaboration with [Dr. Gem Wu](https://scholar.google.com.tw/citations?user=MwIr5fMAAAAJ&hl=en),
        a **hematologist** working at **Chang Gung Memorial Hospital**, Taiwan. With [Dr. Gem Wu](https://scholar.google.com.tw/citations?user=MwIr5fMAAAAJ&hl=en) providing expert support on the hematological aspects of anemia, 
        we have been able to incorporate expert insights into the hematological aspects of anemia, 
        ensuring that our analysis is grounded in medical realities and clinical perspectives. 
        This collaboration enhances the accuracy and relevance of our findings, bridging the gap between data science and clinical expertise.
        """)    
        st.markdown("---")
        st.header("About Anemia")
        st.write("""
        Anemia is a condition characterized by a lack of healthy red blood cells or hemoglobin to carry
        adequate oxygen to the body's tissues. It can cause fatigue, weakness, pale skin, and shortness
        of breath. Early detection is crucial for effective treatment and management.
        
        This tool uses machine learning to predict anemia based on:
        * Demographic information (age, gender)
        * Hematological parameters (Hb, RBC, PCV, MCV, MCH, MCHC)
        
        The model was trained on data from a study conducted at [Aalok Healthcare Ltd., Bangladesh](http://data.mendeley.com/datasets/y7v7ff3wpj/1),
        featuring comprehensive hematological profiles of patients.
        """)
        st.markdown("---")
        st.header("References")
        st.write("""
        - Paper: Mojumdar et al., "AnaDetect: An extensive dataset for advancing anemia detection, diagnostic methods, and predictive analytics in healthcare", PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC11699093/)
        - Anemia: Approach and Evaluation (https://manualofmedicine.com/topics/hematology-oncology/anemia-approach-and-evaluation/)
        - Source Code (Anemia Detection with Machine Learning): "Anemia Detection with Machine Learning", GitHub repository (https://github.com/maladeep/anemia-detection-with-machine-learning)
        - Source Code (Anemia Prediction): "Anemia Prediction", GitHub repository (https://github.com/muscak/anemia-prediction)
        """)
        
    elif app_mode == "Prediction Tool":
        if model is None or df is None:
            st.error("Could not load model or dataset. Please check the files.")
        else:
            # Display prediction interface
            user_input, submitted = display_prediction_interface()
            
            # Make prediction if submitted
            if submitted:
                with st.spinner("Analyzing hematological parameters..."):
                    prediction, probability, error = make_prediction(user_input, model)
                    # Display result
                    display_prediction_result(prediction, probability)
                        
                        # Display reference ranges table
                    display_reference_table(user_input)
                        
                        # Display clinical interpretation
                    display_clinical_interpretation(prediction, user_input)
                        
                        # Add feature comparison radar chart
                    st.subheader("Parameter Comparison with Normal Ranges")
                    radar_fig = create_feature_comparison_radar(user_input, NORMAL_RANGES)
                    st.plotly_chart(radar_fig)
    
    elif app_mode == "Exploratory Analysis":
        if df is None:
            st.error("Could not load dataset. Please check the file path.")
        else:
            display_exploratory_analysis(df)
    
    elif app_mode == "Interpretation Guide":
        display_interpretation_guide()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this app**
    
    This app was developed to demonstrate the application of machine learning in anemia diagnosis.
    
    Data source: [Mojumdar et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11699093/)
    
    **Note:** This app is for educational purposes only and should not replace professional medical advice.
    """)

if __name__ == "__main__":
    main()
