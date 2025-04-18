import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import psutil  # New: For system performance metrics

# Load model and helpers
model = joblib.load("animal_disease_model.pkl")
model_columns = joblib.load("model_columns.pkl")
recommendations = joblib.load("recommendations.pkl")
performance_metrics = joblib.load("performance_metrics.pkl")

# Function to get system performance metrics
def get_system_metrics():
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    network = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
    return cpu, memory, disk, network

# Create gauges for system performance
def create_gauge(title, value, min_value, max_value, color):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [min_value, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_value, 50], 'color': 'lightcoral'},
                {'range': [50, 80], 'color': 'gold'},
                {'range': [80, max_value], 'color': 'lightgreen'}
            ]
        }
    ))

# Streamlit UI
st.title("🐾 Animal Disease Predictor")
st.markdown("Enter the symptoms and animal type to predict the disease and recommended action.")

# Symptoms
symptoms = ["fever", "cough", "diarrhea", "vomiting", "lethargy", "rash", "swelling", "loss_of_appetite"]
input_data = {symptom: st.checkbox(symptom.capitalize()) for symptom in symptoms}

# Animal selection
animal = st.selectbox("Select Animal", ["Dog", "Cat", "Cow", "Goat", "Horse", "Chicken"])

# Create input row
animal_features = {f"animal_{animal}": 1}
for a in ["Dog", "Cat", "Cow", "Goat", "Horse", "Chicken"]:
    if f"animal_{a}" not in animal_features:
        animal_features[f"animal_{a}"] = 0

final_input = {**input_data, **animal_features}

# Align columns
input_df = pd.DataFrame([final_input])
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

# Disease Prediction
if st.button("Predict Disease"):
    # Get prediction probabilities
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    # Make a DataFrame of classes and probabilities
    prob_df = pd.DataFrame({
        "Disease": classes,
        "Confidence (%)": (probabilities * 100).round(2)
    }).sort_values(by="Confidence (%)", ascending=False)

    # Top prediction
    top_disease = prob_df.iloc[0]
    recommendation = recommendations.get(top_disease["Disease"], "No recommendation available.")

    # Show prediction and recommendation
    st.success(f"🧪 Predicted Disease: **{top_disease['Disease']}**")
    st.info(f"🎯 Confidence: **{top_disease['Confidence (%)']}%**")
    st.warning(f"💡 Recommended Action: {recommendation}")

    # Visualize top 3 predictions with a bar chart
    st.subheader("🔎 Top 3 Predicted Diseases")
    st.bar_chart(prob_df.head(3).set_index("Disease"))

    # 🎯 Speedometer-style Gauge Chart for Confidence
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=top_disease['Confidence (%)'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': 'lightcoral'},
                {'range': [50, 75], 'color': 'gold'},
                {'range': [75, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': top_disease['Confidence (%)']
            }
        }
    ))
    st.subheader("📈 Confidence Level Gauge")
    st.plotly_chart(fig)

    # 🎯 Confidence Line Chart
    st.subheader("📉 Confidence Line Chart")
    st.line_chart(prob_df.set_index("Disease"))

    # Top 3 Diseases - Scatter plot
    st.subheader("📌 Confidence Comparison Across Top 3 Diseases")
    fig_scatter = px.scatter(prob_df.head(3), x="Disease", y="Confidence (%)", size="Confidence (%)",
                             color="Disease", title="Top 3 Disease Confidence Distribution",
                             hover_name="Disease")
    st.plotly_chart(fig_scatter)

    # 🎯 System Performance Visualization
    st.subheader("📊 System Performance Overview")
    cpu, memory, disk, network = get_system_metrics()

    

    # Display performance metrics
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metric_values = [performance_metrics['accuracy'],
                     performance_metrics['precision'],
                     performance_metrics['recall'],
                     performance_metrics['f1_score']]

    fig_performance = px.bar(x=metric_names, y=[v * 100 for v in metric_values],
                             labels={'x': 'Metric', 'y': 'Value (%)'},
                             color=metric_names,
                             title="Performance Metrics (in %)")
    st.plotly_chart(fig_performance)

    # 📥 Download prediction report
    csv = prob_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name='disease_prediction.csv',
        mime='text/csv'
    )
