from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Global variable to store processed data between requests
global_df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    global global_df

    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    # Load the uploaded CSV file
    df = pd.read_csv(file)

    # Normalize column names (remove extra whitespace)
    df.columns = df.columns.str.strip()

    # Check for required columns including 'Region'
    if 'Age' not in df.columns or 'Profit Made' not in df.columns or 'Region' not in df.columns:
        return "CSV must contain 'Age', 'Profit Made', and 'Region' columns.", 400

    # Clean 'Profit Made' column (remove '%' if present)
    df['Profit Made'] = df['Profit Made'].astype(str).str.replace('%', '', regex=False)
    df['Profit Made'] = pd.to_numeric(df['Profit Made'], errors='coerce')

    # Drop rows with missing or non-convertible data
    df = df.dropna(subset=['Age', 'Profit Made', 'Region'])

    # Select only numeric columns for clustering
    clustering_df = df[['Age', 'Profit Made']]

    # Standardize the numeric data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_df)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    # Store the processed dataframe globally for use in /more-details
    global_df = df.copy()

    # Convert DataFrame to HTML table (with Cluster column)
    table = df.to_html(classes='table table-bordered table-striped text-center', index=False)

    return render_template('result.html', table=table)

@app.route('/more-details')
def more_details():
    global global_df

    if global_df is None:
        return "No uploaded data found. Please upload a file first."

    # Calculate total profit by region
    profit_by_region = global_df.groupby('Region')['Profit Made'].sum().to_dict()

    return render_template('more_details.html', profit_by_region=profit_by_region)

if __name__ == '__main__':
    app.run(debug=True)
