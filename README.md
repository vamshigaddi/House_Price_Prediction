# Bangalore House Price Prediction Model

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Description](#model-description)
- [Installation](#installation)
- [Usage](#usage)
- [Train Your Own Model](#train-your-own-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project involves developing a machine learning model to predict house prices in Bangalore. By analyzing various features such as location, size, and amenities, the model aims to provide accurate price estimates for residential properties. This tool is useful for real estate agents, potential buyers, and data enthusiasts interested in property price trends in Bangalore.
# DEMO:
https://www.loom.com/share/1c9cebefb83d4453aecc9e410d2b6d33
## Dataset

The dataset used for this model includes several features relevant to house prices in Bangalore, such as:
- Location
- Number of bedrooms (BHK)
- Square footage (total area)
- Number of bathrooms
- Availability (immediate, upcoming, etc.)
- Price

The dataset is preprocessed to handle missing values, outliers, and categorical variables.
Example input CSV format:

| location                | size      | total_sqft | bath | price(Lakhs) |
|-------------------------|-----------|------------|------|-------|
| Electronic City Phase II| 2 BHK     | 1056       | 2    | 39.07 |
| Chikka Tirupathi        | 4 Bedroom | 2600       | 5    | 120   |


## Model Description

The prediction model is built using several machine learning algorithms. The final model is selected based on performance metrics such as RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R² score. The main algorithms considered include:
- Linear Regression
- Decision Trees

After extensive testing and validation, the best-performing model is chosen and fine-tuned for optimal performance.

## Installation

To use this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/vamshigaddi/House_Price_Prediction.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
# Load Pretrained Model
```bash
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_file = 'banglore_home_prices_model.pickle'  # Replace with the path to your trained model file
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Load the columns data (assuming you have the columns saved in a file)
json_file = 'columns.json'  # Replace with the path to your JSON file containing location names
with open(json_file, 'r') as file:
    location_data = json.load(file)


# Extract location names
locations = location_data['data_columns'][3:]  # Offset by 3 to skip first three non-location columns

# Define the prediction function
def predict_price(location, sqft, bath, bhk):
    loc_index = locations.index(location)
    
    # Prepare the input array with zeros
    x = np.zeros(len(locations) + 3)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index + 3] = 1  # Offset by 3 to accommodate the first three columns

    return model.predict([x])[0]

# Example usage
predicted_price = predict_price('1st Phase JP Nagar', 1000, 2, 2)
print(f"Estimated House Price: ₹{predicted_price:.2f} Lakhs")
# output:83.8657025831235 (Lakhs)
```
# or
- Run Streamlit APP
  ```bash
  Streamlit run python.py
  ```
# Train Your Own Model
- Download the Dataset
- Load the Data and Do all the preprocessing
- Select and Build a model
- Train the model
- Inference the values

## Results

The model's performance metrics are :
- accuracy-84%
- MAE-16.55
- R² score-0.862

These metrics are calculated on the test dataset and can be used to evaluate the model's accuracy and reliability.

## Contributing

Contributions to this project are welcome. If you have suggestions for improvements or new features, please submit an issue or a pull request. For major changes, please discuss them with the project maintainers first.

To contribute:

1. Fork the repository.
2. Create a new branch (\`git checkout -b feature-branch\`).
3. Make your changes.
4. Commit your changes (\`git commit -m 'Add some feature'\`).
5. Push to the branch (\`git push origin feature-branch\`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)


