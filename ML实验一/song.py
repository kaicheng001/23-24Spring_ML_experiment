import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

# Load the dataset
data = pd.read_csv('song_data.csv')

# Identifying categorical and numerical columns (excluding 'song_popularity' from numerical)
categorical_cols = data.select_dtypes(include=['object', 'category']).columns.drop('song_name')
numerical_cols = data.select_dtypes(include=[np.number]).columns.drop('song_popularity')

# Dropping 'song_name' as it's irrelevant for modeling
data = data.drop('song_name', axis=1)

# Define transformers for numerical and categorical data
numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Define the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Initially, do not include PCA in the pipeline; it may be added conditionally
    ('linear_regression', LinearRegression())
])

# Splitting the dataset into training and testing sets
X = data.drop('song_popularity', axis=1)
y = data['song_popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Fit the preprocessing to compute VIF on the training set
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=[f"feature_{i}" for i in range(X_train_preprocessed.shape[1])])

# Function to calculate VIF scores
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

vif_scores = calculate_vif(X_train_preprocessed_df)

# Check VIF and conditionally add PCA to the pipeline
if np.any(vif_scores["VIF"] > 5):
    print("Applying PCA due to multicollinearity")
    pipeline.steps.insert(-1, ('pca', PCA(n_components=0.95)))

# Fit the model pipeline, possibly with PCA
pipeline.fit(X_train, y_train)


LR = LinearRegression().fit(X_train, y_train)
train_pred = LR.predict(X_train)
test_pred = LR.predict(X_test)

#Print Models paremeter
print('coeffecients', LR.coef_)
print('intercepts', LR.intercept_)

# Predictions and model evaluation
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)
train_error = mean_squared_error(y_train, y_pred_train)
test_error = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Print metrics
print(f'Training Mean Squared Error: {train_error}')
print(f'Test Mean Squared Error: {test_error}')
print(f'Training R² Score: {train_r2}')
print(f'Test R² Score: {test_r2}')


# Visualizing Actual vs Predicted Popularity for test set

plt.figure(figsize=(10, 6))  # Set the size of the figure
plt.scatter(y_test, y_pred_test, alpha=0.5, label='Predicted', color='orange')  # Increase alpha for more solid color
plt.scatter(y_test, y_test, alpha=0.2, label='Actual', color='blue')  # Plot the actual values for comparison

# Ideal prediction line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Prediction')

# Axes and title
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Actual vs Predicted Popularity')

# Legend
plt.legend(loc='upper left')  # Position the legend in the upper left corner

plt.grid(True)

# Show the plot
plt.show()

# Additional visualization: PCA Component Analysis (Cumulative Explained Variance)
pca = PCA().fit(preprocessor.transform(X_train))
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.text(0.5, 0.9, '95% explained variance', color='red')
plt.show()


continuous_features = [col for col in X.columns if X[col].dtype in ['float64', 'int64']]

# Function to save plots
def save_plots(data, X_train, y_train, model, continuous_features, save_path):
    for feature in continuous_features:
        # Distribution
        plt.figure(figsize=(10, 4))
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        distribution_path = os.path.join(save_path, f'distribution_{feature}.png')
        plt.savefig(distribution_path)
        plt.close()

        # Correlation with target
        plt.figure(figsize=(10, 4))
        sns.scatterplot(x=data[feature], y=data['song_popularity'])
        plt.title(f'{feature} vs. Song Popularity')
        plt.xlabel(feature)
        plt.ylabel('Song Popularity')
        correlation_path = os.path.join(save_path, f'correlation_{feature}.png')
        plt.savefig(correlation_path)
        plt.close()

    # Model visualization for a specific feature
    feature_name = continuous_features[0]
    feature_index = X_train.columns.get_loc(feature_name)

    # Training set scatter plot and regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[feature_name], y_train, color='blue', label='Actual', alpha=0.5)
    predictions = model.predict(X_train)
    plt.scatter(X_train[feature_name], predictions, color='red', label='Predicted', alpha=0.5)
    sorted_idx = np.argsort(X_train[feature_name])
    plt.plot(X_train[feature_name].iloc[sorted_idx], predictions[sorted_idx], color='green', label='Regression Line')
    plt.title(f'Model Predictions vs. Actual for {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Song Popularity')
    plt.legend()
    model_prediction_path = os.path.join(save_path, f'model_prediction_{feature_name}.png')
    plt.savefig(model_prediction_path)
    plt.close()

# save the visiual results
save_plots(data, X_train, y_train, pipeline, continuous_features, "visilual_results")