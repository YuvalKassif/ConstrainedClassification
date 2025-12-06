import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load experiment data
from weights_impact import find_experiment_json_files

base_directory = r"savedModels1/normalized_lr/normalized_lr"
all_experiments = find_experiment_json_files(base_directory)

# Prepare data for DataFrame
data = {
    "acc_diff": [],
    "constrained_class": [],
    "constrained_diff": [],
    # "constrained_diff_percent": [],
    "C_k_mean": [],
    "constraint_percent": [],
    "C_k_0": [],
    "C_k_1": [],
    "C_k_2": [],
    "C_k_3": [],
    "C_k_4": []
}

for experiment in all_experiments:
    # Calculate acc_diff
    acc_diff = experiment["results"]["PAO_results"]["accuracy"] - experiment["results"]["PTO_results"]["accuracy"]
    data["acc_diff"].append(acc_diff)

    # Get constrained class and related info
    constrained_class = experiment["config"]["constrained_class_index"]
    constrained_diff = round(experiment["results"]["counts"][str(constrained_class)] - experiment["results"]["N_K"])
    data["constrained_class"].append(constrained_class)
    data["constrained_diff"].append(constrained_diff)

    # Extract C_k and its mean
    C_k = experiment["config"]["C_k"]
    C_k_mean = statistics.mean(C_k)
    data["C_k_mean"].append(C_k_mean)

    # Extract individual C_k values (for more granular analysis)
    for i in range(5):
        data[f"C_k_{i}"].append(C_k[i])

    # Get constraint_percent
    constraint_percent = experiment["config"]["constraints_percent"]
    data["constraint_percent"].append(constraint_percent)

# Create DataFrame
df = pd.DataFrame(data)

# One-hot encode 'constrained_class' column
encoder = OneHotEncoder(sparse=False)
onehot = encoder.fit_transform(df[['constrained_class']])

# Add the one-hot encoded columns to the DataFrame
onehot_df = pd.DataFrame(onehot, columns=[f"constrained_class_{int(i)}" for i in range(onehot.shape[1])])
df = pd.concat([df.drop(columns=["constrained_class"]), onehot_df], axis=1)

# Save the DataFrame to a file or display it
df.to_csv("experiment_data.csv", index=False)
print(df.head())

# ---- Step 2: EDA ---- #

# Visualize the distribution of the target variable
plt.figure()
sns.histplot(df["acc_diff"], bins=20, kde=True)
plt.title("Distribution of acc_diff")
plt.show()

# Plot relationships between acc_diff and some features
plt.figure()
sns.pairplot(df, x_vars=["C_k_mean", "constraint_percent", "constrained_diff"], y_vars="acc_diff", height=5, aspect=0.8)
plt.suptitle("Relationships between features and acc_diff", y=1.02)
plt.show()

# Correlation heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# ---- Step 3: Model Training and Evaluation ---- #

# Prepare features and target
X = df.drop(columns=["acc_diff"])
y = df["acc_diff"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

# ---- Step 4: Display Feature Importance ---- #

# Get feature importances
importances = model.feature_importances_

# Get feature names
feature_names = X.columns

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
