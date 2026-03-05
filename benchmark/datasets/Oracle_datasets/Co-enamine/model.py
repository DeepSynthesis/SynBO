import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_data():
    """Load the main dataset and descriptor files"""
    # Load main dataset
    df = pd.read_csv("Co-enamine.csv")

    # Load descriptor files
    desc_files = {
        "amine_smiles_ion": "descriptors/amine_ion_desc_dft.csv",
        "amine_smiles_anion": "descriptors/amine_anion_desc_dft.csv",
        "cobalt_smiles": "descriptors/cobalt_desc_dft.csv",
        "oxidant_smiles": "descriptors/oxidant_desc_dft.csv",
        "alkali_smiles": "descriptors/alkali_desc_dft.csv",
        "solvent_smiles": "descriptors/solvent_desc_dft.csv",
    }

    descriptors = {}
    for name, path in desc_files.items():
        descriptors[name] = pd.read_csv(path)

    return df, descriptors


def match_descriptors(df, descriptors):
    """Match descriptors to the main dataset based on SMILES"""
    feature_list = []

    for smiles_col, desc_df in descriptors.items():
        # Create a mapping from SMILES to descriptors
        desc_dict = {}
        for idx, row in desc_df.iterrows():
            smiles = row["SMILES"]
            # Get all descriptor columns (excluding SMILES)
            desc_cols = [col for col in desc_df.columns if col != "SMILES"]
            desc_dict[smiles] = row[desc_cols].values

        # Match descriptors for each sample
        matched_desc = []
        for smiles in df[smiles_col]:
            if smiles in desc_dict:
                matched_desc.append(desc_dict[smiles])
            else:
                # Handle missing descriptors with NaN
                desc_cols = [col for col in desc_df.columns if col != "SMILES"]
                matched_desc.append([np.nan] * len(desc_cols))

        matched_desc = np.array(matched_desc)

        # Add prefix to descriptor names
        desc_cols = [col for col in desc_df.columns if col != "SMILES"]
        prefixed_cols = [f"{smiles_col}_{col}" for col in desc_cols]

        # Create DataFrame for this set of descriptors
        desc_df_matched = pd.DataFrame(matched_desc, columns=prefixed_cols)
        feature_list.append(desc_df_matched)

    # Concatenate all descriptors
    X = pd.concat(feature_list, axis=1)

    # Handle missing values (if any SMILES not found in descriptors)
    X = X.fillna(X.mean())

    return X


def plot_predictions(y_true, y_pred, target_name, r2, mae, rmse):
    """Plot predicted vs true values scatter plot"""
    plt.figure(figsize=(8, 8))
    
    # Create scatter plot
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=60, edgecolor='none')
    
    # Add diagonal line (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add labels and title
    plt.xlabel('True Values', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=14, fontweight='bold')
    plt.title(f'{target_name}: Predicted vs True Values', fontsize=16, fontweight='bold')
    
    # Add metrics text box
    metrics_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend
    plt.legend(loc='lower right', fontsize=11)
    
    # Adjust layout and save
    plt.tight_layout()
    filename = f'{target_name.lower()}_prediction_scatter.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nScatter plot saved as: {filename}")
    plt.close()


def train_and_evaluate(X, y, test_size=0.2, random_state=42, target_name=""):
    """Train LightGBM model with 80:20 train-test split"""
    # Split data into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Initialize LightGBM model with default parameters
    model = LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nTest Set Results:")
    print(f"  R² = {r2:.4f}")
    print(f"  MAE = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    
    # Plot predicted vs true values
    plot_predictions(y_test, y_pred, target_name, r2, mae, rmse)
    
    return {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }


def main():
    """Main function to run LightGBM modeling"""
    print("=" * 60)
    print("LightGBM Modeling for Co-enamine Dataset")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df, descriptors = load_data()
    print(f"Main dataset shape: {df.shape}")
    print(f"Target variables: yield and ee")

    # Match descriptors
    print("\nMatching descriptors...")
    X = match_descriptors(df, descriptors)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")

    # Predict yield
    print("\n" + "=" * 60)
    print("Predicting YIELD")
    print("=" * 60)
    y_yield = df["yield"]
    yield_results = train_and_evaluate(X, y_yield, test_size=0.2, random_state=42, target_name="Yield")

    # Predict ee
    print("\n" + "=" * 60)
    print("Predicting ENANTIOMERIC EXCESS (ee)")
    print("=" * 60)
    y_ee = df["ee"]
    ee_results = train_and_evaluate(X, y_ee, test_size=0.2, random_state=42, target_name="Enantiomeric Excess")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nYield Prediction Results:")
    print(f"  R²: {yield_results['r2']:.4f}")
    print(f"  MAE: {yield_results['mae']:.4f}")
    print(f"  RMSE: {yield_results['rmse']:.4f}")

    print("\nEnantiomeric Excess Prediction Results:")
    print(f"  R²: {ee_results['r2']:.4f}")
    print(f"  MAE: {ee_results['mae']:.4f}")
    print(f"  RMSE: {ee_results['rmse']:.4f}")

    print("\n" + "=" * 60)
    print("Modeling completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()