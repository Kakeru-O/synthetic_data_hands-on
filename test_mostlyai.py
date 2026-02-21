import pandas as pd
from mostlyai.sdk import MostlyAI
import os

def main():
    data_path = "data/raw/bank-marketing.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")
    
    # Use a small sample for faster testing
    df = df.sample(100)
    print(f"Sampled data shape: {df.shape}")
    
    print("\nInitializing MostlyAI in local mode...")
    mostly = MostlyAI(local=True, local_dir="./mostlyai_local")
    
    print("\nStarting training... (this may take a few minutes)")
    columns_config = []
    for col in df.columns:
        if col == "month":
            columns_config.append({'name': 'month', 'model_encoding_type': 'TABULAR_CATEGORICAL'})
        else:
            columns_config.append({'name': col})

    config = {
        'name': 'Bank Marketing Test',
        'tables': [
            {
                'name': 'bank_marketing',
                'data': df,
                'columns': columns_config
            }
        ]
    }
    g = mostly.train(config=config)
    print(f"Training completed. Generator ID: {g.id}")
    
    print("\nGenerating 100 synthetic records...")
    sd = mostly.generate(g, size=100)
    df_synthetic = sd.data()
    
    print("\nGeneration completed. Synthetic data sample:")
    print(df_synthetic.head())
    print(f"\nSynthetic data shape: {df_synthetic.shape}")

if __name__ == "__main__":
    main()
