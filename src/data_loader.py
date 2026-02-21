from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

def download_bank_marketing_data(output_dir="data/raw", filename="bank-marketing.csv"):
    """
    UCI Machine Learning RepositoryからBank Marketingデータセットを取得し、CSVとして保存します。
    """
    print("Downloading Bank Marketing dataset from UCI...")
    
    # fetch dataset 
    bank_marketing = fetch_ucirepo(id=222) 
      
    # data (as pandas dataframes) 
    X = bank_marketing.data.features 
    y = bank_marketing.data.targets 
    
    # Combine features and targets
    df = pd.concat([X, y], axis=1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(df.head())
    
    return output_path

if __name__ == "__main__":
    download_bank_marketing_data()
