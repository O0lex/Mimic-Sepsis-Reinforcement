import pandas as pd
from google.cloud import bigquery
from pathlib import Path
import time

def download_mimic_data():
    # 1. Initialize BigQuery Client (Uses your gcloud login)
    project_id = "mimic-sepsis-trial-v0"
    client = bigquery.Client(project=project_id)
    
    # 2. Define your table path
    table_path = f"{project_id}.sepsis_research_v0.mdp_final_v_complete"
    
    print(f"🚀 Starting download from {table_path}...")
    start_time = time.time()

    # 3. Execute the query
    # We select everything from the table you just saved
    query = f"SELECT * FROM `{table_path}` ORDER BY stay_id, window_idx"
    
    try:
        # to_dataframe() will use the high-speed Storage API if pyarrow is installed
        df = client.query(query).to_dataframe()
        
        # 4. Create output directory if it doesn't exist
        out_dir = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. Save as Parquet
        out_file = out_dir / "mimic_sepsis_final.parquet"
        df.to_parquet(out_file, index=False, compression='snappy')
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("-" * 30)
        print(f"✅ Success! Downloaded {len(df):,} rows.")
        print(f"⏱️ Time taken: {duration:.2f} seconds")
        print(f"📂 File saved to: {out_file.absolute()}")
        print("-" * 30)
        print("Next step: scp this file to mimi.cs.mcgill.ca")

    except Exception as e:
        print(f"❌ Error during download: {e}")

if __name__ == "__main__":
    download_mimic_data()