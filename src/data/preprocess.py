import pandas as pd
import numpy as np
from pathlib import Path

def build_mdp():
    print("🔄 Loading raw Parquet data...")
    df = pd.read_parquet("data/raw/mimic_sepsis_final.parquet")
    
    # 1. Discretize Actions (5x5 Grid)
    # Fluids (mL/4h)
    df['fluid_idx'] = pd.cut(df['action_fluid'], 
                             bins=[-1, 0, 50, 500, 1000, np.inf], 
                             labels=[0, 1, 2, 3, 4]).astype(int)
    
    # Vasopressors (mcg/kg/min)
    df['vaso_idx'] = pd.cut(df['action_vaso'], 
                            bins=[-1, 0, 0.05, 0.2, 0.45, np.inf], 
                            labels=[0, 1, 2, 3, 4]).astype(int)
    
    # Combined Action (0-24)
    df['action'] = df['fluid_idx'] * 5 + df['vaso_idx']
    
    # 2. Handle Rewards (Mortality Join)
    # NOTE: You need to pull 'hospital_expire_flag' from BigQuery 'patients' table
    # For now, we use a placeholder. 0 = survived, 1 = died.
    if 'mortality' not in df.columns:
        print("⚠️ Warning: Mortality data missing. Using dummy rewards.")
        df['reward'] = 0 
    
    # 3. Format for train_cql.py
    print("📦 Compressing into NPZ for training...")
    data_dict = {
        'observations': df[['hr', 'map', 'spo2', 'temp']].values.astype(np.float32),
        'actions': df['action'].values.astype(np.int32),
        'rewards': df['reward'].values.astype(np.float32),
        'terminals': (df['window_idx'] == 17).astype(np.int32).values,
        'stay_ids': df['stay_id'].values
    }
    
    out_path = Path("data/processed/mimic_mdp_final.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data_dict)
    print(f"✅ MDP Ready: {out_path}")

if __name__ == "__main__":
    build_mdp()