import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    """ 
    Apply deterministic binary encoding to 2-category features.

    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings 
    are deterministic and must be consistent between training and serving. 
     
    """
    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset  = set(vals)

    # Deterministic Binary Mapping
    # Critical : These exact mappings are hardcoded in serving pipeline
    # Yes/No mapping (most common pattern in telecom data)

    if valset == {'Yes', 'No'}:
        return s.map({"No" : 0, "Yes" : 1}).astype("Int64")
    
    if valset == {"Female", "Male"}:
        return s.map({'Female' : 0, 'Male' : 1}).astype("Int64")
    
    # Generic Binary Mapping
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0] : 0, sorted_vals[1] : 1}
        return s.astype(str).map(mapping).astype("Int64")
    
    # Non-Binary Features
    return s

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw 
    customer data into ML-ready features. The transformations must be exactly 
    replicated in the serving pipeline to ensure prediction accuracy.
    """
    df = df.copy()
    print(f"🔧 Starting feature engineering on {df.shape[1]} columns...")

    # Step 1: Identify Feature Types
    # Find categorical columns (object type) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include = ["object"]).columns if c != target_col ]
    numeric_cols = df.select_dtypes(include = ['int64', 'float64']).columns.tolist()

    print(f"   📊  Found {len(obj_cols)} categorical and {len(numeric_cols)} numerical columns")

    # Step 2: Split Categorical by Cardinality
    # Binary features (exactly 2 unique values) get binary encoding
    #Multi-category features (>2 unique values) get one-hot encoding
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    print(f"   🔢  Identified {len(binary_cols)} binary and {len(multi_cols)} multi-category features")
    if binary_cols:
        print(f"       Binary features: {binary_cols}")
    if multi_cols:
        print(f"       Multi-category features: {multi_cols}")

    
    # Step 3: Apply Binary Encoding
    # Convert 2-category features to 0/1 using deterministic mappings
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"   ✅  Binary encoded '{c}'  -> binary 0/1(original dtype: {original_dtype})")

    # Step 4: Convert Boolean Columns
    # XGBoost requires integer inputs, not boolean
    bool_cols = df.select_dtypes(include = ['bool']).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   ✅  Converted {len(bool_cols)} boolean columns to integers: {bool_cols}")

    # Step 5: One-Hot Encoding for Multi-Category Features
    #Critical: drop_first = True prevents multicollinearity
    if multi_cols: 
        print(f"    🌟 Applying one-hot encoding to {len(multi_cols)} multi-category features...")
        original_shape = df.shape
        df = pd.get_dummies(df, columns = multi_cols, drop_first = True)

        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"      ✅ Created {new_features} new features from {len(multi_cols)} categorical columns")

    
    # Step 6: Data Type Cleanup
    # Convert nullable integers (Int64) to standard integers for XGBoost
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            #Fill any NaN values with 0 and convert to int 
            df[c] = df[c].fillna(0).astype(int)
    
    print(f"🔧 Feature engineering completed. Final dataset has {df.shape[1]} columns.")
    return df
