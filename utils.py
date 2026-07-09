import pandas as pd
import json



def load_dataset(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        return pd.DataFrame(json.load(uploaded_file))
    
    else:
        raise ValueError(f"Unsupported file type: {name}")
