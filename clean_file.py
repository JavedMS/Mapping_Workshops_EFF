import pandas as pd

# TODO: set index to the id given to the students

# Add to clean_file
file_path_r = "files/raw"
file_path_m = "files/modified"
file_name = ["res-engelsk.xlsx","res-norsk.xlsx"]

for file in file_name:
    data = pd.read_excel(file_path_r + "/" + file)
    data.to_csv(file_path_m + "/" + file[0:-5] + ".csv", index=False)

# Return the unsorted unique column-names 
def get_all_prefixes(df: pd.DataFrame, split_value: str = '.'):
    prefixes = []
    for col in df.columns:
        split = col.rpartition(split_value)
        if (split[1] != split_value):
            # Value was not found
            prefixes.append(split[2])
        else:
            # Value was found
            prefixes.append(split[0])
    # Remove duplicates and return
    return set(prefixes) 

# Insert all the unique column names as value to the key prefix
#TODO
def map_name2prefix(df: pd.DataFrame, prefix: set[str], dict: dict[str:str]):
    for p in prefix:
        return None
    return None

# Set the na to a value
def fill_na(df: pd.DataFrame, val: int):
    df.fillna(val) 

# Combine similar columns into tuples
def col2tuple(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # Find columns that start with the given prefix
    col_to_concat = [col for col in df.columns if col.startswith(prefix)]
    
    if len(col_to_concat) <= 1:
        return df
    
    print(prefix)
    # Concatenate the columns into a tuple
    new_col_name = prefix
    df[prefix] = df[col_to_concat].apply(tuple, axis = 1)
    
    # Remove the new name from the list
    if new_col_name in col_to_concat:
        col_to_concat.remove(new_col_name)

    # Drop the original columns
    df = df.drop(columns=col_to_concat)
    
    return df

# Convert from a tuple to multiple columns
def tuple2col(df: pd.DataFrame, target_col: str, new_cols: list[str]):
    # We need to assert that target_col length is the same as new_cols
    df[new_cols] = pd.DataFrame(df[target_col].tolist(),index = df.index)
    df = df.drop(columns=target_col, axis=1)
 
def test_all_prefixes(prefix: set, df: pd.DataFrame):
    missing_prefix = []
    for p in prefix:
        if p not in df.columns:
            missing_prefix.append(p)
    assert len(missing_prefix) == 0, "Some prefixes have been lost in the dataframe"

def combine_table(df: pd.DataFrame):
    prefixes = get_all_prefixes(df)
    for p in prefixes:
        df = col2tuple(df,p)
    test_all_prefixes(prefixes,df)
    print(len(df.columns))
    return df

data = combine_table(data)
print(len(data.columns))
tuple2col(data,"Hvilke fylker i Norge tror du er best egnet til solkraft?",["A","B","C","D","E","F","G","H","I","J","K"])

print(data.dtypes) # Print the datatypes
print(data.index) # Print the index of the data
print(data.to_numpy()) # Print the data to a numpy object
print(data.describe()) # Describes the data
print(data.head(n=6))