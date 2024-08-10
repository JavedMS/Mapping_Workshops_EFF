import pandas as pd
import numpy as np
import re
import geopandas

land_path = "files/fylker.geojson"

def read_nettskjema_file(file_names : list[str]) -> pd.DataFrame:
    # Read all the files
    file_path_r = "files/raw"
    all_data = []

    for file in file_names:
        all_data.append(pd.read_excel(file_path_r + "/" + file,index_col=None))

    # Combine all the tables together into one
    data = pd.concat(all_data, axis = 0, ignore_index=True)

    # Rename columns to fit the syntax
    data = data.rename(columns={"$answer_time_ms":"answerTime_ms",
                        "$created":"created"})

    # Remove unecessary column
    data = data.drop('$submission_id',axis=1)

    # Create a multi-level table
    split_symbols = '[._]' # These define a level

    columns = [re.split(split_symbols, c) for c in data.columns]

    # Add a sub-column if the column doesnt contain sub-columns
    columns = [c if len(c) == 2 else [c[0], "sub"] for c in columns]

    data.columns = pd.MultiIndex.from_tuples(columns, names = ['Question','Response'])
    data = clean_energysource(data)

    data.insert(loc = 0, column = ("date", "sub"), value = data[("created", "sub")].dt.date)
    data.drop(columns = [("created", "sub"), ("code", "sub")], inplace = True)

    data = set_dummies(data,"renewableVariance")
    data = set_dummies(data, "importOutlook")
    data = set_dummies(data, "powerLine")

    return data

def get_multicolumn_statistics(data: pd.DataFrame):
    print("Main columns:", data.columns.get_level_values(0).unique().to_list())
    print(data.info())
    print(data.describe())
    data.describe(include = ["object"])

def get_geo(df_responses):
    geo_df = geopandas.read_file(land_path)

    # Insert the areas as a column
    geo_df["area"] = geo_df.area

    n_responses = len(df_responses.index)
    df_responses = df_responses.aggregate(lambda x: (x.value_counts() / n_responses)).T
    df_responses.index.names = ["fylkesnummer"]
    df_responses.columns = ["neg", "pos"]

    # Removed unused columns
    df = geo_df.drop(columns = ["objtype", "navnerom", "versjonid", "datauttaksdato", "opphav", "datafangstdato"])

    combined = pd.merge(df, df_responses, on = "fylkesnummer")
    return combined

def clean_energysource(df: pd.DataFrame):
    df[("energySource","offshore")] = df[("energySource","offOn")].apply(lambda x: 1 if x == 1 else 0)
    df[("energySource","onshore")] = df[("energySource","offOn")].apply(lambda x: 1 if x == 2 else 0)
    df[("energySource", "large_onshore")] = df[("energySource", "bSm")].apply(lambda x: 1 if x == 1 else np.nan if np.isnan(x) else 0)
    df[("energySource","solar")] = df[[("energySource","offOffs"), ("energySource", "smSms"), ("energySource","bBs")]].sum(axis=1).apply(lambda x: 1 if x == 2 else 0)
    df.drop(columns = [("energySource","offOn"), ("energySource","offOffs"),("energySource","bBs"), ("energySource", "bSm"),("energySource", "smSms")], inplace = True)
    return df

def set_dummies(df: pd.DataFrame, column: str):
    dummies = pd.get_dummies(df[(column, "sub")])
    dummies = dummies.astype(int)
    columns = [(column, c) for c in dummies.columns]
    dummies.columns = pd.MultiIndex.from_tuples(columns, names = ['Question','Response'])
    df = pd.concat([df, dummies], axis=1)
    #df.drop(columns = [column, "sub"])
    return df

def rename(df: pd.DataFrame):
    rename_dict = {"overground" : "Overhead", "underground" : "Subsurface", "nothing" : "Same as today"}
    df["Category"].replace(rename_dict, inplace = True)

    rename_dict = {"balanced" : "import = export", "more" : "import more", "same" : "import same"}
    df['Category'].replace(rename_dict, inplace=True)
