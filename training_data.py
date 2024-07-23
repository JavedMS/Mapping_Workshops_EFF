import pandas as pd
import re
import numpy as np
import math

# Read all the files
file_path_r = "files/raw"
file_path_m = "files/modified"
file_name = ["res-engelsk.xlsx","res-norsk.xlsx"]
all_data = []

for file in file_name:
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
data.columns = pd.MultiIndex.from_tuples(columns, names = ['Main','Sub'])

def get_transmission():
    counts = data["powerLine"].value_counts() 

    # Get the most chosen category
    obj = counts.idxmax()[0]
    if obj == "overground":
        return "FIXEDOH"
    if obj == "underground":
        return "LOW_OH"
    if obj == "nothing":
        return "DefaultNO"
    return "DefaultYes"

trans = get_transmission()

def get_import():
    counts = data["importOutlook"].value_counts()
    print(counts)

    # Get the most chosen category
    obj = counts.idxmax()[0]
    if obj == "same":
        return 1
    if obj == "more":
        return 2
    if obj == "balanced":
        return 3

import_xxx = get_import()

def get_varnew():
    categories = data["energySource"]

    # Counts for the different categories
    counts = data["energySource"].aggregate(lambda x: x.value_counts())

    solar_bias = 0.5
    offshore_bias = 1

    filtered_rows = counts.loc[2] # Get the row with value 2 (solar)
    
    offshore_counts = counts["offOn"].loc[1]
    offshore_solar_counts = filtered_rows["offOffs"]

    onshore_counts = counts["offOn"].loc[2]
    onshore_solar_counts = filtered_rows["smSms"] + filtered_rows["bBs"] 

    total_solar = (offshore_solar_counts + onshore_solar_counts)*solar_bias
    total_onshore = onshore_counts - onshore_solar_counts*solar_bias
    total_offshore = offshore_counts - offshore_solar_counts*solar_bias
    q_list = [total_solar, total_onshore, total_offshore*offshore_bias, total_offshore*(1-offshore_bias)]

    percentage_list = np.array(q_list) / len(categories.index)
    
    # Test that it sums up to 100% with floating point error
    epsilon = 1e-9
    assert math.isclose(sum(percentage_list), 1, abs_tol=epsilon)

    return percentage_list.tolist()

varnewpcapQ = get_varnew()

def get_corine():
    min_accepted_val = 5
    counts = data["windVisual"].apply(lambda col: (col >= min_accepted_val).sum())

    nr_of_respondents = len(data["windVisual"])
    min_accepted_perc = 0.5


    arr = [1 if count / nr_of_respondents >= min_accepted_perc else 0 for count in counts]

    return arr

corine_onshore = get_corine()

# FIXEDOH 3 [0.24615384615384617, 0.34615384615384615, 0.3261538461538462, 0.08153846153846152] [1, 0, 1, 1, 1, 1, 1, 0, 0]

print("trans", trans)
print("import_xxx", import_xxx)
print("varnewpcapQ", varnewpcapQ)
print("corine_onshore", corine_onshore)