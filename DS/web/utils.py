import pandas as pd
def merge_dataframes(dataframes, tables):
    merged_dataframes = []
    merged_keys ={item:[] for item in dataframes.keys()}
    while True:
        merged = False
        keys = list(dataframes.keys())
        for key in keys:
            df = dataframes[key]
            for col in df.columns:
                if col != "id":
                    col = col.replace("id", "")
                related_key = col
                if related_key not in merged_keys[key] and related_key in tables and related_key in dataframes.keys():
                    related_df = dataframes[related_key]
                    if 'id' in related_df.columns:
                        df[col + 'id'] = df[col + 'id'].astype(str)
                        related_df['id'] = related_df['id'].astype(str)
                        
                        merged_df = pd.merge(df, related_df, how='outer', left_on=col + "id", right_on='id')
                        found = False
                        for i in range(len(merged_dataframes)):
                            if key == merged_dataframes[i]["title"]:
                                merged_dataframes[i]["dataframe"] = merged_df
                                merged_dataframes[i]["key"] += "+"+related_key 
                                found = True
                                break
                        if found!=True: 
                            merged_dataframes.append({"title": key,"dataframe":merged_df, "key":key+"+"+related_key})
                        
                        print(str(key)+" merged with "+related_key)
                        merged_keys[key].append(related_key)
                        dataframes[key] = merged_df
                        merged = True
                        break
            if merged:
                break 
    
        if not merged:  
            break

    return merged_dataframes