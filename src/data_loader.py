import pandas as pd


def load_data(path) -> pd.DataFrame :
    return  pd.read_csv(path)


def data_description(data : pd.DataFrame) :
    print(data.info())
    
