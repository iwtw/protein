import pandas as pd
import os
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( 'csvfile'  )
    parser.add_argument( 'searchdir' )
    parser.add_argument( 'outfile' )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    df = pd.read_csv( args.csvfile , index_col = 0 )

    a = os.listdir( args.searchdir )

    for idx , v in tqdm(df.iterrows()):
        if idx not in a:
            df = df.drop(idx)

    df.to_csv( args.outfile )

