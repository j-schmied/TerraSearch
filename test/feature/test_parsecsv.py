import numpy as np
import pandas as pd


def parse_plz_csv_file() -> list:

    locations_csv = pd.read_csv('../../resources/plz_de.csv', sep=';', encoding='iso-8859-1')
    locations = np.array(pd.Series(locations_csv['Ort'].values).drop_duplicates())

    return locations


def main():
    locations = parse_plz_csv_file()

    print(locations)
    exit(0)


if __name__ == '__main__':
    main()
