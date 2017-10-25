import getopt
import sys
import pandas as pd
from scipy.spatial import cKDTree

def joinByGPS(input_file, output_file, max_distance_in_degree, n_jobs=1):

    print(">>> The process started...")

    # This operation needs to be outside of this scope. It has to be loaded once.
    # Reading open traveling data in csv format
    df_db = pd.read_csv("optd-sample-20161201.csv")
    print(">>> The open traveling data loading")

    # Cleaning missing data
    df_db = df_db.dropna()
    print(">>> Cleaning missing data on open traveling data")

    # Selecting the coordinate over the dataframe
    db_lat_long = df_db[["latitude", "longitude"]]

    # A kd-tree is being created to efficient querying since our data is spatial data
    # In searching/querying, the average time complexity is O(log N)
    # In searching/querying, the worst case time complexity is O(N)
    print(">>> Creating Kd-tree")
    kdtree = cKDTree(db_lat_long.values)
    print(">>> Created Kd-tree")

    # Reading input file and creating a dataframe
    df_input = pd.read_csv(input_file)
    print(">>> The input file read " + input_file)

    # Cleaning missing data
    df_input = df_input.dropna()
    print(">>> Cleaning missing data on the input data")

    # Selecting the coordinate over the dataframe
    input_lat_long = df_input[["geoip_latitude", "geoip_longitude"]]

    # We are applying whole data on the tree. The overall time complexity will be O(M * log N).
    # M denoted the number of rows in the input file, sample_data.csv.
    print(">>> Querying starting")
    d, idx = kdtree.query(input_lat_long.values,
                          k=1,
                          eps=0,
                          p=2,  # Euclidean distance
                          distance_upper_bound=max_distance_in_degree,
                          n_jobs=n_jobs)
    print(">>> Querying finished")


    # Selecting iata_code from the data, optd-sample-20161201.csv
    lhs_df = df_db["iata_code"]

    # Filtering the iata_code by index we collected the previous step
    lhs_df = lhs_df.loc[idx]

    # Filling empty string to get ride of unmatched fields value, NaN.
    n_null = lhs_df.isnull().sum()
    if n_null > 0:
        print (">>> Number of mismatching point : " + str(n_null))
        lhs_df = lhs_df.fillna("")
        print(">>> Mismatching fields replaced with empty string " + str(n_null))

    # After those operations above, we need to rearrange its index to start from 0 to M-1.
    lhs_df = lhs_df.reset_index(drop=True)

    # Selecting uuid from the data sample_data.csv
    rhs_df = df_input['uuid']

    # Creating a new dataframe
    print(">>> Starting concat operations")
    new_df = pd.concat([lhs_df, rhs_df], axis=1)
    print(">>> Completed concat operations")

    # Writing result to CSV file under sample_data directory
    new_df.to_csv(output_file, sep=',', index=False)

    print(">>> Done! please, check out the output file, " + output_file)

    return

def usage():
    print("solution.py -i <input file> -o <output file> -md <max distance in degree> -n <number of jobs>")
    return

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:o:d:j:h', ['input=', 'output=', 'max_distance=', 'n_jobs=', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    if len(opts) != 4:
        usage()
        sys.exit(3)
    input_file = ""
    output = ""
    max_distance = 0.01
    n_jobs = 1
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        elif opt in ('-i', '--input'):
            input_file = arg
        elif opt in ('-o', '--output'):
            output = arg
        elif opt in ('-d', '--max_distance='):
            max_distance = arg
        elif opt in ('-j', '--n_job='):
            n_jobs = arg
        else:
            usage()
            sys.exit(4)

    joinByGPS(input_file, output, float(max_distance), int(n_jobs))

if __name__ == "__main__":
    main()
