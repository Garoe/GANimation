import numpy as np
import os
from tqdm import tqdm
import argparse
import glob
import re
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-ia', '--input_aus_filesdir', type=str, help='Dir with imgs aus files')
parser.add_argument('-op', '--output_path', type=str, help='Output path')
args = parser.parse_args()

def get_data(filepaths):
    # The first line contains the headers, the second the data
    # In the second line the last au_c_size elements are the indicators of present of AU, and before them there are
    # au_r_size indicators of intensity of the AU
    line_elem = 711
    au_c_size = 18
    au_r_size = 17
    start_idx = line_elem - (au_r_size + au_c_size)
    end_idx = line_elem - au_c_size
    data = dict()
    for filepath in tqdm(filepaths):
        content = np.loadtxt(filepath, delimiter=', ', skiprows=1)
        if content.ndim > 1:  # Found more than one face take only the first one
            content = content[0]
        data[os.path.basename(filepath[:-4])] = content[start_idx:end_idx]

    return data

def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def main():
    filepaths = glob.glob(os.path.join(args.input_aus_filesdir, '*.csv'))
    filepaths.sort()

    # create aus file
    data = get_data(filepaths)

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    save_dict(data, os.path.join(args.output_path, "aus"))


if __name__ == '__main__':
    main()
