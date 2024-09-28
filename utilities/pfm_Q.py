import re
import numpy as np
import sys
import json

def load_pfm(file):
    with open(file, 'rb') as f:
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM files are stored in top-to-bottom order

    return data, scale

def save_pfm(filename, image, scale=1):
    with open(filename, 'wb') as file:
        color = False  # Assuming disparity maps are grayscale
        file.write(b'PF\n' if color else b'Pf\n')
        file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder
        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale
        file.write(b'%f\n' % scale)
        image.tofile(file)

def load_Q_matrix(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        Q_matrix = np.array(data["reprojection-matrix"])

    return Q_matrix