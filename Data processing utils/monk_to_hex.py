import numpy as np
import math


def rgb2lab(rgb):

    def func(t):
        if (t > 0.008856):
            return np.power(t, 1/3.0)
        else:
            return 7.787 * t + 16 / 116.0

    #Conversion Matrix
    matrix = [[0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]]
    
    rgb=np.array([rgb[0]/255,rgb[1]/255,rgb[2]/255])

    cie = np.dot(matrix, rgb)

    cie[0] = cie[0] /0.950456
    cie[2] = cie[2] /1.088754

    # Calculate the L
    L = 116 * np.power(cie[1], 1/3.0) - 16.0 if cie[1] > 0.008856 else 903.3 * cie[1]

    # Calculate the a 
    a = 500*(func(cie[0]) - func(cie[1]))

    # Calculate the b
    b = 200*(func(cie[1]) - func(cie[2]))

    #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100 
    return (L , a, b)

if __name__ == "__main__":

    cols=[
            (246,237,228),
            (243,231,219),
            (247,234,208),
            (243,218,186),
            (215,189,150),
            (160,126,86),
            (130,92,67),
            (96,65,52),
            (58,49,42),
            (41,36,32)
        ]
    
    #convert to Lab
    cols_lab=[]
    for col in cols:
        cols_lab.append(rgb2lab(col))


    #for c in enumerate(cols):
    #    print(f"{1+c[0]} - # {hex(c[1][0])} {hex(c[1][1])} {hex(c[1][2])}")

    avg=[]
    for i in range(1,len(cols_lab)):
        d=math.sqrt(math.pow(cols_lab[i][0]-cols_lab[i-1][0],2)+math.pow(cols_lab[i][1]-cols_lab[i-1][1],2)+math.pow(cols_lab[i][2]-cols_lab[i-1][2],2))
        avg.append(d)
        print(f"D( {i} * {i-1} ): {d}")

    print(f"Average d: {sum(avg)/len(avg)}")

