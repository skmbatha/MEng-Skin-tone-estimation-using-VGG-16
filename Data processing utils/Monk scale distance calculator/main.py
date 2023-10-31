import math

colors=[
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

def distance(p1,p2):
    sum_=math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2)+math.pow(p1[2]-p2[2],2)
    return math.sqrt(sum_)

#Calculate distances
for i in range(1,10):
    print(f"Power for {i} & {i-1} : {distance(colors[i],colors[i-1])}")