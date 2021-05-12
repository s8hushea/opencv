import matplotlib.pyplot as plt
import numpy as np
import pylab

'''#Delta X
#DeltaRowNs = [Blue Row N, Red Row N, Yellow Row N, Green Row N, Orange Row N, White Row N, LGreen Row N, LBlue Row N]
#Delta X
#DeltaRowNs = [Blue Row N, Red Row N, Yellow Row N, Green Row N, Orange Row N, White Row N]
DeltaXRowOnes = [[-140, -48, 3.87], [-64, -47, 1.66], [10, -47, 0.01], [87, -47, -3.11], [164, -47, -6.06],
                [-140, -48, 7.77], [-64, -48, 3.3], [12, -47, -1.8], [87, -46, -5.7], [164, -46, -11.09],
                [-140, -48, 7.47], [-63, -47, 2.54], [12, -47, -2.08], [88, -47, -4.94], [165, -46, -10.63],
                [-140, -48, 4.93], [-63, -48, 0.49], [12, -48, -0.85], [87, -46, -2.95], [164, -46, -6.51],
                [-140, -48, 6.81], [-64, -48, 3.16], [11, -47, -1.59], [87, -46, -4.14], [164, -47, -8.25],
                [-139, -48, 4], [-64, -48, 1.55], [11, -47, -0.18], [88, -46, -3.34], [165, -46, -7.91]
                ]
DeltaXRowTwos = [[-151, 14, 2.47], [-68, 14, 0.72], [11, 15, -0.58], [93, 15, -3.01], [176, 16, -8.77],
                [-150, 14, 7.67], [-68, 15, 2.34], [12, 15, -2.38], [94, 16, -8.63], [176, 16, -13.98],
                [-150, 14, 5.46], [-69, 15, 2.26], [12, 16, -2.53], [94, 16, -6.4], [175, 16, -10.74],
                [-150, 14, 1.79], [-69, 14, 1.55], [12, 15, -1.94], [94, 14, -5.44], [174, 16, -6.25],
                [-151, 14, 5.82], [-69, 14, 1.57], [12, 15, -1.77], [93, 15, -5.67], [175, 16, -10.34],
                [-151, 13, 4.71], [-69, 15, 1.82], [12, 15, -2.06], [93, 15, -4.11], [175, 15, -8.81]
                ]
DeltaXRowThrees = [[-164, 86, 2.18], [-75, 87, 2.46], [11, 88, -1.43], [98, 87, -2.17], [188, 89, -7.99],
                  [-164, 85, 7.57], [-74, 88, 2.36], [12, 88, -3.27], [100, 88, -7.76], [189, 89, -15.07],
                  [-165, 87, 6.57], [-74, 87, 0.17], [12, 89, -3.28], [101, 90, -7.24], [188, 90, -11.98],
                  [-164, 86, 4.01], [-75, 88, 1.02], [12, 88, -3.69], [99, 89, -3.93], [189, 90, -9.69],
                  [-165, 87, 6.8], [-74, 87, 1.72], [11, 88, -1.65], [100, 89, -6.09], [189, 90, -11.78],
                  [-165, 87, 5.71], [-74, 87, 0.41], [11, 88, -0.85], [100, 89, -4.27], [189, 90, -10.08]
                ]

#Delta Y
#DeltaRowNs = [Blue Row N, Red Row N, Yellow Row N, Green Row N, Orange Row N, White Row N]
DeltaYRowOnes = [[-140, -48, 0.61], [-64, -47, 1.1], [10, -47, 2.44], [87, -47, 4.18], [164, -47, 5.52],
                [-140, -48, 2.57], [-64, -48, 4.1], [12, -47, 4.24], [87, -46, 4.42], [164, -46, 5.98],
                [-140, -48, 2.06], [-63, -47, 2.76], [12, -47, 3.19], [88, -47, 4.39], [165, -46, 5.37],
                [-140, -48, 0.53], [-63, -48, 2.23], [12, -48, 4.44], [87, -46, 2.9], [164, -46, 5.25],
                [-140, -48, 2.17], [-64, -48, 3.25], [11, -47, 3.94], [87, -46, 3.84], [164, -47, 6.25],
                [-139, -48, 0.93], [-64, -48, 2.9], [11, -47, 2.22], [88, -46, 2.43], [165, -46, 4.59]
                ]
DeltaYRowTwos = [[-151, 14, -1.35], [-68, 14, -0.5], [11, 15, 0.42], [93, 15, 1.55], [176, 16, 1.35],
                [-150, 14, -2.78], [-68, 15, -2.16], [12, 15, -0.65], [94, 16, 0.01], [176, 16, 1.48],
                [-150, 14, -1.93], [-69, 15, -1.84], [12, 16, -0.82], [94, 16, 0.16], [175, 16, 0.97],
                [-150, 14, -2.17], [-69, 14, -0.98], [12, 15, 0.08], [94, 14, 1.99], [174, 16, 2.09],
                [-151, 14, -2.17], [-69, 14, -1], [12, 15, -0.9], [93, 15, 0.51], [175, 16, 1.35],
                [-151, 13, -1.53], [-69, 15, -1.34], [12, 15, -0.17], [93, 15, 1.01], [175, 15, 2.82]
                ]
DeltaYRowThrees = [[-164, 86, -1.36], [-75, 87, -3.31], [11, 88, -3.1], [98, 87, -0.085], [188, 89, -0.53],
                  [-164, 85, -5.95], [-74, 88, -7.56], [12, 88, -5.65], [100, 88, -3.95], [189, 89, -2.9],
                  [-165, 87, -6.79], [-74, 87, -5.73], [12, 89, -5.81], [101, 90, -3.94], [188, 90, -3.6],
                  [-164, 86, -4.79], [-75, 88, -4.89], [12, 88, -3.15], [99, 89, -2.04], [189, 90, -1.21],
                  [-165, 87, -6.29], [-74, 87, -5.4], [11, 88, -4.34], [100, 89, -3.13], [189, 90, -2.77],
                  [-165, 87, -6.7], [-74, 87, -4.54], [11, 88, -2.93], [100, 89, -2.26], [189, 90, -1.84]
                ]

#Delta Z
#DeltaRowNs = [Blue Row N, Red Row N, Yellow Row N, Green Row N, Orange Row N, White Row N]
DeltaZRowOnes = [[-140, -48, -16.87], [-64, -47, -14.54], [10, -47, -19.4], [87, -47, -20.03], [164, -47, -15.64],
                [-140, -48, -33.78], [-64, -48, -35.13], [12, -47, -38.71], [87, -46, -32.05], [164, -46, -32.95],
                [-140, -48, -32.07], [-63, -47, -33.4], [12, -47, -28.64], [88, -47, -27.86], [165, -46, -29.28],
                [-140, -48, -17.64], [-63, -48, -16.56], [12, -48, -21.23], [87, -46, -16.47], [164, -46, -16.29],
                [-140, -48, -28.42], [-64, -48, -26.57], [11, -47, -28.82], [87, -46, -28.11], [164, -47, -26.25],
                [-139, -48, -19.44], [-64, -48, -20.99], [11, -47, -17.05], [88, -46, -16.38], [165, -46, -20.14]
                ]
DeltaZRowTwos = [[-151, 14, -14.08], [-68, 14, -16.97], [11, 15, -17.66], [93, 15, -14.975], [176, 16, -18.25],
                [-150, 14, -33.39], [-68, 15, -35.78], [12, 15, -41.31], [94, 16, -38.1], [176, 16, -34.95],
                [-150, 14, -30.62], [-69, 15, -29.8], [12, 16, -33.97], [94, 16, -31.21], [175, 16, -28.59],
                [-150, 14, -14.68], [-69, 14, -16.37], [12, 15, -22], [94, 14, -17.19], [174, 16, -17.75],
                [-151, 14, -26.76], [-69, 14, -25.99], [12, 15, -29.2], [93, 15, -28.57], [175, 16, -26.5],
                [-151, 13, -20.49], [-69, 15, -20.23], [12, 15, -22.44], [93, 15, -20.37], [175, 15, -21.89]
                ]
DeltaZRowThrees = [[-164, 86, -20.22], [-75, 87, -16.77], [11, 88, -16.71], [98, 87, -16.59], [188, 89, -15.895],
                  [-164, 85, -35.69], [-74, 88, -36.34], [12, 88, -36.9], [100, 88, -33.93], [189, 89, -34.58],
                  [-165, 87, -28.33], [-74, 87, -31.22], [12, 89, -34.48], [101, 90, -29.34], [188, 90, -30.4],
                  [-164, 86, -19.43], [-75, 88, -19.7], [12, 88, -17.88], [99, 89, -17.47], [189, 90, -16.61],
                  [-165, 87, -26.02], [-74, 87, -28.5], [11, 88, -28.02], [100, 89, -25.44], [189, 90, -27.83],
                  [-165, 87, -19.82], [-74, 87, -21.1], [11, 88, -16.11], [100, 89, -15.58], [189, 90, -19.07]
                ]

# show X: delta Pixel X, Y: delta Pixel Y, Z: delta X
fig1 = plt.figure(num = "X: delta Pixel X, Y: delta Pixel Y, Z: delta X")
ax = fig1.add_subplot(projection='3d')
for m, zlow, zhigh in [('o', -20, 50)]:
    xs = [point[0] for point in DeltaXRowOnes]
    ys = [point[1] for point in DeltaXRowOnes]
    zs = [point[2] for point in DeltaXRowOnes]
    ax.scatter(xs, ys, zs, marker=m)
for m, zlow, zhigh in [('^', -20, 50)]:
    xs = [point[0] for point in DeltaXRowTwos]
    ys = [point[1] for point in DeltaXRowTwos]
    zs = [point[2] for point in DeltaXRowTwos]
    ax.scatter(xs, ys, zs, marker=m)

for m, zlow, zhigh in [('x', -20, 50)]:
    xs = [point[0] for point in DeltaXRowThrees]
    ys = [point[1] for point in DeltaXRowThrees]
    zs = [point[2] for point in DeltaXRowThrees]
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel('Delta Pixel X')
ax.set_ylabel('Delta Pixel Y')
ax.set_zlabel('Delta X')

# show Y: delta Pixel X, Y: delta Pixel Y, Z: delta Y
fig2 = plt.figure(num = "X: delta Pixel X, Y: delta Pixel Y, Z: delta Y")
bx = fig2.add_subplot(projection='3d')
for m, zlow, zhigh in [('o', -20, 50)]:
    xs = [point[0] for point in DeltaYRowOnes]
    ys = [point[1] for point in DeltaYRowOnes]
    zs = [point[2] for point in DeltaYRowOnes]
    bx.scatter(xs, ys, zs, marker=m)
for m, zlow, zhigh in [('^', -20, 50)]:
    xs = [point[0] for point in DeltaYRowTwos]
    ys = [point[1] for point in DeltaYRowTwos]
    zs = [point[2] for point in DeltaYRowTwos]
    bx.scatter(xs, ys, zs, marker=m)

for m, zlow, zhigh in [('x', -20, 50)]:
    xs = [point[0] for point in DeltaYRowThrees]
    ys = [point[1] for point in DeltaYRowThrees]
    zs = [point[2] for point in DeltaYRowThrees]
    bx.scatter(xs, ys, zs, marker=m)

bx.set_xlabel('Delta Pixel X')
bx.set_ylabel('Delta Pixel Y')
bx.set_zlabel('Delta Y')

# show X: delta Pixel X, Y: delta Pixel Y, Z: delta Z
fig3 = plt.figure(num = "X: delta Pixel X, Y: delta Pixel Y, Z: delta Z")
cx = fig3.add_subplot(projection='3d')

for m, zlow, zhigh in [('o', -20, 50)]:
    xs = [point[0] for point in DeltaZRowOnes]
    ys = [point[1] for point in DeltaZRowOnes]
    zs = [point[2] for point in DeltaZRowOnes]
    cx.scatter(xs, ys, zs, marker=m)
for m, zlow, zhigh in [('^', -20, 50)]:
    xs = [point[0] for point in DeltaZRowTwos]
    ys = [point[1] for point in DeltaZRowTwos]
    zs = [point[2] for point in DeltaZRowTwos]
    cx.scatter(xs, ys, zs, marker=m)

for m, zlow, zhigh in [('x', -20, 50)]:
    xs = [point[0] for point in DeltaZRowThrees]
    ys = [point[1] for point in DeltaZRowThrees]
    zs = [point[2] for point in DeltaZRowThrees]
    cx.scatter(xs, ys, zs, marker=m)

cx.set_xlabel('Delta Pixel X')
cx.set_ylabel('Delta Pixel Y')
cx.set_zlabel('Delta Z')

plt.show()
'''

x = [500, 576, 650, 727, 804] #row 1
#x = [489, 572, 651, 733, 816] #row 2
#x = [476, 565, 651, 738, 828] #row 3

#y = [-16.87, -14.54, -19.4, -20.03, -15.64] #DeltaZ Blue R1
#y = [-14.08, -16.97, -17.66, -14.975, -18.25] #DeltaZ Blue R2
#y = [-20.22, -16.77, -16.71, -16.59, -15.895] #DeltaZ Blue R3

#y = [-33.78, -35.13, -38.71, -32.05, -32.95] #DeltaZ Red R1
#y = [-33.39, -35.78, -41.31, -38.1, -34.95] #DeltaZ Red R2
#y = [-35.69, -36.34, -36.9, -33.93, -34.58] #DeltaZ Red R3

#y = [-32.07, -33.4, -28.64, -27.64, -29.28] #DeltaZ Yellow R1
#y = [-30.62, -29.8, -33.97, -31.21, -28.59] #DeltaZ Yellow R2
#y = [-28.33, -31.22, -34.48, -29.34, -30.4] #DeltaZ Yellow R3

#y = [-17.64, -16.56, -21.23, -16.47, -16.29] #DeltaZ Green R1
#y = [-14.68, -16.37, -22, -17.19, -17.75] #DeltaZ Green R2
#y = [-19.43, -19.7, -17.88, -17.47, -16.61] #DeltaZ Green R3

#y = [-28.42, -26.57, -28.82, -28.11, -26.25] #DeltaZ Orange R1
#y = [-26.76, -25.99, -29.2, -28.57, -26.5] #DeltaZ Orange R2
#y = [-26.02, -28.5, -28.02, -25.44, -27.83] #DeltaZ Orange R3

#y = [-19.44, -20.99, -17.05, -16.38, -20.14] #DeltaZ White R1
#y = [-20.49, -20.12, -22.44, -20.37, -21.89] #DeltaZ White R2
#y = [-19.82, -21.1, -16.11, -15.58, -19.07] #DeltaZ White R3

#y = [-24.54, -23.76, -23.9, -23.4, -22.66] #DeltaZ LGreen R1
#y = [-23.6, -24.91, -25.94, -24.93, -23.38] #DeltaZ LGreen R2
#y = [-24.25, -23.36, -22.63, -22.51, -23.91] #DeltaZ LGreen R3

#y = [-21.18, -23.09, -19.62, -20.45, -17.18] #DeltaZ LBlue R1
#y = [-19.08, -19.85, -21.78, -20.26, -18.5] #DeltaZ LBlue R2
#y = [-20.73, -22.38, -20.87, -18.22, -16.23] #DeltaZ LBlue R3

#y1 = [3.78, 1.66, 0.01, -3.11, -6.06] #DeltaXBlue R1
#y1 = [2.47, 0.72, -0.58, -3.01, -8.77] #DeltaXBlue R2
#y1 = [2.18, 2.46, -1.43, -2.17, -7.99] #DeltaXBlue R3

#y2 = [7.77, 3.3, -1.8, -5.7, -11.09] #DeltaXRedR1
#y2 = [7.67, 2.34, -2.38, -8.63, -13.98] #DeltaXRedR2
#y2 = [7.57, 2.36, -3.27, -7.76, -15.07] #DeltaXRedR3

#y3 = [7.47, 2.54, -2.08, -4.94, -10.63] #DeltaXYellow R1
#y3 = [5.46, 2.26, -2.53, -6.4, -10.74] #DeltaXYellow R2
#y3 = [6.57, 0.17, -3.28, -7.24, -11.98] #DeltaXYellow R3

#y4 = [4.93, 0.49, -0.85, -2.95, -6.51] #DeltaXGreen R1
#y4 = [1.79, 1.55, -1.94, -5.44, -6.25] #DeltaXGreen R2
#y4 = [4.01, 1.02, -3.69, -3.93, -9.69] #DeltaXGreen R3

#y5 = [6.81, 3.16, -1.59, -4.14, -8.25] #DeltaXOrange R1
#y5 = [5.82, 1.57, -1.77, -5.67, -10.34] #DeltaXOrange R2
#y5 = [6.8, 1.72, -1.65, -6.09, -11.78] #DeltaXOrange R3

#y6 = [4, 1.55, -0.18, -3.34, -7.91] #DeltaXWhite R1
#y6 = [4.71, 1.82, -2.06, -4.11, -8.81] #DeltaXWhite R2
#y6 = [5.71, 0.41, -0.85, -4.27, -10.08] #DeltaXWhite R3

#y7 = [5.16, 2.43, -0.79, -3.55, -7.6] #DeltaXLGreen R1
#y7 = [3.98, 0.78, -2.35, -5.66, -7.55] #DeltaXLGreen R2
#y7 = [4.15, 0.09, -2.57, -5.01, -10.49] #DeltaXLGreen R3

#y8 = [5.98, 1.21, -0.66, -4.73, -6.5] #DeltaXLBlue R1
#y8 = [4.01, 1.09, -2.34, -3.99, -8.23] #DeltaXLBlue R2
#y8 = [5.2, 1.29, -2.78, -4.74, -8.63] #DeltaXLBlue R3

y1 = [0.61, 1.1, 2.44, 4.18, 5.52] #DeltaYBlue R1
#y1 = [-1.35, -0.5, 0.42, 1.55, 1.35] #DeltaYBlue R2
#y1 = [-1.36, -3.31, -3.1, -0.085, -0.53] #DeltaYBlue R3

y2 = [2.57, 4.1, 4.24, 4.42, 5.98] #DeltaYRed R1
#y2 = [-2.78, -2.16, -0.65, 0.01, 1.48] #DeltaYRed R2
#y2 = [-5.95, -7.56, -5.65, -3.95, -2.9] #DeltaYRed R3

y3 = [2.06, 2.76, 3.19, 4.39, 5.37] #DeltaYYellow R1
#y3 = [-1.93, -1.84, -0.82, 0.16, 0.97] #DeltaYYellow R2
#y3 = [-6.79, -5.73, -5.81, -3.94, -3.6] #DeltaYYellow R3

y4 = [0.53, 2.23, 4.44, 2.9, 5.25] #DeltaYGreen R1
#y4 = [-2.17, -0.98, 0.08, 1.99, 2.09] #DeltaYGreen R2
#y4 = [-4.79, -4.89, -3.15, -2.04, -1.21] #DeltaYGreen R3

y5 = [2.17, 3.25, 3.94, 3.84, 6.25] #DeltaYOrange R1
#y5 = [-2.17, -1, -0.9, 0.51, 1.35] #DeltaYOrange R2
#y5 = [-6.29, -5.4, -4.34, -3.13, -2.77] #DeltaYOrange R3

y6 = [0.93, 2.9, 2.22, 2.43, 4.59] #DeltaYWhite R1
#y6 = [-1.53, -1.34, -0.17, 1.01, 2.82] #DeltaYWhite R2
#y6 = [-6.7, -4.54, -2.93, -2.26, -1.84] #DeltaYWhite R3

y7 = [1.26, 2.49, 3.61, 5.36, 5.57] #DeltaYLGreen R1
#y7 = [-2.48, -0.33, 0.35, 1.41, 1.76] #DeltaYLGreen R2
#y7 = [-5.46, -5.09, -3, -2.49, -2.42] #DeltaYLGreen R3

y8 = [1.28, 2.91, 3.2, 4.3, 6.21] #DeltaYLBlue R1
#y8 = [-2.02, -1.14, 0.84, 1.51, 2.59] #DeltaYLBlue R2
#y8 = [-4.55, -4.19, -2.34, -2.44, 0.27] #DeltaYLBlue R3

#x = [536, 612, 692, 768] #Row 1 in_between
#x = [527, 610, 695, 778] #Row 2 in_between

#y = [-1.13, 4.66, 4.69, 2.28] #DeltaZCam Blue_IB Row 1
#y = [-1.79, 1.24, 2.56, -0.25] #DeltaZCam Blue_IB Row 2

#y = [7.55, 8.84, 9.12, 5.03] #DeltaZCam Red_IB Row 1
#y = [5.32, 7.38, 7.6, 4.93] #DeltaZCam Red_IB Row 2

#y = [7.51, 5.62, 5.42, 4.79] #DeltaZCam Yellow_IB Row 1
#y = [4.17, 7.17, 6.34, 4.66] #DeltaZCam Yellow_IB Row 2

#y = [0.15, 4.31, 2.16, 0.74] #DeltaZCam Green_IB Row 1
#y = [-1.47, 1.46, 1.05, 0.2] #DeltaZCam Green_IB Row 2

#y = [3.29, 1.29, 4.48, 1.69] #DeltaZCam Orange_IB Row 1
#y = [0.51, 1.79, 0.92, 1.28] #DeltaZCam Orange_IB Row 2

#y = [0.13, -0.35, -0.75, -0.53] #DeltaZCam White_IB Row 1
#y = [0.51, -0.85, -1.71, -2.14] #DeltaZCam White_IB Row 2

#y = [-0.97, 1.16, 1.18, 0.31] #DeltaZCam LGreen_IB Row 1
#y = [0.16, 1.07, 0.02, 0.21] #DeltaZCam LGreen_IB Row 2

#y = [-0.33, 0.63, 0.5, -2.31] #DeltaZCam LBlue_IB Row 1
#y = [-1.23, -0.77, -2, -3] #DeltaZCam LBlue_IB Row 2



# calc the trendline
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot()
plt.suptitle('Row 3', fontsize=14, fontweight='bold')
#z = np.polyfit(x, y, 1)
z1 = np.polyfit(x, y1, 1)
p1 = np.poly1d(z1)
#pylab.plot(x, p(x),"r--")
#ax.set_title('Trend Line : y={:.4f}x+{:.4f}'.format(z[0], z[1]))
ax.plot(x, p1(x), 'b', label ='y1={:.4f}x+{:.4f}'.format(z1[0], z1[1]))
z2 = np.polyfit(x, y2, 1)
p2 = np.poly1d(z2)
ax.plot(x, p2(x), 'g', label ='y2={:.4f}x+{:.4f}'.format(z2[0], z2[1]))
z3 = np.polyfit(x, y3, 1)
p3 = np.poly1d(z3)
ax.plot(x, p3(x), 'c', label ='y3={:.4f}x+{:.4f}'.format(z3[0], z3[1]))
z4 = np.polyfit(x, y4, 1)
p4 = np.poly1d(z4)
ax.plot(x, p4(x), color='#90EE90', label ='y4={:.4f}x+{:.4f}'.format(z4[0], z4[1]))
z5 = np.polyfit(x, y5, 1)
p5 = np.poly1d(z5)
ax.plot(x, p5(x), color='orange', label='y5={:.4f}x+{:.4f}'.format(z5[0], z5[1]))
z6 = np.polyfit(x, y6, 1)
p6 = np.poly1d(z6)
ax.plot(x, p6(x), color='red', label='y6={:.4f}x+{:.4f}'.format(z6[0], z6[1]))
z7 = np.polyfit(x, y7, 1)
p7 = np.poly1d(z7)
ax.plot(x, p7(x), color='black', label='y7={:.4f}x+{:.4f}'.format(z7[0], z7[1]))
z8 = np.polyfit(x, y8, 1)
p8 = np.poly1d(z8)
ax.plot(x, p8(x), 'y', label='y8={:.4f}x+{:.4f}'.format(z8[0], z8[1]))
plt.legend(loc="upper right")
ax.set_xlabel('Pixel X')
ax.set_ylabel('Delta X')

plt.show()

'''coefficients = np.polyfit(x, y, 3)

poly = np.poly1d(coefficients)


new_x = np.linspace(x[0], x[-1])

new_y = poly(new_x)


plt.plot(x, y, "o", new_x, new_y)
print("y=%.6fx^3+%.6fx^2+%.6fx+(%.6f)"%(poly[3],poly[2],poly[1], poly[0]))

pylab.plot(x, y,'o')'''