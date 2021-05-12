import numpy as np
import json
import calc
import matplotlib.pyplot as plt

blue = [[500, 312, -16.87], [576, 313, -14.54], [650, 313, -19.4], [727, 313, -20.03], [804, 313, -15.64],
        [489, 374, -14.08], [572, 374, -16.97], [651, 375, -17.66], [733, 375, -14.975], [816, 376, -18.25],
        [476, 446, -20.22], [565, 447, -16.77], [651, 448, -16.71], [738, 447, -16.59], [828, 449, -15.895],
        [536, 342, -15.62], [612, 344, -17.15], [692, 343, -18.02], [768, 343, -17.23], [527, 408, -17.01],
        [610, 409, -17.03], [695, 410, -16.49], [778, 410, -16.43]]

red = [[500, 312, -33.78], [576, 312, -35.13], [652, 313, -38.71], [727, 314, -32.05], [804, 314, -32.95],
       [490, 374, -33.39], [572, 375, -35.78], [652, 375, -41.31], [734, 376, -38.1], [816, 376, -34.95],
       [476, 445, -35.69], [566, 448, -36.34], [652, 448, -36.9], [740, 448, -33.93], [829, 449, -34.58],
       [536, 342, -26.97], [612, 344, -28.9], [692, 343, -28.43], [768, 343, -29.49], [527, 408, -29.98],
       [610, 409, -30.21], [695, 410, -29.96], [778, 410, -30.46]]

yellow = [[500, 312, -32.07], [577, 313, -33.4], [652, 313, -28.64], [728, 313, -27.86], [805, 314, -29.28],
          [490, 374, -30.62], [571, 375, -29.8], [652, 376, -33.97], [734, 376, -31.21], [815, 376, -28.59],
          [475, 447, -28.33], [566, 447, -31.22], [652, 449, -34.48], [741, 450, -29.34], [828, 450, -30.4],
          [536, 342, -23.97], [612, 344, -25.84], [692, 343, -25], [768, 343, -24.45], [527, 408, -25.83],
          [610, 409, -25.2], [695, 410, -25.91], [778, 410, -25.23]]

green = [[500, 312, -17.64], [577, 312, -16.56], [652, 312, -21.23], [727, 314, -16.47], [804, 314, -16.29],
         [490, 374, -14.68], [571, 374, -16.37], [652, 375, -22], [734, 374, -17.19], [814, 376, -17.75],
         [476, 446, -19.43], [565, 448, -19.7], [652, 448, -17.88], [739, 449, -17.47], [829, 450, -16.61],
         [536, 342, -16.17], [612, 344, -14.73], [692, 343, -17.07], [768, 343, -16.19], [527, 408, -19.02],
         [610, 409, -17.53], [695, 410, -27.59], [778, 410, -17.06]]

orange = [[500, 312, -28.42], [576, 312, -26.57], [651, 313, -28.82], [727, 314, -28.11], [804, 313, -26.25],
          [489, 374, -26.76], [571, 374, -25.99], [652, 375, -29.2], [733, 375, -28.57], [815, 376, -26.5],
          [475, 447, -26.02], [566, 447, -28.5], [651, 448, -28.02], [740, 449, -25.44], [829, 450, -27.83],
          [535, 342, -25.97], [612, 344, -26.37], [692, 343, -24.22], [768, 343, -25.64], [527, 408, -26.28],
          [610, 409, -26.17], [695, 410, -26.84], [778, 410, -25.87]]

white = [[501, 312, -19.44], [576, 312, -20.99], [651, 313, -17.05], [728, 314, -16.38], [805, 314, -20.14],
         [489, 373, -20.49], [571, 375, -20.23], [652, 375, -22.44], [733, 375, -20.37], [815, 375, -21.89],
         [475, 447, -19.82], [566, 447, -21.1], [651, 448, -16.11], [740, 449, -15.58], [829, 450, -19.07],
         [535, 342, -20.19], [612, 344, -20.54], [692, 343, -19.89], [768, 343, -20.3], [527, 408, -19.88],
         [610, 409, -20.8], [695, 410, -20.22], [778, 410, -21.34]]

lgreen = [[501, 312, -24.54], [576, 312, -23.76], [651, 313, -23.9], [727, 313, -23.4], [804, 313, -22.66],
          [490, 374, -23.6], [572, 374, -24.91], [652, 374, -25.94], [733, 375, -24.93], [814, 376, -23.38],
          [477, 446, -24.25], [567, 447, -23.36], [652, 447, -22.63], [740, 449, -22.51], [828, 449, -23.91],
          [535, 342, -22.81], [612, 344, -23.48], [692, 343, -23.4], [768, 343, -23.28], [527, 408, -23.89],
          [610, 409, -23.11], [695, 410, -23.94], [778, 410, -23.51]]

lblue = [[500, 312, -21.18], [577, 313, -23.09], [652, 313, -19.62], [728, 313, -20.45], [805, 314, -17.18],
         [490, 374, -19.08], [571, 375, -19.85], [652, 376, -21.78], [734, 376, -20.26], [815, 376, -18.5],
         [475, 447, -20.73], [566, 447, -22.38], [652, 449, -20.87], [741, 450, -18.22], [828, 450, -16.23],
         [535, 342, -21.1], [612, 344, -20.43], [692, 343, -20.03], [768, 343, -21.4], [527, 408, -21.78],
         [610, 409, -22.03], [695, 410, -22.26], [778, 410, -21.24]]

test_data_blue = [[564, 306, -16.87], [651, 309, -16.41], [720, 295, -15.84], [786, 317, -10.87], [525, 374, -14.73],
                  [603, 371, -18.99], [678, 381, -16.53], [739, 346, -17.54], [617, 433, -13.27], [705, 422, -13.4],
                  [802, 398, -16.53], [475, 433, -18.17], [553, 452, -16.44], [626, 466, -15.09], [733, 443, -14.56]]

pixels = [[500, 312],
          [576, 313],
          [650, 313],
          [727, 313],
          [804, 313],
          [489, 374],
          [572, 374],
          [651, 375],
          [733, 375],
          [816, 376],
          [476, 446],
          [565, 447],
          [651, 448],
          [738, 447],
          [828, 449]]

#bgr values
blue_boundaries = [[170, 205], [60, 95], [0, 30]]
red_boundaries = [[40, 80], [25, 60], [205, 245]]
yellow_boundaries = [[0, 70], [90, 170], [165, 255]]
green_boundaries = [[45, 70], [90, 115], [0, 10]]
orange_boundaries = [[50, 85], [65, 90], [225, 255]]
white_boundaries = [[225, 245], [200, 225], [210, 235]]
lgreen_boundaries = [[35, 55], [125, 145], [0, 40]]
lblue_boundaries = [[210, 225], [125, 140], [35, 60]]


def euclidean_distance(x1, x2):
    return np.sqrt(((x1 - x2) ** 2))


def main():
    with open('color.json') as f:
        colormap = json.load(f)
    with open('depthvalues.json') as f:
        depthvalues = json.load(f)

    for pixel in pixels:
        color = getColor(pixel[0], pixel[1], colormap)
        row = setRow(pixel[1])
        column = setColumn(pixel[0])

        pos = calc.calculatepixels2coord(pixel[0], pixel[1], depthvalues)
        print(pos)
        # pos[0], pos[1] -> X_ist, Y_ist

        if color == 'blue':
            if row == 1:
                deltaXLine = [-0.0322, 20.2566]
            elif row == 2:
                deltaXLine = [-0.0322, 19.1761]
            elif row == 3:
                deltaXLine = [-0.0285, 17.1786]

            if column == 1:
                deltaYLine = [-0.0143, 4.7007]
            elif column == 2:
                deltaYLine = [-0.0331, 11.6053]
            elif column == 3:
                deltaYLine = [-0.0412, 15.5391]
            elif column == 4:
                deltaYLine = [-0.0316, 13.8313]
            elif column == 5:
                deltaYLine = [-0.044, 18.8015]


        elif color == 'red':
            if row == 1:
                deltaXLine = [-0.0379, 23.4931]
            elif row == 2:
                deltaXLine = [-0.0405, 24.6951]
            elif row == 3:
                deltaXLine = [-0.0414, 25.1631]

            if column == 1:
                deltaYLine = [-0.0533, 17.8517]
            elif column == 2:
                deltaYLine = [-0.0551, 19.8508]
            elif column == 3:
                deltaYLine = [-0.0381, 14.1483]
            elif column == 4:
                deltaYLine = [-0.0353, 13.7402]
            elif column == 5:
                deltaYLine = [-0.0477, 19.9563]

        elif color == 'yellow':
            if row == 1:
                deltaXLine = [-0.0407, 25.5764]
            elif row == 2:
                deltaXLine = [-0.0363, 21.7653]
            elif row == 3:
                deltaXLine = [-0.0384, 23.1014]

            if column == 1:
                deltaYLine = [-0.0433, 14.5692]
            elif column == 2:
                deltaYLine = [-0.0526, 19.0843]
            elif column == 3:
                deltaYLine = [-0.0411, 16.1340]
            elif column == 4:
                deltaYLine = [-0.0504, 20.1990]
            elif column == 5:
                deltaYLine = [-0.0434, 19.4728]

        elif color == 'green':
            if row == 1:
                deltaXLine = [-0.0615, 38.5892]
            elif row == 2:
                deltaXLine = [-0.0666, 40.4382]
            elif row == 3:
                deltaXLine = [-0.0632, 37.9416]

            if column == 1:
                deltaYLine = [-0.0631, 21.7414]
            elif column == 2:
                deltaYLine = [-0.0866, 30.8584]
            elif column == 3:
                deltaYLine = [-0.0731, 27.0013]
            elif column == 4:
                deltaYLine = [-0.0623, 23.6694]
            elif column == 5:
                deltaYLine = [-0.0652, 26.2355]
        elif color == 'orange':
            if row == 1:
                deltaXLine = [-0.0493, 31.2982]
            elif row == 2:
                deltaXLine = [-0.0486, 29.5904]
            elif row == 3:
                deltaXLine = [-0.0513, 31.2301]

            if column == 1:
                deltaYLine = [-0.0630, 21.7414]
            elif column == 2:
                deltaYLine = [-0.0644, 23.2983]
            elif column == 3:
                deltaYLine = [-0.0609, 22.6338]
            elif column == 4:
                deltaYLine = [-0.0520, 20.0709]
            elif column == 5:
                deltaYLine = [-0.0661, 26.6702]

        elif color == 'white':
            if row == 1:
                deltaXLine = [-0.0415, 26.1701]
            elif row == 2:
                deltaXLine = [-0.0362, 21.4363]
            elif row == 3:
                deltaXLine = [-0.0392, 22.8033]

            if column == 1:
                deltaYLine = [-0.0499, 16.6081]
            elif column == 2:
                deltaYLine = [-0.0568, 20.5116]
            elif column == 3:
                deltaYLine = [-0.0489, 18.8265]
            elif column == 4:
                deltaYLine = [-0.0585, 23.5457]
            elif column == 5:
                deltaYLine = [-0.0587, 23.9077]

        elif color == 'lgreen':
            if row == 1:
                deltaXLine = [-0.0347, 21.6199]
            elif row == 2:
                deltaXLine = [-0.0283, 16.3750]
            elif row == 3:
                deltaXLine = [-0.0369, 21.5956]

            if column == 1:
                deltaYLine = [-0.0396, 12.8040]
            elif column == 2:
                deltaYLine = [-0.0531, 18.8767]
            elif column == 3:
                deltaYLine = [-0.0559, 21.6132]
            elif column == 4:
                deltaYLine = [-0.0374, 15.0904]
            elif column == 5:
                deltaYLine = [-0.0474, 20.0388]

        elif color == 'lblue':
            if row == 1:
                deltaXLine = [-0.0575, 35.9560]
            elif row == 2:
                deltaXLine = [-0.0504, 30.4584]
            elif row == 3:
                deltaXLine = [-0.0508, 29.9312]

            if column == 1:
                deltaYLine = [-0.0661, 22.7155]
            elif column == 2:
                deltaYLine = [-0.0630, 22.2226]
            elif column == 3:
                deltaYLine = [-0.0667, 24.1165]
            elif column == 4:
                deltaYLine = [-0.0620, 23.7164]
            elif column == 5:
                deltaYLine = [-0.0659, 25.8992]

        DeltaX = deltaXLine[0] * pixel[0] + deltaXLine[1]
        DeltaY = deltaYLine[0] * pixel[1] + deltaYLine[1]
        z_new = calc.getDeltaZ(pixel[0], pixel[1], pos[0], pos[1])
        x_y_z_new = calc.fixXandY(pixel[0], pixel[1], z_new[0])
        print('Pixel: ',pixel[0:2])
        #pixel.append(DeltaX)
        #pixel.append(DeltaY)
        #pixel.append(DeltaZ)
        #print(deltaXLine)
        #print(deltaYLine)
        print('Xnew', x_y_z_new[0])
        print('Ynew', x_y_z_new[1])
        print('Znew', x_y_z_new[2][0])
        print('--------------------')


    '''x = [test_data_blue[0][0], test_data_blue[1][0], test_data_blue[2][0], test_data_blue[3][0]]
    y = [0.82, -1.28, -3.45, -4.82]
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot()
    plt.suptitle('Row 1', fontsize=14, fontweight='bold')
    z1 = np.polyfit(x, y, 1)
    p1 = np.poly1d(z1)
    ax.plot(x, p1(x), 'b', label='y={:.4f}x+{:.4f}'.format(z1[0], z1[1]))
    deltaXLine = [-0.0322, 20.2566]
    p2 = np.poly1d(deltaXLine)
    ax.plot(x, p2(x), 'r', label='y={:.4f}x+{:.4f}'.format(deltaXLine[0], deltaXLine[1]))
    plt.legend(loc="upper right")
    ax.set_xlabel('PixelX')
    ax.set_ylabel('DeltaX')

    plt.show()'''


def getColor(x, y, colormap):
    bgr = colormap[y][x+2]
    #print(bgr)
    if (blue_boundaries[0][0] <= bgr[0] <= blue_boundaries[0][1]) & (blue_boundaries[1][0] <= bgr[1] <= blue_boundaries[1][1]) & (blue_boundaries[2][0] <= bgr[2] <= blue_boundaries[2][1]):
        return 'blue'
    elif (red_boundaries[0][0] <= bgr[0] << red_boundaries[0][1]) & (red_boundaries[1][0] <= bgr[1] <= red_boundaries[1][1]) & (red_boundaries[2][0] <= bgr[2] <= red_boundaries[2][1]):
        return 'red'
    elif (yellow_boundaries[0][0] <= bgr[0] <= yellow_boundaries[0][1]) & (yellow_boundaries[1][0] <= bgr[1] <= yellow_boundaries[1][1]) & (yellow_boundaries[2][0] <= bgr[2] <= yellow_boundaries[2][1]):
        return 'yellow'
    elif (green_boundaries[0][0] <= bgr[0] <= green_boundaries[0][1]) & (green_boundaries[1][0] <= bgr[1] <= green_boundaries[1][1]) & (green_boundaries[2][0] <= bgr[2] <= green_boundaries[2][1]):
        return 'green'
    elif (orange_boundaries[0][0] <= bgr[0] <= orange_boundaries[0][1]) & (orange_boundaries[1][0] <= bgr[1] <= orange_boundaries[1][1]) & (orange_boundaries[2][0] <= bgr[2] <= orange_boundaries[2][1]):
        return 'orange'
    elif (white_boundaries[0][0] <= bgr[0] <= white_boundaries[0][1]) & (white_boundaries[1][0] <= bgr[1] <= white_boundaries[1][1]) & (white_boundaries[2][0] <= bgr[2] <= white_boundaries[2][1]):
        return 'white'
    elif (lgreen_boundaries[0][0] <= bgr[0] <= lgreen_boundaries[0][1]) & (lgreen_boundaries[1][0] <= bgr[1] <= lgreen_boundaries[1][1]) & (lgreen_boundaries[2][0] <= bgr[2] <= lgreen_boundaries[2][1]):
        return 'lgreen'
    elif (lblue_boundaries[0][0] <= bgr[0] <= lblue_boundaries[0][1]) & (lblue_boundaries[1][0] <= bgr[1] <= lblue_boundaries[1][1]) & (lblue_boundaries[2][0] <= bgr[2] <= lblue_boundaries[2][1]):
        return 'lblue'


def setRow(y):
    y_rows = [313, 375, 447]
    distances = [euclidean_distance(y, ys) for ys in y_rows]
    return (np.argsort(distances)[0]+1)    #Return row number

def setColumn(x):
    x_columns = [488, 571, 651, 733, 816]
    distances = [euclidean_distance(x, xs) for xs in x_columns]
    return (np.argsort(distances)[0]+1)
'''def blue_knn(part):
    distances = [euclidean_distance(part, trainpart) for trainpart in blue]
    k_indices = np.argsort(distances)[:4]
    for n in k_indices:
        knn.append(blue[n])
    return knn, distances[:4]


def red_knn(part):
    distances = [euclidean_distance(part, trainpart) for trainpart in red]
    k_indices = np.argsort(distances)[:4]
    for n in k_indices:
        knn.append(red[n])
    return knn, distances[:4]


def yellow_knn(part):
    distances = [euclidean_distance(part, trainpart) for trainpart in yellow]
    k_indices = np.argsort(distances)[:4]
    for n in k_indices:
        knn.append(yellow[n])
    return knn, distances[:4]


def green_knn(part):
    distances = [euclidean_distance(part, trainpart) for trainpart in green]
    k_indices = np.argsort(distances)[:4]
    for n in k_indices:
        knn.append(green[n])
    return knn, distances[:4]


def orange_knn(part):
    distances = [euclidean_distance(part, trainpart) for trainpart in orange]
    k_indices = np.argsort(distances)[:4]
    for n in k_indices:
        knn.append(orange[n])
    return knn, distances[:4]


def white_knn(part):
    distances = [euclidean_distance(part, trainpart) for trainpart in white]
    k_indices = np.argsort(distances)[:4]
    for n in k_indices:
        knn.append(white[n])
    return knn, distances[:4]


def lgreen_knn(part):
    distances = [euclidean_distance(part, trainpart) for trainpart in lgreen]
    k_indices = np.argsort(distances)[:4]
    for n in k_indices:
        knn.append(lgreen[n])
    return knn, distances[:4]


def lblue_knn(part):
    distances = [euclidean_distance(part, trainpart) for trainpart in lblue]
    k_indices = np.argsort(distances)[:4]
    for n in k_indices:
        knn.append(lblue[n])
    return knn, distances[:4]

def get_weight(d):
    dtotal = d[0] + d[1] + d[2] + d[3]
    total = dtotal/d[0] + dtotal/d[1] + dtotal/d[2] + dtotal/d[3]
    return [(dtotal / d[0]) / total, (dtotal/d[1]) / total, (dtotal/d[2]) / total, (dtotal/d[3]) / total]'''

if __name__ == "__main__":
    main()

