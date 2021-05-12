import numpy as np
import json

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

pixels = [[512, 302],
          [601, 293],
          [669, 310],
          [731, 313],
          [792, 295],
          [494, 375],
          [562, 355],
          [618, 369],
          [705, 385],
          [781, 399],
          [466, 437],
          [528, 433],
          [606, 408],
          [684, 444],
          [799, 455]]

blue_boundaries = [[170, 205], [60, 95], [0, 30]]
red_boundaries = [[40, 80], [25, 60], [205, 245]]
yellow_boundaries = [[0, 70], [90, 170], [165, 255]]
green_boundaries = [[45, 70], [90, 115], [0, 10]]
orange_boundaries = [[50, 85], [65, 90], [225, 255]]
white_boundaries = [[225, 245], [200, 225], [210, 235]]
lgreen_boundaries = [[35, 55], [125, 145], [0, 40]]
lblue_boundaries = [[210, 225], [125, 140], [35, 60]]


def euclidean_distance(x1, x2):
    return np.sqrt(((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2))


def main():
    with open('color.json') as f:
        colormap = json.load(f)

    for pixel in pixels:
        color = getColor(pixel[0], pixel[1], colormap)
        print(color)
        if color == 'blue':
            deltaXLine = [-0.0332, 20.3393]
            DeltaX = deltaXLine[0] * pixel[0] + deltaXLine[1]
            deltaYLine = [0.01, -6.4184]
            DeltaY = deltaYLine[0] * pixel[0] + deltaYLine[1]
            pixel.append(DeltaX)
            pixel.append(DeltaY)
        elif color == 'red':
            deltaXLine = [-0.043, 26.4787]
            DeltaX = deltaXLine[0] * pixel[0] + deltaXLine[1]
            deltaYLine = [0.0121, -8.1777]
            DeltaY = deltaYLine[0] * pixel[0] + deltaYLine[1]
            pixel.append(DeltaX)
            pixel.append(DeltaY)
        elif color == 'yellow':
            deltaXLine = [-0.0413, 25.3398]
            DeltaX = deltaXLine[0] * pixel[0] + deltaXLine[1]
            deltaYLine = [0.0141, -8.7805]
            DeltaY = deltaYLine[0] * pixel[0] + deltaYLine[1]
            pixel.append(DeltaX)
            pixel.append(DeltaY)
        elif color == 'green':
            deltaXLine = [-0.0686, 42.1663]
            DeltaX = deltaXLine[0] * pixel[0] + deltaXLine[1]
            deltaYLine = [0.0112, -7.8822]
            DeltaY = deltaYLine[0] * pixel[0] + deltaYLine[1]
            pixel.append(DeltaX)
            pixel.append(DeltaY)
        elif color == 'orange':
            deltaXLine = [-0.0535, 33.1977]
            DeltaX = deltaXLine[0] * pixel[0] + deltaXLine[1]
            deltaYLine = [0.0108, -7.4058]
            DeltaY = deltaYLine[0] * pixel[0] + deltaYLine[1]
            pixel.append(DeltaX)
            pixel.append(DeltaY)
        elif color == 'white':
            deltaXLine = [-0.0418, 25.3575]
            DeltaX = deltaXLine[0] * pixel[0] + deltaXLine[1]
            deltaYLine = [0.0125, -9.4287]
            DeltaY = deltaYLine[0] * pixel[0] + deltaYLine[1]
            pixel.append(DeltaX)
            pixel.append(DeltaY)
        elif color == 'lgreen':
            deltaXLine = [-0.0359, 21.5497]
            DeltaX = deltaXLine[0] * pixel[0] + deltaXLine[1]
            deltaYLine = [0.0129, -8.4060]
            DeltaY = deltaYLine[0] * pixel[0] + deltaYLine[1]
            pixel.append(DeltaX)
            pixel.append(DeltaY)
        elif color == 'lblue':
            deltaXLine = [-0.05673, 34.6161]
            DeltaX = deltaXLine[0] * pixel[0] + deltaXLine[1]
            deltaYLine = [0.0093, -7.2362]
            DeltaY = deltaYLine[0] * pixel[0] + deltaYLine[1]
            pixel.append(DeltaX)
            pixel.append(DeltaY)

    print(pixels)



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

