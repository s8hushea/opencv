import numpy as np
import json
import calc


def main():
    points = [[459, 251], [474, 297], [819, 254], [807, 301], [614, 357], [614, 399], [396, 516], [420, 550], [855, 503]
        , [837, 538]]
    # import depthvalues
    with open('depthvalues.json') as f:
        transformed_depth = json.load(f)

    positions = []
    for point in points:
        pos = calc.calculatepixels2coord(point[0], point[1], transformed_depth)
        positions.append(pos)

    print('Length1:\n {}'.format(get_length(positions[0], positions[1])))
    print('Length2:\n {}'.format(get_length(positions[2], positions[3])))
    print('Length3:\n {}'.format(get_length(positions[4], positions[5])))
    print('Length4:\n {}'.format(get_length(positions[6], positions[7])))
    print('Length5:\n {}'.format(get_length(positions[8], positions[9])))


def get_length(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


if __name__ == "__main__":
    main()
