from Scripts import check_cube as ckc
import pandas as pd

def media_temp_vel(room, n_cubes, data):
    # Inicialize the dictionary
    cubes = {}

    # Calcule the volume of the cube
    x_dim = abs(room[1] - room[0])
    y_dim = abs(room[3] - room[2])
    z_dim = abs(room[5] - room[4])

    total_volume = x_dim * y_dim * z_dim  # volume of the room
    cube_volume = total_volume / n_cubes  # volume of the cube
    cube_dim = round(pow(cube_volume, 1/3))  # dimension of the cube

    # calculate the number of cubes
    n_cubes_x = int(x_dim / cube_dim)
    n_cubes_y = int(y_dim / cube_dim)
    n_cubes_z = int(z_dim / cube_dim)

    for i in range(1, n_cubes_x+1):
        cubes[i] = {}
        for j in range(1, n_cubes_y+1):
            cubes[i][j] = {}
            for k in range(1, n_cubes_z+1):
                cubes[i][j][k] = {'temp': 0, 'vel': 0, 'count': 0}

    # Calculate the volume of each cube
    cube_volume = total_volume / n_cubes

    # Calculate the media of the temperature and velocity
    for point in data:
        x, y, z = ckc.check_cube(room, cube_volume, point[0:3])
        cubes[x][y][z]['temp'] += point[3]
        cubes[x][y][z]['vel'] += point[4]
        cubes[x][y][z]['count'] += 1

    # Calculate the media
    for i in cubes:
        for j in cubes[i]:
            for k in cubes[i][j]:
                if cubes[i][j][k]['count'] > 0:
                    cubes[i][j][k]['temp'] /= cubes[i][j][k]['count']
                    cubes[i][j][k]['vel'] /= cubes[i][j][k]['count']

    return cubes
