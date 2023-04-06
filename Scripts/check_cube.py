def check_cube(room, cube_volume, point):
    cube_dim = round(pow(cube_volume, 1/3))  # dimension of the cube

    # Calculate the width, height and depth of the room
    width = abs(room[1] - room[0])
    height = abs(room[3] - room[2])
    depth = abs(room[5] - room[4])

    # Calculate the number of cubes
    n_cubes_x = int(width / cube_dim)
    n_cubes_y = int(height / cube_dim)
    n_cubes_z = int(depth / cube_dim)
    n_cubes = n_cubes_x * n_cubes_y * n_cubes_z

    # Calculate the coordinates of the origin
    origin_x = min(room[0], room[1])
    origin_y = min(room[2], room[3])
    origin_z = min(room[4], room[5])

    # Calculate the cube index
    x, y, z = point[0], point[1], point[2]
    cube_x = int((x - origin_x) / (width / (n_cubes_x - 1)))
    cube_y = int((y - origin_y) / (height / (n_cubes_y - 1)))
    cube_z = int((z - origin_z) / (depth / (n_cubes_z - 1)))

    return cube_x + 1,cube_y + 1,cube_z + 1
