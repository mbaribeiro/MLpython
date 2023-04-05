def check_cube(sala, n_cubos, ponto):
    cubo_dim = int(pow(n_cubos, 1/3))  # dimensão de cada cubo
    cubo_vol = (sala[0] / cubo_dim) * (sala[1] / cubo_dim) * (sala[2] / cubo_dim)  # volume de cada cubo
    
    # Verifica em qual cubo o ponto está

    x, y, z = ponto[0],ponto[1],ponto[2]
    cubo_x = int(x / (sala[0] / cubo_dim))
    cubo_y = int(y / (sala[1] / cubo_dim))
    cubo_z = int(z / (sala[2] / cubo_dim))
    cubo_num = (cubo_z * cubo_dim * cubo_dim) + (cubo_y * cubo_dim) + cubo_x + 1
    
    return cubo_num