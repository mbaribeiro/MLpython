import check_cube as ckc
import pandas as pd

def media_temp_velocidade(sala, n_cubos, data):
    # Inicializa um dicionário para armazenar a soma de temperatura e velocidade em cada cubo
    cubos = {}
    for i in range(1, n_cubos+1):
        cubos[i] = {'temp': 0, 'vel': 0, 'count': 0}
    
    # Calcula a média de temperatura e velocidade em cada cubo
    for ponto in data:
        cubo_num = ckc.check_cube(sala, n_cubos, ponto[0:3])
        cubos[cubo_num]['temp'] += ponto[3]
        cubos[cubo_num]['vel'] += ponto[4]
        cubos[cubo_num]['count'] += 1
    
    for cubo_num in cubos:
        if cubos[cubo_num]['count'] > 0:
            cubos[cubo_num]['temp'] /= cubos[cubo_num]['count']
            cubos[cubo_num]['vel'] /= cubos[cubo_num]['count']
    
    # Cria uma tabela com a média de temperatura e velocidade em cada cubo
    result = pd.DataFrame.from_dict(cubos, orient='index')
    result.index.name = 'cubo'
    result.reset_index(inplace=True)
    
    return result