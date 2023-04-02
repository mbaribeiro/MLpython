const = 0
def data():
    global const
    with open('data.txt') as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].split()
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    const = len(data)
    return data