def data(filename):
    global const
    with open(filename) as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].split()
        for j in range(len(data[i])):
            data[i][j] = data[i][j].replace(',','')
            data[i][j] = float(data[i][j])
    const = len(data)
    return data
