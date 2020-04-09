while True:
    value = input("Integer, please [q to quit]:")
    if value == 'q':
        break
    number = int(value)
    print(value, "squared is", number * 2)