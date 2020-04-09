e2f = {'dog' : 'chien', 'cat' : 'chat', 'walrus' : 'morse'}

print(e2f['dog'])

f2e_tuple = list(e2f.items())

print(f2e_tuple)

f2e = {}

for key, value in e2f.items():
    f2e[value] = key

print(f2e)