one = 'один'
two = 'два'
three = 'три'
four = 'четыре'
five = 'пять'
six = 'шесть'
seven = 'семь'
eight = 'восемь'
nine = 'девять'
ten = 'десять'

original = [one, two, three, four, five, six, seven, eight, nine, ten]

def change(anything):
    for string in anything:
        num = anything.index(string)
        if string[-1:] == 'ь':
            anything[num] = string[:-1] + 'и'
        else:
            if string[-1:] == 'е' or string[-1:] == 'и':
                anything[num] = string[:-1] + 'ёх'
            else:
                if string[-1:] == 'а':
                    anything[num] = string[:-1] + 'ух'
                else:
                    if string[-1:] == 'н':
                        anything[num] = string[:-2] + 'ного'

copy = original.copy()

change(copy)

print(original)
print(copy)
