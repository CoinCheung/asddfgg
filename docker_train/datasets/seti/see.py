
with open('./train_split.txt', 'r') as fr:
    lines = fr.read().splitlines()[1:]

lbs = [int(el.split(',')[1]) for el in lines]
print(sum(lbs))
print(len(lbs))


with open('./train_all.txt', 'r') as fr:
    lines = fr.read().splitlines()[1:]

lbs = [int(el.split(',')[1]) for el in lines]
print(sum(lbs))
print(len(lbs))



with open('./val_split.txt', 'r') as fr:
    lines = fr.read().splitlines()[1:]

lbs = [int(el.split(',')[1]) for el in lines]
print(sum(lbs))
print(len(lbs))
