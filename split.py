import random
val_1 = []
remain = []
'''
with open("H:\SFU-VOC_360\VOC_360\ImageSets\Main\\val.txt", "r") as f_val:
    for line in f_val.readlines():
        temp = random.random()
        if temp>0 and temp<0.4:
            val_1.append(line)
        else:
            remain.append(line)
    f_val.close()
with open("H:\SFU-VOC_360\VOC_360\ImageSets\Main\\val_1.txt", "a") as f_val_1:
    for line in val_1:
        f_val_1.write(line)
    f_val_1.close()
'''
train_line = []
count = 0
with open("H:\SFU-VOC_360\VOC_360\ImageSets\Main\\train.txt", "r") as f_train:
    for line in f_train.readlines():
        train_line.append(line)
        count = count + 1
        if count > 3000:
            break
f_train.close()
with open("H:\\SFU-VOC_360\\VOC_360\\ImageSets\\Main\\train_1.txt","w")as f:
    for line in train_line:
        f.write(line)
f.close()
print(count)
