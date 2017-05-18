import random
data_f = "sealcrypto/data_5_1.txt"
label_f = "sealcrypto/labels_5_1.txt"
with open(data_f, 'w') as data_f:
    with open(label_f, 'w') as label_f:
        for i in range(5):
            x_1 = random.randrange(300)/10
            x_2 = random.randrange(300)/10
            x_3 = random.randrange(300)/10
            x_4 = random.randrange(300)/10
            #x_5 = random.randrange(300)/10
            delta = random.randrange(-20,20)/10
            #y = .1 * (x_1) + .9 * (x_2) + 1.4*(x_3) + 2*(x_4) + delta
            y = .45 * (x_1)+ delta
            #3.7*(x_5)  + delta
            new_line = "{}\n".format(x_1)
            #new_line = "{}\t{}\t{}\t{}\n".format(x_1, x_2, x_3, x_4)
            #, x_5)
            #print(new_line)
            data_f.write(new_line)
            label_f.write(str(y) + "\n")



