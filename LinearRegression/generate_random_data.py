import random
data_f = "sealcrypto/data_50_quad.txt"
label_f = "sealcrypto/labels_50_quad.txt"
with open(data_f, 'w') as data_f:
    with open(label_f, 'w') as label_f:
        for i in range(50):
            x_1 = random.randrange(300)/10
            delta = random.randrange(-200,200)/10
            if delta <0:
                y =  3.8 + 2.5 *x_1 + 1.2*(pow(x_1,2)) + delta - 0.2 *pow(delta, 2)
            else:
                y =  3.8 + 2.5 *x_1 + 1.2*(pow(x_1,2)) + delta + 0.2 *pow(delta, 2)
            new_line = "{}\t{}\t{}\n".format(1, x_1, pow(x_1, 2))
            data_f.write(new_line)
            label_f.write(str(y) + "\n")



