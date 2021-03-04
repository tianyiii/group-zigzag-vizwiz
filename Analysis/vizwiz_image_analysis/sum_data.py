# run more than 20k images at once raise out-of-memory error
# so I split the train dataset to 24 subdirs each cosists of 1000 images
# each txt file is the model output for each subdir
# this file manage to concate everything and produce the final output
# txt format: first line defualtdict for counting bounding boxes
#             second (last) line: defaultdict counting 80 classes


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_list = ["001001.txt", "001002.txt", 
               "002001.txt", "002002.txt",
               "003001.txt", "003002.txt",
               "004001.txt", "004002.txt",
               "005001.txt", "005002.txt",
               "006001.txt", "006002.txt",
               "007001.txt", "007002.txt",
               "008001.txt", "008002.txt",
               "009001.txt", "009002.txt",
               "010001.txt", "010002.txt",
               "011001.txt", "010002.txt",
               "012001.txt", "012002.txt"]

#file_list = ["001001.txt"]
box_cnt = defaultdict(int)
class_cnt = defaultdict(int)

def readdict(dic, line, i=0):
    ele = np.array(line.split(","))
    if i:
        ele[1]=ele[1][1:]
    else:
        ele[1]=ele[1][2:]
    ele[-1]=ele[-1][1:-2]
    for j in range(1,len(ele)):
        dual = ele[j].split(':')
        if i:
            dic[dual[0][2:-1]] += int(dual[1])
        else:
            dic[int(dual[0])] += int(dual[1])


for file in file_list:
    with open(file,"rt") as f:
        lines = f.read().splitlines()
        this_box_cnt = lines[0]
        this_class_cnt = lines[2]
        readdict(box_cnt, lines[0])
        readdict(class_cnt, lines[-1], 1)

   
#print(box_cnt)
#print(class_cnt)
img_box_count_mp = []
for k,v in box_cnt.items():
    for _ in range(v):
        img_box_count_mp.append(k)

bc_analysis = pd.DataFrame(img_box_count_mp)
print(bc_analysis.describe())

plt.bar(box_cnt.keys(),box_cnt.values())
plt.draw()
plt.savefig("box_count_bar.png")
plt.close()

sorted_class = sorted(class_cnt.items(), key=lambda x: x[1], reverse=True)
all_cases = sum(map(lambda x:x[1], sorted_class))
plot_class = [sorted_class[i] for i in range(10)]
others = all_cases - sum(map(lambda x:x[1], plot_class))
plot_class.append(("Others", others))


plt.pie([float(v[1]) for v in plot_class], labels=[v[0] for v in plot_class],
           autopct=None)

plt.draw()
plt.savefig("class_count_pi.png")
plt.close()
