# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 21:28:02 2021

@author: han1h
"""

file = r"C:\Users\han1h\Desktop\output-predictions-zero-shot.txt"
file = r"C:\Users\han1h\Desktop\output-predictions-finetune.txt"

with open(file, 'r') as f:
    output = f.read()

output = output.split("\n\n\n\n\n")[:-1]

ans = []
for out in output:
    temp = ''
    outsplit = out.split("---------\n")[1:]
    temp += (outsplit[0].split("\n\n")[1] + "\n" + outsplit[0].split("\n\n")[-2] + "\n")
    temp += (outsplit[1] + "\n")
    temp += (outsplit[4] + "\n\n")
    temp += (outsplit[4].split("\n")[1].split(": ")[1][0] + "\n")
    ans.append(temp)
    
cnt = 0
wr = []
for a in ans:
    if a[-2] == 'F':
        wr.append(a)
        
sb = 0
sa = 0
eb = 0
ea = 0
for w in wr:
    a = w.split("\n")[3][11:19]
    if a == 'starts b':
        sb += 1
    if a == 'starts a':
        sa += 1
    if a == 'ends bef':
        eb += 1
    if a == 'ends aft':
        ea += 1
        

        
ground_truth_file = r"C:\Users\han1h\Desktop\11711\event-process-tracie\data\iid\tracie_test.txt"

with open(ground_truth_file, 'r') as f:
    g = f.read()
    
g = g.split("\n")[:-1]

baseline_file = r"C:\Users\han1h\Desktop\eval_results_lm.txt"
with open(baseline_file, 'r') as f:
    b = f.read()
b = b.split("\n")[:-1]
bb = []
for line in b:
    bb.append(line.split(" a")[0])

for i in range(4248):
    if g[i].split("answer: ")[1][0] != bb[i].split(": ")[1][0]:
        cnt += 1
        
useful_case = []
for i in range(4248):
    if g[i].split("answer: ")[1][0] != bb[i].split(": ")[1][0] and ans[i][-2] == 'T':
        useful_case.append(ans[i])
        