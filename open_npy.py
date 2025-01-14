import numpy as np
data =  np.load('/home/gokul/Mask3D/data/processed/scannet200/train/0a07bf65-9173-464c-8a76-eadc1652fe6b.npy')
print(data[1,:])
exit()
import json 
count = 0
with open('data/raw/scene0007_00/scene0007_00.aggregation.json') as file:
    data = json.load(file)
for seg in data['segGroups']:
    count = count + len(seg['segments'])

print(count)

file_segs = 'data/raw/scene0007_00/scene0007_00_vh_clean_2.0.010000.segs.json'

with open(file_segs) as file:
    data = json.load(file)
uni_segs = np.unique(np.asarray(data['segIndices']))

print(len(uni_segs))