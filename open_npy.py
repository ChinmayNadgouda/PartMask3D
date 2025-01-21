import numpy as np
data =  np.load('/home/gokul/ConceptGraphs/concept-graphs/conceptgraph/Mask3D/data/processed/scannet200/validation/cf67ee02-2498-44a6-a534-fe6c970c6298.npy')
print(np.unique(data[:,-2]))
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