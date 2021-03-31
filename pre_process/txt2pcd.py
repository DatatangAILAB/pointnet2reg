import numpy as np
from pypcd import pypcd
import sys

assert (len(sys.argv)==3)

with open(sys.argv[1],"r") as f_txt:
  lines=f_txt.readlines()
  pcs=[]
  for line in lines:
    pcs.append([float(i) for i in line.strip().split()])

md_new = {'version': .7,
  'fields': ['x', 'y', 'z'],
  'size': [4, 4, 4],
  'type': ['F', 'F', 'F'],
  'count': [1, 1, 1],
  'width': len(pcs),
  'height': 1,
  'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  'points': len(pcs),
  'data': 'ascii'}

pc_new = pypcd.PointCloud(md_new, np.asarray(pcs))
pc_new.save_pcd(sys.argv[2])


