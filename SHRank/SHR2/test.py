#coding=utf-8

# file = open("filename.txt")
# while 1:
#     lines = file.readlines()
#     if not lines:
#         break
#     for line in lines:
#         print(line)
#
# file.close()

import numpy as np

A=np.asarray([[0., 0.,0.,0.525,0., 0.,0., 0.,   0.,   0.   ],
 [0.,   0.,   0.,   0.,   1.,   0.,   0.,   0.,   0.,   0.   ],
 [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.   ],
 [1.,   0.,   0.,   0.,   0.,   0.5,  0.,   0.,   0.,   0.   ],
 [0.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.   ],
 [0.,   0.,   0.,   0.475,0.,   0.,   0.,   0.,   0.,   1.   ],
 [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.   ],
 [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.   ],
 [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.   ],
 [0.,   0.,   0.,   0.,   0.,   0.5,  0.,   0.,   0.,   0.   ]])