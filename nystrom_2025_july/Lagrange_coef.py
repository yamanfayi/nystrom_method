# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 00:27:58 2025

@author: Faik
"""

import numpy as np 
def Lagrange_x_coefs(x1,y1,x2,y2,x3,y3,x4,y4):
    constant_x_L1= -x2*x3*x4/( (x1-x2)*(x1-x3)*(x1-x4)             )
    x_L1=    (x2*x3+x3*x4+x3*x4)/( (x1-x2)*(x1-x3)*(x1-x4)             )
    x2_L1=(-x4-x3-x2)/( (x1-x2)*(x1-x3)*(x1-x4)             )
    x3_L1=1/( (x1-x2)*(x1-x3)*(x1-x4)             )
    constant_y_L1= -y2*y3*y4/( (y1-y2)*(y1-y3)*(y1-y4)             )
    y_L1=    (y2*y3+y3*y4+y3*y4)/( (y1-y2)*(y1-y3)*(y1-y4)             )
    y2_L1=(-y4-y3-y2)/( (y1-y2)*(y1-y3)*(y1-y4)             )
    y3_L1=1/( (y1-y2)*(y1-y3)*(y1-y4)             )
    constant=constant_x_L1*constant_y_L1
    x=x_L1*constant_y_L1
    y=constant_x_L1*y_L1
    xy=x_L1*y_L1
    x2=constant_y_L1*x2_L1
    y2=constant_x_L1*y2_L1
    x3y3= y3_L1* x3_L1
    
    return  constant,x,y,xy

x1 = 0.0151834 
y1 = 0.00770954
x2 = 0.0407863 
y2 = 0.007144
x3 = 0.0299534 
y3 = 0.0287724
x4 = 0.0525265 
y4 = 0.0266618

res=Lagrange_x_coefs(x1, y1, x2, y2, x3, y3, x4, y4)
print(res)