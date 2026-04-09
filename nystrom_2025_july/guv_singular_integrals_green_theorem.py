# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 14:03:00 2025

@author: Faik
"""
#This code calculates guu compoınents of  singular integrals in the EFIE with Greens Theorem
# singular integrals of guu=(g+ (1/k^2)diff[g,u,2])
#Firstly, taylor series expansion is applied to guu
#The result will be (1/(4pi) )(  term1+term2+term3+term4+ term 5+ series 1 +series 2           )
#term1=  (u^2+0.5v^2)/r3  weakly singular
#term2=  (2u^2 -v^2)/(k^2  r^5)   hyper singular
#term3= k(-2j/3)
#term4=  -k^2 r/3
#term 5= j k^3 v^2  /6
#series 1=  (from n=4 to infinity )  (- j k )^n r^(n-1)  /  n!
#series2=  (from n=4 to infinity ) (  (-j r)^n  k^(n-2)  / n!    )* ( (2u^2-v^2)/r^5  + ik  (2u^2-v^2)/r^4   - k^2  u^2/r^3   )
#(series 1 n=4)  + (series 1 n=5)=  (k^4  r^3  /4 !)  - j k^5 r^4 /5!
#(series 2 n=4)  + (series 2 n=5)= (k^2/4!)((2u^2-v^2)/r)+(2u^2-v^2)(k^2 / 4!  -  jk^3/5!)+ (k^4/5!)*(-3ru^2-rv^2)
# In this code, these kernels multiplied with lagrange polynomiial  then we will integrate them
#  2x2 gauss legendre points at reference square at [-1,1]x [-1,1]
#  (x1,y1)=0.25(-1/3^0.5,-1/3^0.5),  (x2,y2)=0.25(1/3^0.5,-1/3^0.5),  (x3,y3)=0.25(1/3^0.5,1/3^0.5), (x4,y4)=0.25(-1/3^0.5,1/3^0.5)   
# The lagrange polynoms compare to each nodes are:
#L_1= 0.25*(1-sqrt(3)u)(1-sqrt(3)v)=0.25+0.75 uv - sqrt(3)u-sqrt(3)v
#L_2= 0.25*(1+sqrt(3)u)(1-sqrt(3)v)=0.25-0.75 uv + sqrt(3)u-sqrt(3)v
#L_3= 0.25*(1+sqrt(3)u)(1+sqrt(3)v)=0.25+0.75 uv + sqrt(3)u+sqrt(3)v
#L_4= 0.25*(1-sqrt(3)u)(1+sqrt(3)v)=0.25-0.75 uv - sqrt(3)u+sqrt(3)v
#IDEA is to calculate integral term L_i by using hadamart part
# For each term we have 4 types of kernel :   kernel,  u*kernel, v*kernel,u*v*kernel
# for example :  term1 , u*term1, v*term1, u*v*term1
import numpy as np
def guu_term2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            du = u1 - u2
            dv = v1 - v2
            du2_dv2 = du**2 + dv**2
            sqrt_term1 = np.sqrt(u1**2 + v1**2 - 2 * t * (u1**2 - u1*u2 + v1*dv) +t**2 * (u1**2 - 2*u1*u2 + u2**2 + dv**2))
            arctanh_arg = (u1**2 - u1*u2 - t*(u1**2 - 2*u1*u2 + u2**2 + dv**2) + v1*dv) / (sqrt_term1 * np.sqrt(du2_dv2))
            sqrt_term2 = np.sqrt((t-1)**2*u1**2 - 2*(t-1)*t*u1*u2 + v1**2 +t**2*(u2**2 + dv**2) - 2*t*v1*dv)
            sqrt_term3 = np.sqrt(u1**2 - 2*t*u1**2 + t**2*u1**2 + 2*t*u1*u2 - 2*t**2*u1*u2 +t**2*u2**2 + v1**2 - 2*t*v1**2 + t**2*v1**2 + 2*t*v1*v2 -2*t**2*v1*v2 + t**2*v2**2)
            sqrt_uv_diff = np.sqrt(du2_dv2)
            result = (t + du * sqrt_term1 / du2_dv2- du * (u1**2 - u1*u2 + v1*dv) * np.arctanh(arctanh_arg) / du2_dv2**(3/2)+ u1 * np.arctanh(arctanh_arg) / sqrt_uv_diff+ 0.5 * ((-du * sqrt_term2 / du2_dv2)+ (dv * (-u2*v1 + u1*v2) * np.arctanh(arctanh_arg) / du2_dv2**(3/2)))+ (2 * v1 * np.log(np.abs(-v1 + t*v1 - t*v2)) / dv)- t * np.log(np.abs(-u1 + t*u1 - t*u2 + sqrt_term3))- v1 * np.log(np.abs(u1 - t*u1 + t*u2 + sqrt_term3)) / dv+ ((u2*v1 - u1*v2) * np.log(np.abs(-u1**2 + t*u1**2 + u1*u2 - 2*t*u1*u2 + t*u2**2 - v1**2+ t*v1**2 + v1*v2 - 2*t*v1*v2 + t*v2**2 + sqrt_uv_diff * sqrt_term3)) / (dv * sqrt_uv_diff)))
            return result*(v2-v1)
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))
def guu_xterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            du = u1 - u2
            dv = v1 - v2
            du2_dv2 = du**2 + dv**2
            sqrt_common = np.sqrt(u1**2 + v1**2 - 2 * t * (u1**2 - u1 * u2 + v1 * dv) + t**2 * (u1**2 - 2 * u1 * u2 + u2**2 + dv**2))
            sqrt_diff = np.sqrt(du2_dv2)
            sqrt_full = np.sqrt((1 - 2 * t + t**2) * u1**2 + 2 * (1 - t) * t * u1 * u2 + v1**2 +t * v1 * (-2 * v1 + 2 * v2) + t**2 * (u2**2 + v1**2 - 2 * v1 * v2 + v2**2))
            sqrt_expr = np.sqrt((-1 + t)**2 * u1**2 - 2 * (-1 + t) * t * u1 * u2 + v1**2 + t**2 * (u2**2 + dv**2) - 2 * t * v1 * dv)
            log_arg1 = (u1**2 - u1 * u2 - t * (u1**2 - 2 * u1 * u2 + u2**2 + dv**2) + v1 * dv) / (sqrt_expr * sqrt_diff)
            log_arg2 = ((-1 + t) * u1**2 + u1 * (u2 - 2 * t * u2) + t * (u2**2 + dv**2) + v1 * (-v1 + v2)) / (sqrt_expr * sqrt_diff)
            log_expr1 = np.log(1 + log_arg1)
            log_expr2 = np.log(1 + log_arg2)
            log_expr3 = np.log(1 - log_arg2)
            common_denom = 1 / (2 * du2_dv2**(5 / 2))
            term1 = -4 * (u1**2 - u1 * u2 + 2 * v1 * dv) * sqrt_expr * du2_dv2**(3 / 2)
            term2 = t * sqrt_expr * du2_dv2**(3 / 2) * (u1**2 - 2 * u1 * u2 + u2**2 + 2 * dv**2)
            term3 = 2 * (u1**2 - u1 * u2 + v1 * dv) * (u1**2 - u1 * u2 + 2 * v1 * dv) * du2_dv2 * (log_expr1 - log_expr2)
            term4 = - (u1**2 + 2 * v1**2) * du2_dv2**2 * (log_expr1 - log_expr2)
            term5_inner = 3 * (u1**2 - u1 * u2 + v1 * dv) * sqrt_expr * sqrt_diff +  0.5 * (2 * u1**4 - 4 * u1**3 * u2 + v1**2 * (-u2**2 + 2 * dv**2) +  2 * u1 * u2 * v1 * (-2 * v1 + 3 * v2) +u1**2 * (2 * u2**2 + 4 * v1**2 - 4 * v1 * v2 - v2**2)) * (-log_expr3 + log_expr2)
            term5 = (u1**2 - 2 * u1 * u2 + u2**2 + 2 * dv**2) * term5_inner
            term6_inner = -4 * v1 * sqrt_expr * du2_dv2**(3 / 2) * dv +  t * sqrt_expr * du2_dv2**(3 / 2) * dv**2 -v1**2 * du2_dv2**2 * (log_expr1 - log_expr2) +  2 * v1 * (u1**2 - u1 * u2 + v1 * dv) * du2_dv2 * dv * (log_expr1 - log_expr2)
            term6_inner2 = 3 * (u1**2 - u1 * u2 + v1 * dv) * sqrt_expr * sqrt_diff + 0.5 * (2 * u1**4 - 4 * u1**3 * u2 + v1**2 * (-u2**2 + 2 * dv**2) +     2 * u1 * u2 * v1 * (-2 * v1 + 3 * v2) +    u1**2 * (2 * u2**2 + 4 * v1**2 - 4 * v1 * v2 - v2**2)) * (-log_expr3 + log_expr2)
            term6 = (v1 - v2)**2 * term6_inner2
            result = common_denom * (term1 + term2 + term3 + term4 + term5 - 0.5 * (term6_inner + term6))
            return result*(v2-v1)
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+p_x*guu_term1(x1, y1, x2, y2, x3, y3, x4, y4,0, 0)
def guu_yterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
           delta_u = u1 - u2
           delta_v = v1 - v2
           D = u1**2 - 2 * u1 * u2 + u2**2 + delta_v**2
           sqrtD = np.sqrt(D)
           sqrt_term = np.sqrt((u1 - t * u1 + t * u2)**2 + (v1 - t * v1 + t * v2)**2)
           common_den = (D**(5/2)) * delta_v
           inner_sqrt = np.sqrt(((-1 + t)**2) * u1**2 - 2 * (-1 + t) * t * u1 * u2 + v1**2 + t**2 * (u2**2 + delta_v**2) - 2 * t * v1 * delta_v)
           log_arg1 = 1 + (u1**2 - u1 * u2 - t * D + v1 * delta_v) / (inner_sqrt * sqrtD)
           log_arg2 = 1 + ((-1 + t) * u1**2 + u1 * (u2 - 2 * t * u2) + t * (u2**2 + delta_v**2) + v1 * (-v1 + v2)) / (inner_sqrt * sqrtD)
           log_arg3 = -1 * ((-1 + t) * u1**2 + u1 * (u2 - 2 * t * u2) + t * (u2**2 + delta_v**2) + v1 * (-v1 + v2)) / (inner_sqrt * sqrtD)
           result = (0.25 / common_den) * (delta_v * (t * delta_u * D**(3/2) * delta_v * sqrt_term +2 * D**(3/2) * (u2 * v1 + u1 * (-2 * v1 + v2)) * sqrt_term -u1 * v1 * D**2 * (np.log(np.abs(log_arg1)) - np.log(np.abs(log_arg2))) +(u1**2 - u1 * u2 + v1 * delta_v) * D * (2 * u1 * v1 - u2 * v1 - u1 * v2) * (np.log(np.abs(log_arg1)) - np.log(np.abs(log_arg2))) +delta_u * delta_v * ( 3 * (u1**2 - u1 * u2 + v1 * delta_v) * sqrtD * sqrt_term +0.5 * (2 * u1**4 - 4 * u1**3 * u2 + v1**2 * (-u2**2 + 2 * delta_v**2) +2 * u1 * u2 * v1 * (-2 * v1 + 3 * v2) +u1**2 * (2 * u2**2 + 4 * v1**2 - 4 * v1 * v2 - v2**2)) * (-np.log(np.abs(1 - log_arg3)) + np.log(np.abs(log_arg2))))) +(2 * t * v1 * D**(5/2) * delta_v -t**2 * D**(5/2) * delta_v**2 +2 * sqrtD * delta_v * sqrt_term * (t * u2 * delta_v**3 + u2**3 * ((-3 + t) * v1 - t * v2) +u1**3 * (v1 - t * v1 + (2 + t) * v2) +u1**2 * u2 * ((-5 + 3 * t) * v1 - (4 + 3 * t) * v2) +u1 * (-(-1 + t) * delta_v**3 + u2**2 * ((7 - 3 * t) * v1 + (2 + 3 * t) * v2))) +4 * v1**2 * D**(5/2) * np.log(np.abs((-1 + t) * v1 - t * v2)) +2 * t * D**(5/2) * delta_v * ((-2 + t) * v1 - t * v2) * np.log(np.abs(-u1 + t * u1 - t * u2 + sqrt_term)) -2 * v1**2 * D**(5/2) * np.log(np.abs(u1 - t * u1 + t * u2 + sqrt_term)) -2 * delta_u * (D - 2 * delta_v**2) * (u2 * v1 - u1 * v2)**2 * np.log(np.abs(-u1**2 + t * u1**2 + u1 * u2 - 2 * t * u1 * u2 + t * u2**2 -v1**2 + t * v1**2 + v1 * v2 - 2 * t * v1 * v2 + t * v2**2 +sqrtD * sqrt_term))))
           return result*(v2-v1)
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+p_y*guu_term1(x1, y1, x2, y2, x3, y3, x4, y4, 0,0)
def guu_xyterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
          delta_v = v1 - v2
          D = u1**2 - 2*u1*u2 + u2**2 + delta_v**2
          sqrtD = np.sqrt(D)
          inner = np.sqrt(  (-1 + t)**2 * u1**2- 2 * (-1 + t) * t * u1 * u2  + v1**2+ t**2 * (u2**2 + delta_v**2)  - 2 * t * v1 * delta_v)

    # Log arguments with absolute value
          A = 1 + (u1**2 - u1*u2 - t*D + v1*delta_v) / (inner * sqrtD)
          B = 1 + ((-1 + t) * u1**2 + u1*(u2 - 2*t*u2) + t*(u2**2 + delta_v**2) + v1*(-v1 + v2)) / (inner * sqrtD)
          logA = np.log(np.abs(A))
          logB = np.log(np.abs(B))
          term1 = -1/3 * t**2 * inner * D**(5/2) * (D + delta_v**2) * delta_v
          term2 = 0.5 * t * inner * D**(5/2) * (v1 * (u2**2 + 6*delta_v**2)+ u1**2 * (3*v1 - 2*v2)+ 2*u1*u2 * (-2*v1 + v2)  )
          term3 = inner * D**(5/2) * (2*u1*u2*v1  + u1**2 * (-3*v1 + v2)    + 6*v1**2 * (-v1 + v2))
          term4 = 0.25 * D * (  v1 * (u2**2 + 6*delta_v**2)+ u1**2 * (3*v1 - 2*v2)+ 2*u1*u2 * (-2*v1 + v2)) * (     6 * (u1**2 - u1*u2 + v1*delta_v) * inner * sqrtD- (2*u1**4 - 4*u1**3*u2 + v1**2 * (-u2**2 + 2*delta_v**2)+ 2*u1*u2*v1 * (-2*v1 + 3*v2)  + u1**2 * (2*u2**2 + 4*v1**2 - 4*v1*v2 - v2**2)) * (logA - logB))
          term5 = -1/12 * (D + delta_v**2) * delta_v * (  2*(15*(u1**2 - u1*u2 + v1*delta_v)**2- 4*(u1**2 + v1**2)*D  + 5*t*(u1**2 - u1*u2 + v1*delta_v)*D  ) * inner * sqrtD - 3*(2*u1**6 - 6*u1**5*u2      + v1**3 * (-3*u2**2 + 2*delta_v**2) * delta_v  + 3*u1*u2*v1**2 * (u2**2 - 2*v1**2 + 6*v1*v2 - 4*v2**2)    + u1**4 * (6*u2**2 + 6*v1**2 - 6*v1*v2 - 3*v2**2)+ 3*u1**2*v1 * (2*v1**3 + u2**2*(v1 - 4*v2)- 4*v1**2*v2 + v1*v2**2 + v2**3)  + u1**3 * (-2*u2**3 + 3*u2*(-4*v1**2 + 6*v1*v2 + v2**2))  ) * (logA - logB))
          term6 = -0.5 * v1 * (u1**2 + 2*v1**2) * D**3 * (logA - logB)
          term7 = -0.5 * (u1**2 - u1*u2 + v1*delta_v) * D**2 * (    2*u1*u2*v1  + u1**2 * (-3*v1 + v2)  + 6*v1**2 * (-v1 + v2)) * (logA - logB)
          return ((term1 + term2 + term3 + term4 + term5 + term6 + term7) / D**(7/2))*(v2-v1)
    return ((f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0)))+ p_x* p_y*guu_term1(x1, y1, x2, y2, x3, y3, x4, y4, 0,0)+p_y*guu_xterm1(x1, y1, x2, y2, x3, y3, x4, y4, 0,0)+p_x*guu_yterm1(x1, y1, x2, y2, x3, y3, x4, y4, 0,0)


def guu_term1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            num = v1 - t * v1 + t * v2
            denom1 = (-1 + t)**2 * u1**2 - 2 * (-1 + t) * t * u1 * u2 + v1**2 + t**2 * (u2**2 + (v1 - v2)**2) - 2 * t * v1 * (v1 - v2)
            denom2 = -u2 * v1 + u1 * v2
            denom = np.sqrt(denom1) * denom2
            return (-num / denom)*(v2-v1)
    
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))

def guu_xterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            D1 = (-1 + t)**2 * u1**2 - 2 * (-1 + t) * t * u1 * u2 + v1**2 + t**2 * (u2**2 + (v1 - v2)**2) - 2 * t * v1 * (v1 - v2)
            D2 = u1**2 - 2 * u1 * u2 + u2**2 + (v1 - v2)**2
            sqrtD1 = np.sqrt(np.abs(D1))
            sqrtD2 = np.sqrt(np.abs(D2))
            term1_num = (-1 + t) * u1**2 + u1 * (u2 - 2 * t * u2) + t * (u2**2 - (v1 - v2)**2) + v1 * (v1 - v2)
            term1_den = sqrtD1 * D2
            term1 = term1_num / term1_den
            term2_num = (2 * u1**2 - 4 * u1 * u2 + 2 * u2**2 + (v1 - v2)**2)
            term2_den = D2 ** (1.5)
            log_arg = -u1**2 + t * u1**2 + u1 * u2 - 2 * t * u1 * u2 + t * u2**2 - v1**2 + t * v1**2 + sqrtD1 * sqrtD2 + v1 * v2 - 2 * t * v1 * v2 + t * v2**2
            term2 = (1 / term2_den) * term2_num * np.log(np.abs(log_arg))
            return (v2-v1)*(term1 - term2)
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+p_x*guu_term2(x1,y1,x2,y2,x3,y3,x4,y4,0,0)
def guu_yterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            D1 = (-1 + t)**2 * u1**2 - 2 * (-1 + t) * t * u1 * u2 + v1**2 + t**2 * (u2**2 + (v1 - v2)**2) - 2 * t * v1 * (v1 - v2)
            D2 = u1**2 - 2 * u1 * u2 + u2**2 + (v1 - v2)**2
            sqrtD1 = np.sqrt(np.abs(D1))
            sqrtD2 = np.sqrt(np.abs(D2))
            term1_num = u2 * ((-1 + 2 * t) * v1 - 2 * t * v2) - u1 * (2 * (-1 + t) * v1 + v2 - 2 * t * v2)
            term1_den = sqrtD1 * D2
            term1 = - term1_num / term1_den
            term2_num = (u1 - u2) * (v1 - v2)
            term2_den = D2 ** (1.5)
            log_arg = -u1**2 + t * u1**2 + u1 * u2 - 2 * t * u1 * u2 + t * u2**2 - v1**2 + t * v1**2 + sqrtD1 * sqrtD2 + v1 * v2 - 2 * t * v1 * v2 + t * v2**2
            term2 = (1 / term2_den) * term2_num * np.log(np.abs(log_arg))
            return (v2-v1)*(term1 - term2)
    
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+p_y*guu_term2(x1,y1,x2,y2,x3,y3,x4,y4,0,0)

def guu_xyterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            delta_u = u1 - u2
            delta_v = v1 - v2
            D = u1**2 - 2 * u1 * u2 + u2**2 + delta_v**2
            sqrtD = np.sqrt(D)
            inner_sqrt = np.sqrt((-1 + t)**2 * u1**2 - 2 * (-1 + t) * t * u1 * u2 + v1**2 + t**2 * (u2**2 + delta_v**2) - 2 * t * v1 * delta_v)
            
            log_a = np.log(np.abs(1 + (u1**2 - u1 * u2 - t * D + v1 * delta_v) / (inner_sqrt * sqrtD)))
            log_b = np.log(np.abs(1 + ((-1 + t) * u1**2 + u1 * (u2 - 2 * t * u2) + t * (u2**2 + delta_v**2) + v1 * (-v1 + v2)) / (inner_sqrt * sqrtD)))
            log_c = np.log(np.abs(1 - ((-1 + t) * u1**2 + u1 * (u2 - 2 * t * u2) + t * (u2**2 + delta_v**2) + v1 * (-v1 + v2)) / (inner_sqrt * sqrtD)))

            term1 = -1/3 * t**2 * inner_sqrt * D**(5/2) * (D + delta_v**2) * delta_v
            term2 = 0.5 * t * inner_sqrt * D**(5/2) * (v1 * (u2**2 + 6 * delta_v**2) + u1**2 * (3 * v1 - 2 * v2) + 2 * u1 * u2 * (-2 * v1 + v2))
            term3 = inner_sqrt * D**(5/2) * (2 * u1 * u2 * v1 + u1**2 * (-3 * v1 + v2) + 6 * v1**2 * (-v1 + v2))

            term4_inner = 6 * (u1**2 - u1 * u2 + v1 * delta_v) * inner_sqrt * sqrtD - \
                  (2 * u1**4 - 4 * u1**3 * u2 + v1**2 * (-u2**2 + 2 * delta_v**2) + 2 * u1 * u2 * v1 * (-2 * v1 + 3 * v2) +
                   u1**2 * (2 * u2**2 + 4 * v1**2 - 4 * v1 * v2 - v2**2)) * (log_a - log_b)
            term4 = 0.25 * D * (v1 * (u2**2 + 6 * delta_v**2) + u1**2 * (3 * v1 - 2 * v2) + 2 * u1 * u2 * (-2 * v1 + v2)) * term4_inner

            term5_inner = 2 * (15 * (u1**2 - u1 * u2 + v1 * delta_v)**2 - 4 * (u1**2 + v1**2) * D + 5 * t * (u1**2 - u1 * u2 + v1 * delta_v) * D) * inner_sqrt * sqrtD - \
                  3 * (2 * u1**6 - 6 * u1**5 * u2 + v1**3 * (-3 * u2**2 + 2 * delta_v**2) * delta_v + 3 * u1 * u2 * v1**2 * (u2**2 - 2 * v1**2 + 6 * v1 * v2 - 4 * v2**2) + \
                  u1**4 * (6 * u2**2 + 6 * v1**2 - 6 * v1 * v2 - 3 * v2**2) + 3 * u1**2 * v1 * (2 * v1**3 + u2**2 * (v1 - 4 * v2) - 4 * v1**2 * v2 + v1 * v2**2 + v2**3) + \
                  u1**3 * (-2 * u2**3 + 3 * u2 * (-4 * v1**2 + 6 * v1 * v2 + v2**2))) * (log_a - log_b)
            term5 = -1/12 * (D + delta_v**2) * delta_v * term5_inner

            term6 = -0.5 * v1 * (u1**2 + 2 * v1**2) * D**3 * (log_a - log_b)
            term7 = -0.5 * (u1**2 - u1 * u2 + v1 * delta_v) * D**2 * (2 * u1 * u2 * v1 + u1**2 * (-3 * v1 + v2) + 6 * v1**2 * (-v1 + v2)) * (log_a - log_b)

            return ((term1 + term2 + term3 + term4 + term5 + term6 + term7) / D**(7/2))*(v2-v1)
    
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+p_x*p_y*guu_term2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y)+p_x*guu_yterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y)+p_y*guu_xterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y)

def guu_term3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    return 0.5*np.abs(x1*y2+x2*y3+x3*y4+x4*y1-(y1*x2+y2*x3+y3*x4+y4*x1)     )
def guu_xterm3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            return (v2-v1)*0.5*( (1/3)*t**3 *(u1-u2)**2-t**2*(u1-u2)*(u1+p_x) +t*u1*(u1+2*p_x)   )
    
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))
def guu_yterm3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            return (v2-v1)*(1/6)*t*(2*t**2*(u1-u2)*(v1-v2)+6*u1*(v1+p_y)+3*t*( u1*(-2*v1-p_y+v2)+u2*(v1+p_y)  )     )
    
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))
def guu_xyterm3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    xs=p_x
    ys=p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            return (t*(-3*t**3*(u1-u2)**2*(v1-v2)+12*u1*(u1+2*xs)*(v1+ys)+4*t**2*(u1-u2)*(2*(v1-v2)*xs-u2*(v1+ys)+u1*(3*v1-2*v2+ys))+6*t*(u1**2*(-3*v1+v2-2*ys)+2*u2*xs*(v1+ys)+2*u1*(xs*(-2*v1+v2-ys)+u2*(v1+ys))))/24)*(v2-v1)
    
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))

def guu_term4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.339981  )   *0.5
    p2=(   1+ 0.339981  )   *0.5
    p3=(    1-0.861136   )  *0.5
    p4=(    1+0.861136   )  *0.5
    w1=0.652145
    w2=0.347855
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            x= u1+(u2-u1)*t-p_x
            y=v1+(v2-v1)*t-p_y
            r=(x**2+y**2)**0.5
            return    (0.5*x*r-0.5*y**2*np.log(r-x))    *0.5*    (v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )
def guu_xterm4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.339981  )   *0.5
    p2=(   1+ 0.339981  )   *0.5
    p3=(    1-0.861136   )  *0.5
    p4=(    1+0.861136   )  *0.5
    w1=0.652145
    w2=0.347855
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            u= u1+(u2-u1)*t
            v=v1+(v2-v1)*t
            r2 = u**2 + v**2 - 2*u*p_x + p_x**2 - 2*v*p_y + p_y**2
            r=r2**0.5
            return    ((1/6)*r*( 2*u**2 + 2*v**2 - p_x*u - p_x**2 - 4*v*p_y + 2*p_y**2 )-0.5*p_x*(v - p_y)**2 * np.log( np.abs(-u + p_x + r) ))*(v2-v1)*0.5
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )
def guu_yterm4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.339981  )   *0.5
    p2=(   1+ 0.339981  )   *0.5
    p3=(    1-0.861136   )  *0.5
    p4=(    1+0.861136   )  *0.5
    w1=0.652145
    w2=0.347855
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            u = u1 + t*(u2 - u1)
            v = v1 + t*(v2 - v1)
            R2 = u**2 + v**2 - 2*u*xs + xs**2 - 2*v*ys + ys**2
            R  = np.sqrt(R2)
            return (v * (0.5*(u - xs)*R - 0.5*(v - ys)**2 * np.log(np.abs(-u + xs + R))))         *(v2-v1)*0.5
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )

def guu_xyterm4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.339981  )   *0.5
    p2=(   1+ 0.339981  )   *0.5
    p3=(    1-0.861136   )  *0.5
    p4=(    1+0.861136   )  *0.5
    w1=0.652145
    w2=0.347855
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            u = u1 + t*(u2 - u1)
            v = v1 + t*(v2 - v1)
            R2 = u**2 + v**2 - 2*u*xs + xs**2 - 2*v*ys + ys**2
            R  = np.sqrt(R2)
            termA = (1/6)*R*( 2*u**2 + 2*v**2 - u*xs - xs**2 - 4*v*ys + 2*ys**2 )
            termB = 0.5*xs*(v - ys)**2 * np.log(np.abs(-u + xs + R))

            return v * (termA - termB) *(v2-v1)*0.5
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )
def guu_term5(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.339981  )   *0.5
    p2=(   1+ 0.339981  )   *0.5
    p3=(    1-0.861136   )  *0.5
    p4=(    1+0.861136   )  *0.5
    w1=0.652145
    w2=0.347855
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            u = u1 + t*(u2 - u1)
            v = v1 + t*(v2 - v1)
            R2 = u**2 + v**2 - 2*u*xs + xs**2 - 2*v*ys + ys**2
            R  = np.sqrt(R2)
            termA = (1/6)*R*( 2*u**2 + 2*v**2 - u*xs - xs**2 - 4*v*ys + 2*ys**2 )
            termB = 0.5*xs*(v - ys)**2 * np.log(np.abs(-u + xs + R))

            return u*(v-ys)**2 *(v2-v1)*0.5
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )

def guu_xterm5(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.339981  )   *0.5
    p2=(   1+ 0.339981  )   *0.5
    p3=(    1-0.861136   )  *0.5
    p4=(    1+0.861136   )  *0.5
    w1=0.652145
    w2=0.347855
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            u = u1 + t*(u2 - u1)
            v = v1 + t*(v2 - v1)
            R2 = u**2 + v**2 - 2*u*xs + xs**2 - 2*v*ys + ys**2
            R  = np.sqrt(R2)
            termA = (1/6)*R*( 2*u**2 + 2*v**2 - u*xs - xs**2 - 4*v*ys + 2*ys**2 )
            termB = 0.5*xs*(v - ys)**2 * np.log(np.abs(-u + xs + R))

            return 0.5*u**2*(v-ys)**2  *  0.5* (v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )
def guu_yterm5(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.339981  )   *0.5
    p2=(   1+ 0.339981  )   *0.5
    p3=(    1-0.861136   )  *0.5
    p4=(    1+0.861136   )  *0.5
    w1=0.652145
    w2=0.347855
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            u = u1 + t*(u2 - u1)
            v = v1 + t*(v2 - v1)
            R2 = u**2 + v**2 - 2*u*xs + xs**2 - 2*v*ys + ys**2
            R  = np.sqrt(R2)
            termA = (1/6)*R*( 2*u**2 + 2*v**2 - u*xs - xs**2 - 4*v*ys + 2*ys**2 )
            termB = 0.5*xs*(v - ys)**2 * np.log(np.abs(-u + xs + R))

            return v*u*(v-ys)**2  *  0.5* (v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )
def guu_xyterm5(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.339981  )   *0.5
    p2=(   1+ 0.339981  )   *0.5
    p3=(    1-0.861136   )  *0.5
    p4=(    1+0.861136   )  *0.5
    w1=0.652145
    w2=0.347855
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            u = u1 + t*(u2 - u1)
            v = v1 + t*(v2 - v1)
            R2 = u**2 + v**2 - 2*u*xs + xs**2 - 2*v*ys + ys**2
            R  = np.sqrt(R2)
            termA = (1/6)*R*( 2*u**2 + 2*v**2 - u*xs - xs**2 - 4*v*ys + 2*ys**2 )
            termB = 0.5*xs*(v - ys)**2 * np.log(np.abs(-u + xs + R))

            return 0.5*v*u**2*(v-ys)**2  *  0.5* (v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )
def guu_r3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.538469  )   *0.5
    p2=(   1+ 0.538469  )   *0.5
    p3=(    1-0.90618   )  *0.5
    p4=(    1+0.90618   )  *0.5
    p5=0.5
    w5=128/225
    w1=0.478629
    w2=w1
    w3=0.236927
    w4=w3
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            u = u1 + t*(u2 - u1)
            v = v1 + t*(v2 - v1)
            R2 = u**2 + v**2 
            R  = np.sqrt(R2)

            return ( (1/8)*u*R*(2*u**2+5*v**2 )-    (3/8)*v**4*np.log(  R-u  )        )        *(v2-v1)*0.5
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )
def guu_xr3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.538469  )   *0.5
    p2=(   1+ 0.538469  )   *0.5
    p3=(    1-0.90618   )  *0.5
    p4=(    1+0.90618   )  *0.5
    p5=0.5
    w5=128/225
    w1=0.478629
    w2=w1
    w3=0.236927
    w4=w3
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            return (0.2*np.sqrt((u1+t*(u2-u1))**2+(v1+t*(v2-v1))**2-2*(u1+t*(u2-u1))*xs+xs**2-2*(v1+t*(v2-v1))*ys+ys**2) * ( (u1+t*(u2-u1))**4 + (v1+t*(v2-v1))**4 - 2.75*(u1+t*(u2-u1))**3*xs - 0.25*xs**4 - 4*(v1+t*(v2-v1))**3*ys + 6*(v1+t*(v2-v1))**2*ys**2 - 4*(v1+t*(v2-v1))*ys**3 + ys**4 + xs**2*(-1.125*(v1+t*(v2-v1))**2 + 2.25*(v1+t*(v2-v1))*ys - 1.125*ys**2) + (u1+t*(u2-u1))*xs*(-0.875*(v1+t*(v2-v1))**2 - 0.25*xs**2 + 1.75*(v1+t*(v2-v1))*ys - 0.875*ys**2) + (u1+t*(u2-u1))**2*(2*(v1+t*(v2-v1))**2 + 2.25*xs**2 - 4*(v1+t*(v2-v1))*ys + 2*ys**2) ) - 0.375*xs*(v1+t*(v2-v1)-ys)**4 * np.log(np.abs(-(u1+t*(u2-u1))+xs+np.sqrt((u1+t*(u2-u1))**2+(v1+t*(v2-v1))**2-2*(u1+t*(u2-u1))*xs+xs**2-2*(v1+t*(v2-v1))*ys+ys**2))))*0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )
def guu_yr3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.538469  )   *0.5
    p2=(   1+ 0.538469  )   *0.5
    p3=(    1-0.90618   )  *0.5
    p4=(    1+0.90618   )  *0.5
    p5=0.5
    w5=128/225
    w1=0.478629
    w2=w1
    w3=0.236927
    w4=w3
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            return ((v1+t*(v2-v1))*(0.125*((u1+t*(u2-u1))-xs)*np.sqrt((u1+t*(u2-u1))**2+(v1+t*(v2-v1))**2-2*(u1+t*(u2-u1))*xs+xs**2-2*(v1+t*(v2-v1))*ys+ys**2)*(2*(u1+t*(u2-u1))**2+5*(v1+t*(v2-v1))**2-4*(u1+t*(u2-u1))*xs+2*xs**2-10*(v1+t*(v2-v1))*ys+5*ys**2) - 0.375*((v1+t*(v2-v1))-ys)**4*np.log(np.abs(-(u1+t*(u2-u1))+xs+np.sqrt((u1+t*(u2-u1))**2+(v1+t*(v2-v1))**2-2*(u1+t*(u2-u1))*xs+xs**2-2*(v1+t*(v2-v1))*ys+ys**2)))))*0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )
def guu_xyr3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.538469  )   *0.5
    p2=(   1+ 0.538469  )   *0.5
    p3=(    1-0.90618   )  *0.5
    p4=(    1+0.90618   )  *0.5
    p5=0.5
    w5=128/225
    w1=0.478629
    w2=w1
    w3=0.236927
    w4=w3
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            return (v1+t*(v2-v1))*(0.2*np.sqrt((u1+t*(u2-u1))**2+(v1+t*(v2-v1))**2-2*(u1+t*(u2-u1))*xs+xs**2-2*(v1+t*(v2-v1))*ys+ys**2)*((u1+t*(u2-u1))**4+(v1+t*(v2-v1))**4-2.75*(u1+t*(u2-u1))**3*xs-0.25*xs**4-4*(v1+t*(v2-v1))**3*ys+6*(v1+t*(v2-v1))**2*ys**2-4*(v1+t*(v2-v1))*ys**3+ys**4+xs**2*(-1.125*(v1+t*(v2-v1))**2+2.25*(v1+t*(v2-v1))*ys-1.125*ys**2)+(u1+t*(u2-u1))*xs*(-0.875*(v1+t*(v2-v1))**2-0.25*xs**2+1.75*(v1+t*(v2-v1))*ys-0.875*ys**2)+(u1+t*(u2-u1))**2*(2*(v1+t*(v2-v1))**2+2.25*xs**2-4*(v1+t*(v2-v1))*ys+2*ys**2)) - 0.375*xs*(v1+t*(v2-v1)-ys)**4*np.log(np.abs(-(u1+t*(u2-u1))+xs+np.sqrt((u1+t*(u2-u1))**2+(v1+t*(v2-v1))**2-2*(u1+t*(u2-u1))*xs+xs**2-2*(v1+t*(v2-v1))*ys+ys**2))))*0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )
def guu_r4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            res=(1/90.0)*(3*(u1 - t*u1 + t*u2)**6/(-u1 + u2) - 10*t**6*(u1 - u2)**3*(v1 - v2)**2 + 15*t**6*(-u1 + u2)*(v1 - v2)**4 + 18*(u1 - t*u1 + t*u2)**5*xs/(u1 - u2) - 36*t**5*(u1 - u2)**2*(v1 - v2)**2*xs - 180*t*u1**2*xs*(v1**2 + xs**2 - 2*v1*ys + ys**2) + 90*t*u1*(v1**2 + xs**2 - 2*v1*ys + ys**2)**2 + 60*t*u1**3*(v1**2 + 3*xs**2 - 2*v1*ys + ys**2) + 90*t**4*(u1 - u2)*(v1 - v2)*xs*(u1*(2*v1 - v2 - ys) + u2*(-v1 + ys)) + 12*t**5*(u1 - u2)**2*(v1 - v2)*(u1*(5*v1 - 3*v2 - 2*ys) + 2*u2*(-v1 + ys)) + 18*t**5*(v1 - v2)**3*(u1*(5*v1 - v2 - 4*ys) + 4*u2*(-v1 + ys)) - 45*t**2*(v1**2 + xs**2 - 2*v1*ys + ys**2)*(-u2*(v1**2 + xs**2 - 2*v1*ys + ys**2) + u1*(5*v1**2 - 4*v1*v2 + xs**2 - 6*v1*ys + 4*v2*ys + ys**2)) - 60*t**3*xs*(u2**2*(v1**2 + xs**2 - 2*v1*ys + ys**2) + u1**2*(6*v1**2 + v2**2 + xs**2 + 4*v2*ys + ys**2 - 6*v1*(v2 + ys)) - 2*u1*u2*(3*v1**2 + xs**2 + 2*v2*ys + ys**2 - 2*v1*(v2 + 2*ys))) - 15*t**4*(u1 - u2)*(u2**2*(v1**2 + 3*xs**2 - 2*v1*ys + ys**2) - 2*u1*u2*(4*v1**2 - 3*v1*v2 + 3*xs**2 - 5*v1*ys + 3*v2*ys + ys**2) + u1**2*(10*v1**2 + 3*v2**2 + 3*xs**2 + 6*v2*ys + ys**2 - 4*v1*(3*v2 + 2*ys))) + 180*t**2*u1*xs*(-u2*(v1**2 + xs**2 - 2*v1*ys + ys**2) + u1*(2*v1**2 + xs**2 + ys*(v2 + ys) - v1*(v2 + 3*ys))) + 20*t**3*u1*(3*u2**2*(v1**2 + 3*xs**2 - 2*v1*ys + ys**2) - 6*u1*u2*(2*v1**2 + 3*xs**2 + ys*(v2 + ys) - v1*(v2 + 3*ys)) + u1**2*(10*v1**2 + v2**2 + 9*xs**2 + 6*v2*ys + 3*ys**2 - 4*v1*(2*v2 + 3*ys))) - 45*t**4*(v1 - v2)**2*(-u2*(3*v1**2 + xs**2 - 6*v1*ys + 3*ys**2) + u1*(5*v1**2 + xs**2 + 2*v2*ys + 3*ys**2 - 2*v1*(v2 + 4*ys))) - 30*t**2*u1**2*(-3*u2*(v1**2 + 3*xs**2 - 2*v1*ys + ys**2) + u1*(5*v1**2 + 9*xs**2 + 2*v2*ys + 3*ys**2 - 2*v1*(v2 + 4*ys))) + 60*t**3*(v1 - v2)*(-2*u2*(v1 - ys)*(v1**2 + xs**2 - 2*v1*ys + ys**2) + u1*(5*v1**3 - 3*v1**2*(v2 + 4*ys) - 2*ys*(xs**2 + ys**2) - v2*(xs**2 + 3*ys**2) + 3*v1*(xs**2 + ys*(2*v2 + 3*ys)))))
            return res*(v2-v1)
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))

def guu_xr4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            res=(1/30.0)*((5/7)*t**7*(u1-u2)**2*(u1**4-4*u1**3*u2+u2**4+3*u1**2*(2*u2**2+(v1-v2)**2)-2*u1*u2*(2*u2**2+3*(v1-v2)**2)+3*u2**2*(v1-v2)**2+3*(v1-v2)**4) - (1/3)*t**6*(u1-u2)*(15*u1**5-12*u1**4*(5*u2+xs)-u1**2*(60*u2**3+72*u2**2*xs+20*(v1-v2)**2*xs+15*u2*(v1-v2)*(7*v1-4*v2-3*ys))+u1*(15*u2**4+48*u2**3*xs+40*u2*(v1-v2)**2*xs+15*u2**2*(v1-v2)*(5*v1-2*v2-3*ys)+15*(v1-v2)**3*(3*v1-v2-2*ys))-u2*(12*u2**3*xs+20*u2*(v1-v2)**2*xs+15*u2**2*(v1-v2)*(v1-ys)+30*(v1-v2)**3*(v1-ys))+3*u1**3*(30*u2**2+16*u2*xs+5*(v1-v2)*(3*v1-2*v2-ys))) + t*u1**2*(5*u1**4-24*u1**3*xs-40*u1*xs*(v1**2+xs**2-2*v1*ys+ys**2)+15*(v1**2+xs**2-2*v1*ys+ys**2)**2+15*u1**2*(v1**2+3*xs**2-2*v1*ys+ys**2)) + t**5*(15*u1**6-12*u1**5*(5*u2+2*xs)+3*u1**4*(30*u2**2+15*v1**2+6*v2**2+32*u2*xs+3*xs**2+8*v2*ys+ys**2-10*v1*(2*v2+ys))+u2**2*(16*u2*(v1-v2)*xs*(v1-ys)+3*u2**2*(v1**2+3*xs**2-2*v1*ys+ys**2)+6*(v1-v2)**2*(3*v1**2+xs**2-6*v1*ys+3*ys**2))+3*u1**2*(5*u2**4+32*u2**3*xs+16*u2*(v1-v2)*xs*(2*v1-v2-ys)+6*u2**2*(6*v1**2+v2**2+3*xs**2+4*v2*ys+ys**2-6*v1*(v2+ys))+(v1-v2)**2*(15*v1**2+v2**2+2*xs**2+8*v2*ys+6*ys**2-10*v1*(v2+2*ys)))-4*u1**3*(15*u2**3+36*u2**2*xs+2*(v1-v2)*xs*(5*v1-3*v2-2*ys)+3*u2*(10*v1**2+3*v2**2+3*xs**2+6*v2*ys+ys**2-4*v1*(3*v2+2*ys)))-12*u1*u2*(2*u2**3*xs+2*u2*(v1-v2)*xs*(3*v1-v2-2*ys)+u2**2*(3*v1**2+3*xs**2+2*v2*ys+ys**2-2*v1*(v2+2*ys))+(v1-v2)**2*(5*v1**2+xs**2+2*v2*ys+3*ys**2-2*v1*(v2+4*ys)))) - 5*t**2*u1*(3*u1**5-3*u1**4*(u2+4*xs)-3*u2*(v1**2+xs**2-2*v1*ys+ys**2)**2+3*u1*(v1**2+xs**2-2*v1*ys+ys**2)*(3*v1**2+4*u2*xs+xs**2+2*v2*ys+ys**2-2*v1*(v2+2*ys))+3*u1**3*(3*v1**2+4*u2*xs+6*xs**2+v2*ys+2*ys**2-v1*(v2+5*ys))-2*u1**2*(3*u2*(v1**2+3*xs**2-2*v1*ys+ys**2)+2*xs*(5*v1**2+3*xs**2+2*v2*ys+3*ys**2-2*v1*(v2+4*ys)))) + (5/3)*t**3*(15*u1**6-6*u1**5*(5*u2+8*xs)+3*u2**2*(v1**2+xs**2-2*v1*ys+ys**2)**2-6*u1*u2*(v1**2+xs**2-2*v1*ys+ys**2)*(5*v1**2-4*v1*v2+4*u2*xs+xs**2-6*v1*ys+4*v2*ys+ys**2)+3*u1**4*(5*u2**2+15*v1**2+v2**2+32*u2*xs+18*xs**2+8*v2*ys+6*ys**2-10*v1*(v2+2*ys))+3*u1**2*(15*v1**4+2*v2**2*xs**2+xs**4+8*v2*xs**2*ys+6*v2**2*ys**2+2*xs**2*ys**2+8*v2*ys**3+ys**4-20*v1**3*(v2+2*ys)+6*u2**2*(v1**2+3*xs**2-2*v1*ys+ys**2)+6*v1**2*(v2**2+2*xs**2+8*v2*ys+6*ys**2)-12*v1*(v2*xs**2+v2**2*ys+xs**2*ys+3*v2*ys**2+ys**3)+16*u2*xs*(2*v1**2+xs**2+ys*(v2+ys)-v1*(v2+3*ys)))-4*u1**3*(12*u2**2*xs+3*u2*(5*v1**2+9*xs**2+ys*(2*v2+3*ys)-2*v1*(v2+4*ys))+2*xs*(10*v1**2+v2**2+6*v2*ys-4*v1*(2*v2+3*ys)+3*(xs**2+ys**2)))) - 5*t**4*(5*u1**6-3*u1**5*(5*u2+4*xs)+3*u1**4*(5*u2**2+5*v1**2+v2**2+12*u2*xs+3*xs**2+3*v2*ys+ys**2-5*v1*(v2+ys))+u2**2*(v1**2+xs**2-2*v1*ys+ys**2)*(3*v1**2+2*u2*xs+3*v2*ys-3*v1*(v2+ys))-u1**3*(5*u2**3+36*u2**2*xs+2*xs*(10*v1**2+3*v2**2+xs**2+6*v2*ys+ys**2-4*v1*(3*v2+2*ys))+3*u2*(10*v1**2+v2**2+9*xs**2+6*v2*ys+3*ys**2-4*v1*(2*v2+3*ys)))+3*u1**2*(4*u2**3*xs+2*u2*xs*(6*v1**2+v2**2+xs**2+4*v2*ys+ys**2-6*v1*(v2+ys))+3*u2**2*(2*v1**2+3*xs**2+ys*(v2+ys)-v1*(v2+3*ys))+(v1-v2)*(5*v1**3-v2**2*ys-5*v1**2*(v2+2*ys)-ys*(xs**2+ys**2)-v2*(xs**2+3*ys**2)+v1*(v2**2+2*xs**2+8*v2*ys+6*ys**2)))-3*u1*u2*(u2**2*(v1**2+3*xs**2-2*v1*ys+ys**2)+2*u2*xs*(3*v1**2+xs**2+ys*(2*v2+ys)-2*v1*(v2+2*ys))+(v1-v2)*(5*v1**3-3*v1**2*(v2+4*ys)-2*ys*(xs**2+ys**2)-v2*(xs**2+3*ys**2)+3*v1*(xs**2+ys*(2*v2+3*ys))))) )
            return res*(v2-v1)
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))

def guu_yr4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            res=(t/630.0)*(3*u1**5*(6*(7-21*t+35*t**2-35*t**3+21*t**4-7*t**5+t**6)*v1 + t*(21-70*t+105*t**2-84*t**3+35*t**4-6*t**5)*v2) - 3*u1**4*(30*t**6*u2*(v1-v2) + 210*v1*xs + 42*t**4*(10*u2*v1-6*u2*v2+5*v1*xs-4*v2*xs) + 70*t**2*(5*u2*v1-u2*v2+10*v1*xs-4*v2*xs) - 105*t**3*(5*u2*v1-2*u2*v2+5*v1*xs-3*v2*xs) - 35*t**5*(5*u2*v1-4*u2*v2+v1*xs-v2*xs) - 105*t*(u2*v1+5*v1*xs-v2*xs)) + u1**3*(60*t**6*(3*u2**2+(v1-v2)**2)*(v1-v2) - 70*t**5*(3*u2**2*(4*v1-3*v2) + 6*u2*(v1-v2)*xs + (v1-v2)**2*(6*v1-3*v2-2*ys)) + 420*v1*(v1**2+3*xs**2-2*v1*ys+ys**2) - 105*t**3*(20*v1**3 + 3*u2**2*(4*v1-v2) + 18*u2*(2*v1-v2)*xs - 10*v1**2*(3*v2+2*ys) + 4*v1*(3*v2**2+3*xs**2+6*v2*ys+ys**2) - v2*(v2**2+9*xs**2+6*v2*ys+3*ys**2)) + 140*t**2*(3*u2**2*v1 + 15*v1**3 + 6*u2*(4*v1-v2)*xs - 5*v1**2*(3*v2+4*ys) - v2*(9*xs**2+2*v2*ys+3*ys**2) + v1*(3*v2**2+18*xs**2+16*v2*ys+6*ys**2)) - 210*t*(6*v1**3 + 6*u2*v1*xs - v1**2*(3*v2+10*ys) - v2*(3*xs**2+ys**2) + 4*v1*(3*xs**2+ys*(v2+ys))) + 84*t**4*(9*u2**2*(2*v1-v2) + 6*u2*(4*v1-3*v2)*xs + (v1-v2)*(15*v1**2+3*v2**2+3*xs**2+6*v2*ys+ys**2-5*v1*(3*v2+2*ys)))) + t*u2*(-6*t**5*(3*u2**4+10*u2**2*(v1-v2)**2+15*(v1-v2)**4)*(v1-v2) + 7*t**4*(3*u2**4*v1 + 15*u2**3*(v1-v2)*xs + 30*u2*(v1-v2)**3*xs + 15*(v1-v2)**4*(5*v1-4*ys) + 10*u2**2*(v1-v2)**2*(3*v1-2*ys)) + 315*v1*(v1**2+xs**2-2*v1*ys+ys**2)**2 - 210*t*(v1**2+xs**2-2*v1*ys+ys**2)*(5*v1**3 - v1**2*(5*v2+6*ys) - v2*(xs**2+ys**2) + v1*(2*u2*xs+xs**2+6*v2*ys+ys**2)) - 42*t**3*(3*u2**3*v1*xs + 6*u2*(v1-v2)**2*xs*(3*v1-2*ys) + 2*u2**2*(v1-v2)*(3*v1**2+3*xs**2-4*v1*ys+ys**2) + 6*(v1-v2)**3*(5*v1**2+xs**2-8*v1*ys+3*ys**2)) + 105*t**2*(3*u2*(v1-v2)*xs*(3*v1**2+xs**2-4*v1*ys+ys**2) + u2**2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2) + 3*(v1-v2)**2*(5*v1**3 - 12*v1**2*ys - 2*ys*(xs**2+ys**2) + 3*v1*(xs**2+3*ys**2)))) - 3*u1**2*(60*t**6*u2*(u2**2+(v1-v2)**2)*(v1-v2) + 420*v1*xs*(v1**2+xs**2-2*v1*ys+ys**2) - 70*t**5*(u2**3*(3*v1-2*v2) + 3*u2**2*(v1-v2)*xs + (v1-v2)**3*xs + u2*(v1-v2)**2*(5*v1-2*(v2+ys))) + 84*t**4*(u2**3*(3*v1-v2) + u2**2*(9*v1*xs-6*v2*xs) + (v1-v2)**2*xs*(5*v1-2*(v2+ys)) + u2*(v1-v2)*(10*v1**2+v2**2+3*xs**2+4*v2*ys+ys**2-8*v1*(v2+ys))) - 210*t*(u2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2) + xs*(5*v1**3+3*v1*xs**2+v1*ys*(4*v2+3*ys) - v1**2*(3*v2+8*ys) - v2*(xs**2+ys**2))) + 140*t**2*(3*u2**2*v1*xs + u2*(5*v1**3+9*v1*xs**2+v1*ys*(4*v2+3*ys) - v1**2*(3*v2+8*ys) - v2*(3*xs**2+ys**2)) + xs*(10*v1**3 - 12*v1**2*(v2+ys) + 3*v1*(v2**2+xs**2+4*v2*ys+ys**2) - 2*v2*(xs**2+ys*(v2+ys)))) - 105*t**3*(u2**3*v1 + u2**2*(9*v1*xs-3*v2*xs) + (v1-v2)*xs*(10*v1**2+v2**2+xs**2+4*v2*ys+ys**2-8*v1*(v2+ys)) + u2*(10*v1**3 - 12*v1**2*(v2+ys) + 3*v1*(v2**2+3*xs**2+4*v2*ys+ys**2) - 2*v2*(3*xs**2+ys*(v2+ys))))) + 3*u1*(30*t**6*(u2**2+(v1-v2)**2)**2*(v1-v2) - 35*t**5*(u2**2+(v1-v2)**2)*(u2**2*(2*v1-v2) + 4*u2*(v1-v2)*xs + (v1-v2)**2*(6*v1-v2-4*ys)) + 210*v1*(v1**2+xs**2-2*v1*ys+ys**2)**2 - 105*t*(v1**2+xs**2-2*v1*ys+ys**2)*(6*v1**3 - v1**2*(5*v2+8*ys) - v2*(xs**2+ys**2) + 2*v1*(2*u2*xs+xs**2+3*v2*ys+ys**2)) + 42*t**4*(u2**4*v1 + u2**3*(8*v1*xs-4*v2*xs) + 4*u2*(v1-v2)**2*xs*(4*v1-v2-2*ys) + 2*u2**2*(v1-v2)*(6*v1**2+3*xs**2+ys*(2*v2+ys) - 3*v1*(v2+2*ys)) + (v1-v2)**3*(15*v1**2 - 5*v1*(v2+4*ys) + 2*(xs**2 + ys*(2*v2+3*ys)))) - 105*t**3*(2*u2**3*v1*xs + 2*u2*(v1-v2)*xs*(6*v1**2+xs**2+ys*(2*v2+ys) - 3*v1*(v2+2*ys)) + u2**2*(4*v1**3 + 6*v1*xs**2 + 2*v1*ys*(2*v2+ys) - 3*v1**2*(v2+2*ys) - v2*(3*xs**2+ys**2)) + (v1-v2)**2*(10*v1**3 - 5*v1**2*(v2+4*ys) - 2*ys*(xs**2+ys**2) - v2*(xs**2+3*ys**2) + 4*v1*(xs**2+ys*(2*v2+3*ys)))) + 70*t**2*(2*u2**2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2) + 4*u2*xs*(4*v1**3 - 3*v1**2*(v2+2*ys) - v2*(xs**2+ys**2) + 2*v1*(xs**2+2*v2*ys+ys**2)) + (v1-v2)*(15*v1**4 - 10*v1**3*(v2+4*ys) + (xs**2+ys**2)*(xs**2 + ys*(4*v2+ys)) + 12*v1**2*(xs**2 + ys*(2*v2+3*ys)) - 6*v1*(2*ys*(xs**2+ys**2) + v2*(xs**2+3*ys**2))))) )
            return res*(v2-v1)
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))
def guu_xyr4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            res=(1/30.0)*(-(5/8)*t**8*(u1-u2)**2*(u1**4-4*u1**3*u2+u2**4+3*u1**2*(2*u2**2+(v1-v2)**2)-2*u1*u2*(2*u2**2+3*(v1-v2)**2)+3*u2**2*(v1-v2)**2+3*(v1-v2)**4)*(v1-v2) + t*u1**2*v1*(5*u1**4-24*u1**3*xs-40*u1*xs*(v1**2+xs**2-2*v1*ys+ys**2)+15*(v1**2+xs**2-2*v1*ys+ys**2)**2+15*u1**2*(v1**2+3*xs**2-2*v1*ys+ys**2)) + (1/7)*t**7*(u1-u2)*(5*u1**5*(7*v1-6*v2)+u1**4*(-5*u2*(29*v1-24*v2)+24*(-v1+v2)*xs)+u1**2*(u2**3*(-170*v1+120*v2)-144*u2**2*(v1-v2)*xs-40*(v1-v2)**3*xs-15*u2*(v1-v2)**2*(17*v1-8*v2-6*ys))+u2*(-5*u2**4*v1-24*u2**3*(v1-v2)*xs-40*u2*(v1-v2)**3*xs-15*(v1-v2)**4*(5*v1-4*ys)-15*u2**2*(v1-v2)**2*(3*v1-2*ys))+u1**3*(10*u2**2*(23*v1-18*v2)+96*u2*(v1-v2)*xs+15*(v1-v2)**2*(7*v1-2*(2*v2+ys)))+u1*(5*u2**4*(11*v1-6*v2)+96*u2**3*(v1-v2)*xs+80*u2*(v1-v2)**3*xs+15*u2**2*(v1-v2)**2*(13*v1-4*v2-6*ys)+15*(v1-v2)**4*(7*v1-2*(v2+2*ys)))) - 0.5*t**2*u1*(5*u1**5*(7*v1-v2)-6*u1**4*(5*u2*v1+24*v1*xs-4*v2*xs)-30*u2*v1*(v1**2+xs**2-2*v1*ys+ys**2)**2+15*u1**3*(7*v1**3-3*v1**2*(v2+4*ys)-v2*(3*xs**2+ys**2)+v1*(8*u2*xs+15*xs**2+4*v2*ys+5*ys**2))+15*u1*(v1**2+xs**2-2*v1*ys+ys**2)*(7*v1**3+8*u2*v1*xs-5*v1**2*(v2+2*ys)-v2*(xs**2+ys**2)+3*v1*(xs**2+ys*(2*v2+ys)))-20*u1**2*(3*u2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2)+2*xs*(6*v1**3-v1**2*(3*v2+10*ys)-v2*(xs**2+ys**2)+4*v1*(xs**2+ys*(v2+ys))))) - (1/6)*t**6*(15*u1**6*(7*v1-5*v2)-6*u1**5*(75*u2*v1-50*u2*v2+24*v1*xs-20*v2*xs)+u2**2*(24*u2**3*v1*xs+40*u2*(v1-v2)**2*xs*(3*v1-2*ys)+15*u2**2*(v1-v2)*(3*v1**2+3*xs**2-4*v1*ys+ys**2)+30*(v1-v2)**3*(5*v1**2+xs**2-8*v1*ys+3*ys**2))+15*u1**4*(10*u2**2*(5*v1-3*v2)+8*u2*(5*v1-4*v2)*xs+(v1-v2)*(21*v1**2+6*v2**2+3*xs**2+8*v2*ys+ys**2-12*v1*(2*v2+ys)))+15*u1**2*(5*u2**4*(3*v1-v2)+16*u2**3*(3*v1-2*v2)*xs+8*u2*(v1-v2)**2*xs*(5*v1-2*(v2+ys))+6*u2**2*(v1-v2)*(10*v1**2+v2**2+3*xs**2+4*v2*ys+ys**2-8*v1*(v2+ys))+(v1-v2)**3*(21*v1**2+v2**2+2*xs**2+8*v2*ys+6*ys**2-12*v1*(v2+2*ys))) -20*u1**3*(15*u2**3*(2*v1-v2)+12*u2**2*(4*v1-3*v2)*xs+2*(v1-v2)**2*xs*(6*v1-3*v2-2*ys)+3*u2*(v1-v2)*(15*v1**2+3*v2**2+3*xs**2+6*v2*ys+ys**2-5*v1*(3*v2+2*ys))) -30*u1*u2*(u2**4*v1+u2**3*(8*v1*xs-4*v2*xs)+4*u2*(v1-v2)**2*xs*(4*v1-v2-2*ys)+2*u2**2*(v1-v2)*(6*v1**2+3*xs**2+ys*(2*v2+ys)-3*v1*(v2+2*ys))+(v1-v2)**3*(15*v1**2-5*v1*(v2+4*ys)+2*(xs**2+ys*(2*v2+3*ys))))) + t**5*(5*u1**6*(7*v1-4*v2)-12*u1**5*(10*u2*v1-5*u2*v2+6*v1*xs-4*v2*xs)+3*u1**4*(35*v1**3+10*u2**2*(5*v1-2*v2)+16*u2*(5*v1-3*v2)*xs-30*v1**2*(2*v2+ys)-4*v2*(v2**2+3*xs**2+3*v2*ys+ys**2)+5*v1*(6*v2**2+3*xs**2+8*v2*ys+ys**2))+u2**2*(8*u2*(v1-v2)*xs*(3*v1**2+xs**2-4*v1*ys+ys**2)+3*u2**2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2)+6*(v1-v2)**2*(5*v1**3-12*v1**2*ys-2*ys*(xs**2+ys**2)+3*v1*(xs**2+3*ys**2))) -4*u1**3*(5*u2**3*(4*v1-v2)+36*u2**2*(2*v1-v2)*xs+2*(v1-v2)*xs*(15*v1**2+3*v2**2+xs**2+6*v2*ys+ys**2-5*v1*(3*v2+2*ys))+3*u2*(20*v1**3-10*v1**2*(3*v2+2*ys)+4*v1*(3*v2**2+3*xs**2+6*v2*ys+ys**2)-v2*(v2**2+9*xs**2+6*v2*ys+3*ys**2))) + 3*u1**2*(5*u2**4*v1+16*u2**3*(3*v1-v2)*xs+8*u2*(v1-v2)*xs*(10*v1**2+v2**2+xs**2+4*v2*ys+ys**2-8*v1*(v2+ys))+(v1-v2)**2*(35*v1**3-30*v1**2*(v2+2*ys)+5*v1*(v2**2+2*xs**2+8*v2*ys+6*ys**2)-4*(v2*xs**2+v2**2*ys+xs**2*ys+3*v2*ys**2+ys**3))+6*u2**2*(10*v1**3-12*v1**2*(v2+ys)+3*v1*(v2**2+3*xs**2+4*v2*ys+ys**2)-2*v2*(3*xs**2+ys*(v2+ys)))) -12*u1*u2*(2*u2**3*v1*xs+2*u2*(v1-v2)*xs*(6*v1**2+xs**2+ys*(2*v2+ys)-3*v1*(v2+2*ys))+u2**2*(4*v1**3+6*v1*xs**2+2*v1*ys*(2*v2+ys)-3*v1**2*(v2+2*ys)-v2*(3*xs**2+ys**2))+(v1-v2)**2*(10*v1**3-5*v1**2*(v2+4*ys)-2*ys*(xs**2+ys**2)-v2*(xs**2+3*ys**2)+4*v1*(xs**2+ys*(2*v2+3*ys))))) + (5/3)*t**3*(3*u1**6*(7*v1-2*v2)-6*u1**5*(6*u2*v1-u2*v2+12*v1*xs-4*v2*xs)+3*u2**2*v1*(v1**2+xs**2-2*v1*ys+ys**2)**2 -6*u1*u2*(v1**2+xs**2-2*v1*ys+ys**2)*(6*v1**3 - v1**2*(5*v2+8*ys) - v2*(xs**2+ys**2) + 2*v1*(2*u2*xs+xs**2+3*v2*ys+ys**2)) + 3*u1**4*(5*u2**2*v1+21*v1**3+8*u2*(5*v1-v2)*xs-6*v1**2*(3*v2+5*ys)-2*v2*(6*xs**2+ys*(v2+2*ys))+v1*(3*v2**2+20*v2*ys+10*(3*xs**2+ys**2))) + 3*u1**2*(21*v1**5-30*v1**4*(v2+2*ys)+6*u2**2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2)+10*v1**3*(v2**2+2*xs**2+8*v2*ys+6*ys**2)-24*v1**2*(v2*xs**2+v2**2*ys+xs**2*ys+3*v2*ys**2+ys**3) - 2*v2*(xs**2+ys**2)*(xs**2+ys*(2*v2+ys)) + 8*u2*xs*(5*v1**3+3*v1*xs**2+v1*ys*(4*v2+3*ys)-v1**2*(3*v2+8*ys)-v2*(xs**2+ys**2)) + 3*v1*(8*v2*ys*(xs**2+ys**2) + (xs**2+ys**2)**2 + 2*v2**2*(xs**2+3*ys**2))) - 4*u1**3*(12*u2**2*v1*xs + 3*u2*(6*v1**3 - v1**2*(3*v2+10*ys) - v2*(3*xs**2+ys**2) + 4*v1*(3*xs**2 + ys*(v2+ys))) + 2*xs*(15*v1**3 - 5*v1**2*(3*v2+4*ys) - v2*(3*xs**2 + ys*(2*v2+3*ys)) + v1*(3*v2**2 + 16*v2*ys + 6*(xs**2 + ys**2))))) - (5/4)*t**4*(5*u1**6*(7*v1-3*v2) - 6*u1**5*(15*u2*v1 - 5*u2*v2 + 16*v1*xs - 8*v2*xs) + 3*u1**4*(35*v1**3 + 5*u2**2*(5*v1 - v2) + 16*u2*(5*v1 - 2*v2)*xs - 5*v1**2*(9*v2 + 8*ys) + 5*v1*(3*v2**2 + 6*xs**2 + 8*v2*ys + 2*ys**2) - v2*(v2**2 + 18*xs**2 + 8*v2*ys + 6*ys**2)) + u2**2*(v1**2+xs**2-2*v1*ys+ys**2)*(15*v1**3 + 8*u2*v1*xs - 3*v1**2*(5*v2+6*ys) - 3*v2*(xs**2+ys**2) + 3*v1*(xs**2 + ys*(6*v2+ys))) + 3*u1**2*(16*u2**3*v1*xs + 6*u2**2*(5*v1**3 + 9*v1*xs**2 + v1*ys*(4*v2+3*ys) - v1**2*(3*v2+8*ys) - v2*(3*xs**2+ys**2)) + (v1-v2)*(35*v1**4 - 40*v1**3*(v2+2*ys) + 8*v2*ys*(xs**2+ys**2) + (xs**2+ys**2)**2 + 2*v2**2*(xs**2+3*ys**2) + 10*v1**2*(v2**2+2*xs**2+8*v2*ys+6*ys**2) - 16*v1*(v2*xs**2 + v2**2*ys + xs**2*ys + 3*v2*ys**2 + ys**3)) + 8*u2*xs*(10*v1**3 - 12*v1**2*(v2+ys) + 3*v1*(v2**2+xs**2+4*v2*ys+ys**2) - 2*v2*(xs**2+ys*(v2+ys)))) - 4*u1**3*(5*u2**3*v1 + 12*u2**2*(4*v1 - v2)*xs + 3*u2*(15*v1**3 - 5*v1**2*(3*v2 + 4*ys) - v2*(9*xs**2 + 2*v2*ys + 3*ys**2) + v1*(3*v2**2 + 18*xs**2 + 16*v2*ys + 6*ys**2)) + 2*xs*(20*v1**3 - 10*v1**2*(3*v2 + 2*ys) + 4*v1*(3*v2**2 + xs**2 + 6*v2*ys + ys**2) - v2*(v2**2 + 6*v2*ys + 3*(xs**2 + ys**2)))) - 6*u1*u2*(2*u2**2*v1*(v1**2 + 3*xs**2 - 2*v1*ys + ys**2) + 4*u2*xs*(4*v1**3 - 3*v1**2*(v2+2*ys) - v2*(xs**2 + ys**2) + 2*v1*(xs**2 + 2*v2*ys + ys**2)) + (v1-v2)*(15*v1**4 - 10*v1**3*(v2+4*ys) + (xs**2+ys**2)*(xs**2 + ys*(4*v2+ys)) + 12*v1**2*(xs**2 + ys*(2*v2+3*ys)) - 6*v1*(2*ys*(xs**2+ys**2) + v2*(xs**2 + 3*ys**2))))))
            return res*(v2-v1)
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))
def guu_series2_term1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.538469  )   *0.5
    p2=(   1+ 0.538469  )   *0.5
    p3=(    1-0.90618   )  *0.5
    p4=(    1+0.90618   )  *0.5
    p5=0.5
    w5=128/225
    w1=0.478629
    w2=w1
    w3=0.236927
    w4=w3
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            x = u1 + t*(u2 - u1)-xs
            y = v1 + t*(v2 - v1)-ys
            R2 = x**2 + y**2 
            r  = np.sqrt(R2)
            return (x*r+2*y**2*np.log(-x+r))*0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )

def guu_series2_xterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.538469  )   *0.5
    p2=(   1+ 0.538469  )   *0.5
    p3=(    1-0.90618   )  *0.5
    p4=(    1+0.90618   )  *0.5
    p5=0.5
    w5=128/225
    w1=0.478629
    w2=w1
    w3=0.236927
    w4=w3
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            x = u1 + t*(u2 - u1)-xs
            y = v1 + t*(v2 - v1)-ys
            xx=x+xs
            yy=y+ys
            R2 = x**2 + y**2 
            r  = np.sqrt(R2)
            return ((1/3)*(2*xx**2-7*yy**2-xx*xs-xs**2+14*yy*ys-7*ys**2)*r+2*xs*y**2*np.log(r-x))*0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )


def guu_series2_yterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.538469  )   *0.5
    p2=(   1+ 0.538469  )   *0.5
    p3=(    1-0.90618   )  *0.5
    p4=(    1+0.90618   )  *0.5
    p5=0.5
    w5=128/225
    w1=0.478629
    w2=w1
    w3=0.236927
    w4=w3
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            x = u1 + t*(u2 - u1)-xs
            y = v1 + t*(v2 - v1)-ys
            xx=x+xs
            yy=y+ys
            R2 = x**2 + y**2 
            r  = np.sqrt(R2)
            return (yy*( x*r+2*y**2*np.log(r-x)      ))*0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )


def guu_series2_xyterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.538469  )   *0.5
    p2=(   1+ 0.538469  )   *0.5
    p3=(    1-0.90618   )  *0.5
    p4=(    1+0.90618   )  *0.5
    p5=0.5
    w5=128/225
    w1=0.478629
    w2=w1
    w3=0.236927
    w4=w3
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            x = u1 + t*(u2 - u1)-xs
            y = v1 + t*(v2 - v1)-ys
            xx=x+xs
            yy=y+ys
            R2 = x**2 + y**2 
            r  = np.sqrt(R2)
            return yy*  ((1/3)*r*( 2*xx**2-7*yy**2-xx*xs-xs**2+14*yy*ys -7*ys**2   ) + 2*xs*y**2*np.log(r-x)       )                *0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )

def guu_series2_term2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            return (-(1/12.0)*t*(2*(-4+6*t-4*t**2+t**3)*u1**3 - 2*u1**2*(3*t**3*u2 - 12*xs - 4*t**2*(2*u2+xs) + 6*t*(u2+2*xs)) + t*u2*(t**2*(-2*u2**2 + 3*(v1-v2)**2) + 6*(v1**2 - 2*xs**2 - 2*v1*ys + ys**2) - 8*t*(v1**2 - u2*xs + v2*ys - v1*(v2+ys))) + u1*(t**3*(6*u2**2 - 3*(v1-v2)**2) - 4*t**2*(2*u2**2 + 4*u2*xs - (v1-v2)*(3*v1 - v2 - 2*ys)) + 12*(v1**2 - 2*xs**2 - 2*v1*ys + ys**2) - 6*t*(3*v1**2 - 4*u2*xs - 2*xs**2 + 2*v2*ys + ys**2 - 2*v1*(v2+2*ys))))  )    *(v2-v1)
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))

def guu_series2_xterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            return (v2-v1)*(  (1/6.0)*((3/5)*t**5*(u1-u2)**2*(u1**2-2*u1*u2+u2**2-(v1-v2)**2) + t*u1**2*(3*u1**2-8*u1*xs-3*(v1**2-2*xs**2-2*v1*ys+ys**2)) - 0.5*t**4*(u1-u2)*(6*u1**3 - 4*u1**2*(3*u2+xs) + u1*(6*u2**2+8*u2*xs-3*(v1-v2)*(2*v1-v2-ys)) + u2*(3*v1**2 - 4*u2*xs + 3*v2*ys - 3*v1*(v2+ys))) + t**3*(6*u1**4 - 4*u1**3*(3*u2+2*xs) - u2**2*(v1**2-2*xs**2-2*v1*ys+ys**2) + u1**2*(6*u2**2 - 6*v1**2 - v2**2 + 16*u2*xs + 2*xs**2 - 4*v2*ys - ys**2 + 6*v1*(v2+ys)) + 2*u1*u2*(3*v1**2 - 4*u2*xs - 2*xs**2 + 2*v2*ys + ys**2 - 2*v1*(v2+2*ys))) - 3*t**2*u1*(2*u1**3 - 2*u1**2*(u2+2*xs) + u2*(v1**2-2*xs**2-2*v1*ys+ys**2) + u1*(-2*v1**2 + 4*u2*xs + 2*xs**2 - v2*ys - ys**2 + v1*(v2+3*ys))))    )
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))

def guu_series2_yterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            return (v2-v1)*(  (1/3.0)*((1/5)*t**5*(u1-u2)*(2*u1**2-4*u1*u2+2*u2**2-3*(v1-v2)**2)*(v1-v2) - 0.25*t**4*(u1**3*(8*v1-6*v2) - 6*u1**2*(3*u2*v1-2*u2*v2+v1*xs-v2*xs) + u2*(-2*u2**2*v1 + 6*u2*(-v1+v2)*xs + 3*(v1-v2)**2*(3*v1-2*ys)) + 3*u1*(u2**2*(4*v1-2*v2) + 4*u2*(v1-v2)*xs - (v1-v2)**2*(4*v1 - v2 - 2*ys))) + t*u1*v1*(2*u1**2 - 6*u1*xs - 3*(v1**2 - 2*xs**2 - 2*v1*ys + ys**2)) - 0.5*t**2*(u1**3*(8*v1-2*v2) - 6*u1**2*(u2*v1 + 3*v1*xs - v2*xs) + 3*u2*v1*(v1**2 - 2*xs**2 - 2*v1*ys + ys**2) - 3*u1*(4*v1**3 - 3*v1**2*(v2+2*ys) + v2*(2*xs**2 - ys**2) + v1*(-4*u2*xs - 4*xs**2 + 2*ys*(2*v2+ys)))) + t**3*(u1**3*(4*v1-2*v2) + u1**2*(-6*u2*v1 + 2*u2*v2 - 6*v1*xs + 4*v2*xs) + u2*(3*v1**3 - v1**2*(3*v2+4*ys) + v2*(2*xs**2 - ys**2) + v1*(-2*u2*xs - 2*xs**2 + 4*v2*ys + ys**2)) + u1*(2*u2**2*v1 + u2*(8*v1*xs - 4*v2*xs) - (v1-v2)*(6*v1**2 - 2*xs**2 + 2*v2*ys + ys**2 - 3*v1*(v2+2*ys)))))    )
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))
def guu_series2_xyterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            return (v2-v1)*(  (1/6.0)*(-(1/2)*t**6*(u1-u2)**2*(u1**2-2*u1*u2+u2**2-(v1-v2)**2)*(v1-v2) + t*u1**2*v1*(3*u1**2-8*u1*xs-3*(v1**2-2*xs**2-2*v1*ys+ys**2)) - 0.5*t**2*u1*(3*u1**3*(5*v1-v2) - 4*u1**2*(3*u2*v1+8*v1*xs-2*v2*xs) + 6*u2*v1*(v1**2-2*xs**2-2*v1*ys+ys**2) - 3*u1*(5*v1**3 - v1**2*(3*v2+8*ys) + v2*(2*xs**2-ys**2) + v1*(-8*u2*xs-6*xs**2 + 4*v2*ys + 3*ys**2))) + (1/5)*t**5*(u1-u2)*(3*u1**3*(5*v1-4*v2) + u1**2*(-33*u2*v1 + 24*u2*v2 - 8*v1*xs + 8*v2*xs) + u2*(-3*u2**2*v1 + 8*u2*(-v1+v2)*xs + 3*(v1-v2)**2*(3*v1-2*ys)) + u1*(3*u2**2*(7*v1-4*v2) + 16*u2*(v1-v2)*xs - 3*(v1-v2)**2*(5*v1-2*(v2+ys)))) + t**3*(2*u1**4*(5*v1-2*v2) - 4*u1**3*(4*u2*v1 - u2*v2 + 4*v1*xs - 2*v2*xs) - u2**2*v1*(v1**2-2*xs**2-2*v1*ys+ys**2) + u1**2*(6*u2**2*v1 - 10*v1**3 + 8*u2*(3*v1 - v2)*xs - 4*v2*xs**2 + 12*v1**2*(v2+ys) + 2*v2*ys*(v2+ys) - 3*v1*(v2**2 - 2*xs**2 + 4*v2*ys + ys**2)) - 2*u1*u2*(-4*v1**3 + 3*v1**2*(v2+2*ys) + v2*(-2*xs**2 + ys**2) + v1*(4*u2*xs + 4*xs**2 - 2*ys*(2*v2+ys)))) - 0.25*t**4*(6*u1**4*(5*v1-3*v2) - 4*u1**3*(18*u2*v1 - 9*u2*v2 + 8*v1*xs - 6*v2*xs) + 3*u1**2*(6*u2**2*(3*v1 - v2) + 8*u2*(3*v1-2*v2)*xs - (v1-v2)*(10*v1**2 + v2**2 - 2*xs**2 + 4*v2*ys + ys**2 - 8*v1*(v2+ys))) + u2**2*(-9*v1**3 + 3*v1**2*(3*v2+4*ys) + 3*v2*(-2*xs**2 + ys**2) + v1*(8*u2*xs + 6*xs**2 - 3*ys*(4*v2+ys))) - 6*u1*u2*(2*u2**2*v1 + u2*(8*v1*xs - 4*v2*xs) - (v1-v2)*(6*v1**2 - 2*xs**2 + 2*v2*ys + ys**2 - 3*v1*(v2+2*ys)))) )    )
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))




x1=0
y1=0
x2=0.0462663
y2=0
x3=0.0646801
y3=0.0328262
x4=0.0275032
y4=0.0374615
xs1=0.0151834
ys1=0.00770954
xs2=0.0407863
ys2=0.007144
xs3=0.0299534
ys3=0.0287724
xs4=0.0525265
ys4=0.0266618

print(guu_term1(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1))
x1=-1/20
y1=-1/20
x2=1/20
y2=-1/20
x3=1/20
y3=1/20
x4=-1/20
y4=1/20
xs1=-3**(-0.5)/20
ys1=-3**(-0.5)/20
xs2=3**(-0.5)/20
ys2=-3**(-0.5)/20
xs3=3**(-0.5)/20
ys3=3**(-0.5)/20
xs4=-3**(-0.5)/20
ys4=3**(-0.5)   /20
#xs2=0.0407863
#ys2=0.007144
#xs3=0.0299534
#ys3=0.0287724


#res=guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)
#res1=0.25*( guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)- 20*3**0.5*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)  +20*3**0.5*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1) -400*3*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)          )
#res2=0.25*( guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)-20*3**0.5*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)  +20*3**0.5*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2) -400*3*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)          )
#res3=0.25*( guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)-20*3**0.5*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)  +20*3**0.5*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3) -400*3*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)          )
#res4=0.25*( guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)-20*3**0.5*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)  +20*3**0.5*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4) -400*3*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)          )
#print(res,res1+res2+res3+res4,res1,res2,res3,res4)














x1=0
y1=0
x2=0.01
y2=0
x3=0.01
y3=0.01
x4=0
y4=0.01
xs1=  (-3**(-0.5)   +1)*0.005
ys1= (-3**(-0.5)+1)*0.005
xs2= (3**(-0.5)+1)*0.005
ys2= (-3**(-0.5)+1)*0.005
xs3= (3**(-0.5)+1)*0.005
ys3= (3**(-0.5)+1)*0.005
xs4= (-3**(-0.5)+1)*0.005
ys4= (3**(-0.5)+1)*0.005
#





constant=1
#xcoef= -3**(0.5)#0*(xs3*xs4+xs3*xs2+xs2*xs4)/(  -xs2*xs3*xs4)
#ycoef= -3**0.5#0*(ys3*ys4+ys3*ys2+ys2*ys4)/(  -ys2*ys3*ys4)
#xycoef=xcoef*ycoef
#res=guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)

#r1=0.25*(constant*guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)+xcoef*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)  +ycoef*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)  +xycoef*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)           )
#r2=0.25*(constant*guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)+xcoef*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)  +ycoef*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)  +xycoef*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)           )
#r3=0.25*(constant*guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)+xcoef*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)  +ycoef*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)  +xycoef*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)           )
#r4=0.25*(constant*guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)+xcoef*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)  +ycoef*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)  +xycoef*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)           )

#print(res,r1,r2,r3,r4,r1+r2+r3+r4)


#print(guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4),guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4) ,guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4) ,guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)   )
#x1=-1
#y1=-1
#x2=1
#y2=-1
#x3=1
#y3=1
#x4=-1
#y4=1
#xs1=  -3**(-0.5)
#ys1= -3**(-0.5)
#xs2= 3**(-0.5)
#ys2= -3**(-0.5)
#xs3= 3**(-0.5)
#ys3= 3**(-0.5)
#xs4= -3**(-0.5)
#ys4= 3**(-0.5)

#I1=guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)
#I2=guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)
#I3=guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)
#I4=guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)
#I78=guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, 0.0159319185473049, -0.0111677971553096)
#print(I1)
#print(I78)


#x1=-1  /2
#y1=-1/2
#x2=1/2
#y2=-1/2
#x3=1/2
#y3=1/2
#x4=-1/2
#y4=1/2
#xs1=-3**(-0.5)   /2
#ys1=-3**(-0.5)    /2
#xs2=3**(-0.5)    /2 
#ys2=-3**(-0.5)   /2
#xs3=3**(-0.5)     /2 
#ys3=3**(-0.5)     /2
#xs4=-3**(-0.5)    /2 
#ys4=3**(-0.5)      /2
#res=guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)
#xcoef=-3**(0.5)    
#ycoef=xcoef
#xycoef=3
#r1=0.25*(guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)+xcoef*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)  +ycoef*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)  +xycoef*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1)           )
#r2=0.25*(guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)+xcoef*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)  +ycoef*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)  +xycoef*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs2, ys2)           )
#r3=0.25*(guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)+xcoef*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)  +ycoef*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)  +xycoef*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs3, ys3)           )
#r4=0.25*(guu_term2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)+xcoef*guu_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)  +ycoef*guu_yterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)  +xycoef*guu_xyterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs4, ys4)           )

#print(res,r1,r2,r3,r4,r1+r2+r3+r4)


