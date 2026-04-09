# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 14:03:00 2025

@author: Faik
"""
#This code calculates guu compoınents of  singular integrals in the EFIE with Greens Theorem
# singular integrals of guu=(g+ (1/k^2)diff[g,u,2])
#Firstly, taylor series expansion is applied to guu
#The result will be (1/(4pi) )(  term1+term2+term3+term4+ term 5+ series 1 +series 2           )
#term2=  (v^2+0.5u^2)/r3  weakly singular
#term1=  (2v^2 -u^2)/(k^2  r^5)   hyper singular
#term3= k(-2j/3)
#term4=  -k^2 r/3
#term 5= j k^3 u^2  /6
#series 1=  (from n=4 to infinity )  (- j k )^n r^(n-1)  /  n!
#series2=  (from n=4 to infinity ) (  (-j r)^n  k^(n-2)  / n!    )* ( (2v^2-u^2)/r^5  + ik  (2v^2-u^2)/r^4   - k^2  v^2/r^3   )
#(series 1 n=4)  + (series 1 n=5)=  (k^4  r^3  /4 !)  - j k^5 r^4 /5!
#(series 2 n=4)  + (series 2 n=5)= (k^2/4!)((2u^2-v^2)/r)+(2u^2-v^2)(k^2 / 4!  -  jk^3/5!)+ (k^4/5!)*(-3rv^2-ru^2)
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
def gvv_term1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return ( (np.sqrt((1-t)**2*u1**2 + (2-2*t)*t*u1*u2 + v1**2 + t*v1*(-2*v1+2*v2) + t**2*(u2**2 + v1**2 - 2*v1*v2 + v2**2))*(-t**2*u2**7*v1**5+ (1-t)**2*u1**7*v2**5+ t*u1*u2**6*v1**4*((-2+2*t)*v1 + 5*t*v2)+ u1**6*u2*v2**4*(-5*(1-t)**2*v1 + (2-2*t)*t*v2)+ u1**2*u2**5*v1**3*(-(1-t)**2*v1**2 + (10-10*t)*t*v1*v2 - 10*t**2*v2**2)+ u1**4*u2**3*v1*v2**2*(-10*(1-t)**2*v1**2 + (20-20*t)*t*v1*v2 - 5*t**2*v2**2)+ u1**5*u2**2*v2**3*(10*(1-t)**2*v1**2 + t*(-10+10*t)*v1*v2 + t**2*v2**2)+ u1**3*u2**4*v1**2*v2*(5*(1-t)**2*v1**2 + t*(-20+20*t)*v1*v2 + 10*t**2*v2**2) )) / (((-1+t)*v1 - t*v2)* (-u2*v1 + u1*v2)**2* (u2**2*v1**2 - 2*u1*u2*v1*v2 + u1**2*v2**2)**2* ((1-t)**2*u1**2 + (2-2*t)*t*u1*u2 + v1**2 + t*v1*(-2*v1+2*v2) + t**2*(u2**2 + v1**2 - 2*v1*v2 + v2**2))))*(v2-v1)
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))
def gvv_xterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            d  = u1**2 - 2*u1*u2 + u2**2 + (v1 - v2)**2
            s1 = np.sqrt((1-t)**2*u1**2 - 2*(1-t)*t*u1*u2 + v1**2+ t**2*(u2**2 + (v1 - v2)**2) - 2*t*v1*(v1 - v2))
            s2 = np.sqrt(d)
            num1 = -(( -1 + t)*u1**2) + (-1 + 2*t)*u1*u2 + t*(-u2**2 + (v1 - v2)**2) + v1*(-v1 + v2)
            term1 = num1 / (s1 * d)
            A = ((-1 + t)*u1**2 + u1*(u2 - 2*t*u2) + t*(u2**2 + (v1 - v2)**2) + v1*(-v1 + v2)) / (s1 * s2)
            term2 = 0.5*(u1 - u2)**2 / (d**1.5) * ( -np.log(np.abs(1 - A)) + np.log(np.abs(1 + A)) )
            result= term1 + term2
            return result*(v2-v1)
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+p_x*gvv_term1(x1, y1, x2, y2, x3, y3, x4, y4,0, 0)
def gvv_yterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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

           return np.real(    (u2 * ((-1 + 2*t) * v1 - 2*t * v2) - u1 * (2*( -1 + t) * v1 + v2 - 2*t * v2))/ (np.sqrt((1 - t)**2 * u1**2 - 2*(1 - t)*t * u1*u2 + v1**2 + t**2 * (u2**2 + (v1 - v2)**2) - 2*t * v1 * (v1 - v2))* (u1**2 - 2*u1*u2 + u2**2 + (v1 - v2)**2))+ np.log(np.abs(u1 - t*u1 + t*u2 + np.sqrt((1 - t)**2 * u1**2 - 2*(1 - t)*t * u1*u2 + v1**2 + t**2 * (u2**2 + (v1 - v2)**2) - 2*t * v1 * (v1 - v2))))/ (v1 - v2)- np.log(np.abs(( -1 + t) * v1 - t * v2)) / (v1 - v2)+ ((u1 - u2) * (u1**2 - 2*u1*u2 + u2**2 + 2*(v1 - v2)**2)* np.log(np.abs(-u1**2 + t*u1**2 + u1*u2 - 2*t*u1*u2 + t*u2**2- v1**2 + t*v1**2 + v1*v2 - 2*t*v1*v2 + t*v2**2+ np.sqrt((1 - t)**2 * u1**2 - 2*(1 - t)*t * u1*u2 + v1**2 + t**2 * (u2**2 + (v1 - v2)**2) - 2*t * v1 * (v1 - v2))* np.sqrt(u1**2 - 2*u1*u2 + u2**2 + (v1 - v2)**2))))/ (np.sqrt(u1**2 - 2*u1*u2 + u2**2 + (v1 - v2)**2)* (v1 - v2)* (1j*(u1 - u2) + (v1 - v2))* (-1j*(u1 - u2) + (v1 - v2))) )*(v2-v1)
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+p_y*gvv_term1(x1, y1, x2, y2, x3, y3, x4, y4, 0,0)
def gvv_xyterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    x1=x1-p_x
    x2=x2-p_x
    x3=x3-p_x
    x4=x4-p_x
    y1=y1-p_y
    y2=y2-p_y
    y3=y3-p_y
    y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        xs=p_x
        ys=p_y
        if v2-v1==0:
            return 0
        else:
            expr=0#((np.sqrt((1-t)**2*u1**2 - 2*(1-t)*t*u1*u2 + v1**2 + t**2*(u2**2+(v1-v2)**2) - 2*t*v1*(v1-v2)) * (-((u1-u2)**2*(v1-v2))/((u1**2 - 2*u1*u2 + u2**2 + (v1-v2)**2)**2)) + (xs*ys)/(((1-t)*v1 - t*v2)*(-u2*v1 + u1*v2)) + ((-1+t)*u1**5*v2*(v2+ys) + t*(3*u2**3*v1**2*(v1-v2)**2 + u2*v1*(v1-v2)**4*ys + (v1-v2)**5*xs*ys - u2**5*v1*(v1+ys) + 2*u2**2*(v1-v2)**3*xs*(v1+ys) + u2**4*(v1-v2)*xs*(2*v1+ys)) - v1*(u2*v1*(v1-v2)**3*ys + (v1-v2)**4*xs*ys + u2**4*xs*(v1+ys) + u2**3*v1*(v1-v2)*(2*v1+ys) + u2**2*(v1-v2)**2*xs*(v1+2*ys)) + u1**4*(-(-1+t)*u2*v1*(2*v2+ys) +(-1+t)*v1*xs*(2*v2+ys) + u2*v2*((2-3*t)*v2+(3-4*t)*ys) + v2*xs*(v2-2*t*v2 - t*ys)) + u1*( -(v1-v2)**3*v2*((1-t)*v1 - t*v2)*ys - u2**2*v1*(v1-v2)*(3*(1-t)*v1**2 + 3*(1-t)*v1*v2 - v1*ys - 2*v2*(3*t*v2+ys)) + u2**4*((-1+3*t)*v1**2 - v1*ys + t*v2*ys + 2*t*v1*(v2+2*ys)) - 2*u2*(v1-v2)**2*xs*((1-t)*v1**2 + 2*(1-t)*v1*ys - t*v2*(v2+2*ys)) + 2*u2**3*xs*((2-3*t)*v1**2 + 2*v1*(t*(v2-ys)+ys) + t*v2*(v2+2*ys))) + u1**3*(-(v1-v2)*v2**2*(3*(1-t)*v1+v2-3*t*v2-ys) + 2*u2*xs*(-(1-t)*v1**2 - 2*(1-t)*v1*(v2+ys) + v2*((-1+3*t)*v2+2*t*ys)) + u2**2*((-1+t)*v1**2 +(-4+6*t)*v1*v2 +(-3+4*t)*v1*ys + v2*(-v2+3*t*v2-3*ys+6*t*ys))) + u1**2*( u2*(v1-v2)*v2*(6*(1-t)*v1**2 - v2*(3*t*v2+ys) - v1*(3*t*v2+2*ys)) + u2**2*xs*((-5+6*t)*v1**2 - 2*v1*v2 + 6*(1-t)*v1*ys + v2*(v2-6*t*v2-6*t*ys)) + (v1-v2)**2*xs*(2*(1-t)*v1*(v2+ys) + v2*(v2-2*t*v2-2*t*ys)) + u2**3*((2-3*t)*v1**2 + v1*((2-6*t)*v2+3*ys-6*t*ys) + v2*(ys - t*(v2+4*ys)))))) / (((1-t)**2*u1**2 - 2*(1-t)*t*u1*u2 + v1**2 + t**2*(u2**2+(v1-v2)**2) - 2*t*v1*(v1-v2)) * (u1**2 - 2*u1*u2 + u2**2 + (v1-v2)**2)**2 * (u2*v1 - u1*v2)) + xs*np.log(np.abs(u1 - t*u1 + t*u2 + np.sqrt((1-t)**2*u1**2 - 2*(1-t)*t*u1*u2 + v1**2 + t**2*(u2**2+(v1-v2)**2) - 2*t*v1*(v1-v2))))/(v1-v2) - xs*np.log(np.abs((1-t)*v1 - t*v2))/(v1-v2) + (1/((u1**2 - 2*u1*u2 + u2**2 + (v1-v2)**2)**2.5 * (v1-v2))) * (u1-u2) * (u1**4*xs + u2**4*xs + 3*u2**2*(v1-v2)**2*xs + 2*(v1-v2)**4*xs + u2*(v1-v2)**3*(2*v1-ys) - u2**3*(v1-v2)*(v1+ys) + u1**3*(-v2**2 - 4*u2*xs - v2*ys + v1*(v2+ys)) - u1*(4*u2**3*xs + 6*u2*(v1-v2)**2*xs + (v1-v2)**3*(2*v2-ys) - u2**2*(v1-v2)*(2*v1+v2+3*ys)) + u1**2*(6*u2**2*xs + 3*(v1-v2)**2*xs - u2*(v1-v2)*(v1+2*v2+3*ys))) * np.log(np.abs(-u1**2 + t*u1**2 + u1*u2 - 2*t*u1*u2 + t*u2**2 - v1**2 + t*v1**2 + v1*v2 - 2*t*v1*v2 + t*v2**2 + np.sqrt((1-t)**2*u1**2 - 2*(1-t)*t*u1*u2 + v1**2 + t**2*(u2**2+(v1-v2)**2) - 2*t*v1*(v1-v2)) * np.sqrt(u1**2 - 2*u1*u2 + u2**2 + (v1-v2)**2) + v1*v2 - 2*t*v1*v2 + t*v2**2)))
            return (v2-v1)*expr
    return ((f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0)))#+ p_x* p_y*gvv_term1(x1, y1, x2, y2, x3, y3, x4, y4, 0,0)+p_y*gvv_xterm1(x1, y1, x2, y2, x3, y3, x4, y4, 0,0)+p_x*gvv_yterm1(x1, y1, x2, y2, x3, y3, x4, y4, 0,0)


def gvv_term2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            d   = u1**2 - 2*u1*u2 + u2**2 + (v1 - v2)**2
            s1  = np.sqrt((1 - t)**2 * u1**2- 2*(1 - t)*t*u1*u2+ v1**2+ t**2*(u2**2 + (v1 - v2)**2)- 2*t*v1*(v1 - v2))
            s2  = np.sqrt(d)
            delta_v = v1 - t*v1 + t*v2
            term1 = t
            term2 = 0.5 * (u2 - u1) * s1 / d
            term3 = v1 * np.log(np.abs(delta_v)) / (v1 - v2)
            term4 = -0.5 * t * np.log(delta_v**2)
            expr3 = u1 - t*u1 + t*u2 + s1
            coef  = (-1 + t)*v1 - t*v2
            expr4 = (-u1**2 + t*u1**2 + u1*u2 - 2*t*u1*u2 + t*u2**2- v1**2 + t*v1**2 + v1*v2 - 2*t*v1*v2 + t*v2**2+ s1 * s2)
            term5 = (1/(v1 - v2)) * 0.5 * (t * (v2 - v1)+ coef * np.log(np.abs(expr3))+ (u2*v1 - u1*v2)/s2 * np.log(np.abs(expr4)))
            A1 = (u1**2 - u1*u2 - t*d + v1*(v1 - v2)) / (s1 * s2)
            A2 = ((-1 + t)*u1**2 + u1*(u2 - 2*t*u2) + t*(u2**2 + (v1 - v2)**2) + v1*(-v1 + v2)) / (s1 * s2)
            term6 = 0.25 * (v1 - v2) * (u1*v2 - u2*v1) / d**1.5 * (np.log(np.abs(1 + A1)) - np.log(np.abs(1 + A2)))
            return (term1 + term2 + term3 + term4 + term5 + term6)*(v2-v1)
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))

def gvv_xterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            #i can not find anti derivative
            d = u1**2 - 2*u1*u2 + u2**2 + (v1 - v2)**2
            sqrt_d = np.sqrt(d)
            E = ((1 - t)**2 * u1**2
                 - 2*(1 - t)*t * u1*u2
                 + v1**2
                 + t**2 * (u2**2 + (v1 - v2)**2)
                 - 2*t * v1 * (v1 - v2))
            sqrt_E = np.sqrt(E)

    # ArcTanh argümanı
            Z = (u1**2 - u1*u2 - t*d + v1*(v1 - v2)) / (sqrt_E * sqrt_d)
            atanhZ = np.arctanh(Z)

    # Katsayı
            coeff = (2*u1**4 - 4*u1**3*u2
                     + v1**2 *(-u2**2 + 2*(v1 - v2)**2)
                     + 2*u1*u2*v1 *(-2*v1 + 3*v2)
                     + u1**2*(2*u2**2 + 4*v1**2 - 4*v1*v2 - v2**2))

    # Birinci büyük grup
            T1 = -4*u1*(u1 - u2)*sqrt_E * d**1.5
            T2 =  t*(u1 - u2)**2* sqrt_E * d**1.5
            T3 = 2*u1*(u1 - u2)*(u1**2 - u1*u2 + v1*(v1 - v2))*d * atanhZ
            T4 = -2*u1**2 * d**2 * atanhZ
            T5 = (u1 - u2)**2*(3*(u1**2 - u1*u2 + v1*(v1 - v2))*sqrt_E*sqrt_d
                      + coeff*atanhZ)
            part1 = 0.25*(T1 + T2 + T3 + T4 + T5)/d**2.5

    # İkinci büyük grup
            alpha = u1 + t*(u2 - u1)
            beta  = np.abs(v1 + t*(v2 - v1))
            sum1  = alpha + sqrt_E

            log1 = np.log(sum1)
            log2 = np.log(beta)
            log3 = np.log(np.abs(-u1**2 + t*u1**2 + u1*u2 - 2*t*u1*u2 + t*u2**2
                         - v1**2 + t*v1**2 + v1*v2 - 2*t*v1*v2 + t*v2**2
                         + sqrt_E*sqrt_d))

            part2 = 0.5*(
                -t
                - t*log1
                + v1*log1/(v1 - v2)
                - 2*v1*log2/(v1 - v2)
                + t*2*log2
                + (-u2*v1 + u1*v2)/(sqrt_d*(v1 - v2)) * log3
                )

            return (part1 + part2)*(v2-v1)
    
           
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+p_x*gvv_term2(x1, y1, x2, y2, x3, y3, x4, y4,0, 0)
def gvv_yterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return 0
    
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+p_y*gvv_term2(x1,y1,x2,y2,x3,y3,x4,y4,0,0)

def gvv_xyterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return 0
    
    return (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+p_x*p_y*gvv_term2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y)+p_x*gvv_yterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y)+p_y*gvv_xterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y)

def gvv_term3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    return 0.5*np.abs(x1*y2+x2*y3+x3*y4+x4*y1-(y1*x2+y2*x3+y3*x4+y4*x1)     )
def gvv_xterm3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
def gvv_yterm3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
def gvv_xyterm3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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

def gvv_term4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
def gvv_xterm4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
def gvv_yterm4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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

def gvv_xyterm4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
def gvv_term5(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    p1=(   1-  0.339981  )   *0.5
    p2=(   1+ 0.339981  )   *0.5
    p3=(    1-0.861136   )  *0.5
    p4=(    1+0.861136   )  *0.5
    w1=0.652145
    w2=0.347855
    
    #x1=x1-p_x
    #x2=x2-p_x
    #x3=x3-p_x
    #x4=x4-p_x
    #y1=y1-p_y
    #y2=y2-p_y
    #y3=y3-p_y
    #y4=y4-p_y
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            u = u1 + t*(u2 - u1)
            v = v1 + t*(v2 - v1)
            return ((1/3)*u**3-xs*u**2+u*xs**2)   *   (v2-v1)*0.5
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )

def gvv_xterm5(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return (v2-v1)*0.5*(0.25*u**4  -  (2/3)*xs*u**3+0.5*xs**2*u**2)
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )
def gvv_yterm5(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return ((1/3)*u**3*v-u**2*v*xs+u*v*xs**2)*(v2-v1)*0.5
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )
def gvv_xyterm5(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return (v*((1/4)*u**4-(2/3)*u**3*xs+0.5*u**2*xs**2))*0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w1*f(x1,y1,x2,y2,p2)+w2*f(x1,y1,x2,y2,p3)+w2*f(x1,y1,x2,y2,p4)     )   +(w1*f(x2,y2,x3,y3,p1)+w1*f(x2,y2,x3,y3,p2) +w2*f(x2,y2,x3,y3,p3) +w2*f(x2,y2,x3,y3,p4)     ) + (w1*f(x3,y3,x4,y4,p1)+w1*f(x3,y3,x4,y4,p2)+w2*f(x3,y3,x4,y4,p3)+w2*f(x3,y3,x4,y4,p4)      )+(      w1*f(x4,y4,x1,y1,p1)+w1*f(x4,y4,x1,y1,p2)   +w2*f(x4,y4,x1,y1,p3)    +w2*f(x4,y4,x1,y1,p4)         )
def gvv_r3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
def gvv_xr3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
def gvv_yr3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
def gvv_xyr3(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
def gvv_r4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            res=(1/90.0)*(3*(u1 - t*u1 + t*u2)**6/(-u1 + u2) - 10*t**6*(u1 - u2)**3*(v1 - v2)**2 + 15*t**6*(-u1 + u2)*(v1 - v2)**4 + 18*(u1 - t*u1 + t*u2)**5*xs/(u1 - u2) - 36*t**5*(u1 - u2)**2*(v1 - v2)**2*xs - 180*t*u1**2*xs*(v1**2 + xs**2 - 2*v1*ys + ys**2) + 90*t*u1*(v1**2 + xs**2 - 2*v1*ys + ys**2)**2 + 60*t*u1**3*(v1**2 + 3*xs**2 - 2*v1*ys + ys**2) + 90*t**4*(u1 - u2)*(v1 - v2)*xs*(u1*(2*v1 - v2 - ys) + u2*(-v1 + ys)) + 12*t**5*(u1 - u2)**2*(v1 - v2)*(u1*(5*v1 - 3*v2 - 2*ys) + 2*u2*(-v1 + ys)) + 18*t**5*(v1 - v2)**3*(u1*(5*v1 - v2 - 4*ys) + 4*u2*(-v1 + ys)) - 45*t**2*(v1**2 + xs**2 - 2*v1*ys + ys**2)*(-u2*(v1**2 + xs**2 - 2*v1*ys + ys**2) + u1*(5*v1**2 - 4*v1*v2 + xs**2 - 6*v1*ys + 4*v2*ys + ys**2)) - 60*t**3*xs*(u2**2*(v1**2 + xs**2 - 2*v1*ys + ys**2) + u1**2*(6*v1**2 + v2**2 + xs**2 + 4*v2*ys + ys**2 - 6*v1*(v2 + ys)) - 2*u1*u2*(3*v1**2 + xs**2 + 2*v2*ys + ys**2 - 2*v1*(v2 + 2*ys))) - 15*t**4*(u1 - u2)*(u2**2*(v1**2 + 3*xs**2 - 2*v1*ys + ys**2) - 2*u1*u2*(4*v1**2 - 3*v1*v2 + 3*xs**2 - 5*v1*ys + 3*v2*ys + ys**2) + u1**2*(10*v1**2 + 3*v2**2 + 3*xs**2 + 6*v2*ys + ys**2 - 4*v1*(3*v2 + 2*ys))) + 180*t**2*u1*xs*(-u2*(v1**2 + xs**2 - 2*v1*ys + ys**2) + u1*(2*v1**2 + xs**2 + ys*(v2 + ys) - v1*(v2 + 3*ys))) + 20*t**3*u1*(3*u2**2*(v1**2 + 3*xs**2 - 2*v1*ys + ys**2) - 6*u1*u2*(2*v1**2 + 3*xs**2 + ys*(v2 + ys) - v1*(v2 + 3*ys)) + u1**2*(10*v1**2 + v2**2 + 9*xs**2 + 6*v2*ys + 3*ys**2 - 4*v1*(2*v2 + 3*ys))) - 45*t**4*(v1 - v2)**2*(-u2*(3*v1**2 + xs**2 - 6*v1*ys + 3*ys**2) + u1*(5*v1**2 + xs**2 + 2*v2*ys + 3*ys**2 - 2*v1*(v2 + 4*ys))) - 30*t**2*u1**2*(-3*u2*(v1**2 + 3*xs**2 - 2*v1*ys + ys**2) + u1*(5*v1**2 + 9*xs**2 + 2*v2*ys + 3*ys**2 - 2*v1*(v2 + 4*ys))) + 60*t**3*(v1 - v2)*(-2*u2*(v1 - ys)*(v1**2 + xs**2 - 2*v1*ys + ys**2) + u1*(5*v1**3 - 3*v1**2*(v2 + 4*ys) - 2*ys*(xs**2 + ys**2) - v2*(xs**2 + 3*ys**2) + 3*v1*(xs**2 + ys*(2*v2 + 3*ys)))))
            return res*(v2-v1)
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))

def gvv_xr4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            res=(1/30.0)*((5/7)*t**7*(u1-u2)**2*(u1**4-4*u1**3*u2+u2**4+3*u1**2*(2*u2**2+(v1-v2)**2)-2*u1*u2*(2*u2**2+3*(v1-v2)**2)+3*u2**2*(v1-v2)**2+3*(v1-v2)**4) - (1/3)*t**6*(u1-u2)*(15*u1**5-12*u1**4*(5*u2+xs)-u1**2*(60*u2**3+72*u2**2*xs+20*(v1-v2)**2*xs+15*u2*(v1-v2)*(7*v1-4*v2-3*ys))+u1*(15*u2**4+48*u2**3*xs+40*u2*(v1-v2)**2*xs+15*u2**2*(v1-v2)*(5*v1-2*v2-3*ys)+15*(v1-v2)**3*(3*v1-v2-2*ys))-u2*(12*u2**3*xs+20*u2*(v1-v2)**2*xs+15*u2**2*(v1-v2)*(v1-ys)+30*(v1-v2)**3*(v1-ys))+3*u1**3*(30*u2**2+16*u2*xs+5*(v1-v2)*(3*v1-2*v2-ys))) + t*u1**2*(5*u1**4-24*u1**3*xs-40*u1*xs*(v1**2+xs**2-2*v1*ys+ys**2)+15*(v1**2+xs**2-2*v1*ys+ys**2)**2+15*u1**2*(v1**2+3*xs**2-2*v1*ys+ys**2)) + t**5*(15*u1**6-12*u1**5*(5*u2+2*xs)+3*u1**4*(30*u2**2+15*v1**2+6*v2**2+32*u2*xs+3*xs**2+8*v2*ys+ys**2-10*v1*(2*v2+ys))+u2**2*(16*u2*(v1-v2)*xs*(v1-ys)+3*u2**2*(v1**2+3*xs**2-2*v1*ys+ys**2)+6*(v1-v2)**2*(3*v1**2+xs**2-6*v1*ys+3*ys**2))+3*u1**2*(5*u2**4+32*u2**3*xs+16*u2*(v1-v2)*xs*(2*v1-v2-ys)+6*u2**2*(6*v1**2+v2**2+3*xs**2+4*v2*ys+ys**2-6*v1*(v2+ys))+(v1-v2)**2*(15*v1**2+v2**2+2*xs**2+8*v2*ys+6*ys**2-10*v1*(v2+2*ys)))-4*u1**3*(15*u2**3+36*u2**2*xs+2*(v1-v2)*xs*(5*v1-3*v2-2*ys)+3*u2*(10*v1**2+3*v2**2+3*xs**2+6*v2*ys+ys**2-4*v1*(3*v2+2*ys)))-12*u1*u2*(2*u2**3*xs+2*u2*(v1-v2)*xs*(3*v1-v2-2*ys)+u2**2*(3*v1**2+3*xs**2+2*v2*ys+ys**2-2*v1*(v2+2*ys))+(v1-v2)**2*(5*v1**2+xs**2+2*v2*ys+3*ys**2-2*v1*(v2+4*ys)))) - 5*t**2*u1*(3*u1**5-3*u1**4*(u2+4*xs)-3*u2*(v1**2+xs**2-2*v1*ys+ys**2)**2+3*u1*(v1**2+xs**2-2*v1*ys+ys**2)*(3*v1**2+4*u2*xs+xs**2+2*v2*ys+ys**2-2*v1*(v2+2*ys))+3*u1**3*(3*v1**2+4*u2*xs+6*xs**2+v2*ys+2*ys**2-v1*(v2+5*ys))-2*u1**2*(3*u2*(v1**2+3*xs**2-2*v1*ys+ys**2)+2*xs*(5*v1**2+3*xs**2+2*v2*ys+3*ys**2-2*v1*(v2+4*ys)))) + (5/3)*t**3*(15*u1**6-6*u1**5*(5*u2+8*xs)+3*u2**2*(v1**2+xs**2-2*v1*ys+ys**2)**2-6*u1*u2*(v1**2+xs**2-2*v1*ys+ys**2)*(5*v1**2-4*v1*v2+4*u2*xs+xs**2-6*v1*ys+4*v2*ys+ys**2)+3*u1**4*(5*u2**2+15*v1**2+v2**2+32*u2*xs+18*xs**2+8*v2*ys+6*ys**2-10*v1*(v2+2*ys))+3*u1**2*(15*v1**4+2*v2**2*xs**2+xs**4+8*v2*xs**2*ys+6*v2**2*ys**2+2*xs**2*ys**2+8*v2*ys**3+ys**4-20*v1**3*(v2+2*ys)+6*u2**2*(v1**2+3*xs**2-2*v1*ys+ys**2)+6*v1**2*(v2**2+2*xs**2+8*v2*ys+6*ys**2)-12*v1*(v2*xs**2+v2**2*ys+xs**2*ys+3*v2*ys**2+ys**3)+16*u2*xs*(2*v1**2+xs**2+ys*(v2+ys)-v1*(v2+3*ys)))-4*u1**3*(12*u2**2*xs+3*u2*(5*v1**2+9*xs**2+ys*(2*v2+3*ys)-2*v1*(v2+4*ys))+2*xs*(10*v1**2+v2**2+6*v2*ys-4*v1*(2*v2+3*ys)+3*(xs**2+ys**2)))) - 5*t**4*(5*u1**6-3*u1**5*(5*u2+4*xs)+3*u1**4*(5*u2**2+5*v1**2+v2**2+12*u2*xs+3*xs**2+3*v2*ys+ys**2-5*v1*(v2+ys))+u2**2*(v1**2+xs**2-2*v1*ys+ys**2)*(3*v1**2+2*u2*xs+3*v2*ys-3*v1*(v2+ys))-u1**3*(5*u2**3+36*u2**2*xs+2*xs*(10*v1**2+3*v2**2+xs**2+6*v2*ys+ys**2-4*v1*(3*v2+2*ys))+3*u2*(10*v1**2+v2**2+9*xs**2+6*v2*ys+3*ys**2-4*v1*(2*v2+3*ys)))+3*u1**2*(4*u2**3*xs+2*u2*xs*(6*v1**2+v2**2+xs**2+4*v2*ys+ys**2-6*v1*(v2+ys))+3*u2**2*(2*v1**2+3*xs**2+ys*(v2+ys)-v1*(v2+3*ys))+(v1-v2)*(5*v1**3-v2**2*ys-5*v1**2*(v2+2*ys)-ys*(xs**2+ys**2)-v2*(xs**2+3*ys**2)+v1*(v2**2+2*xs**2+8*v2*ys+6*ys**2)))-3*u1*u2*(u2**2*(v1**2+3*xs**2-2*v1*ys+ys**2)+2*u2*xs*(3*v1**2+xs**2+ys*(2*v2+ys)-2*v1*(v2+2*ys))+(v1-v2)*(5*v1**3-3*v1**2*(v2+4*ys)-2*ys*(xs**2+ys**2)-v2*(xs**2+3*ys**2)+3*v1*(xs**2+ys*(2*v2+3*ys))))) )
            return res*(v2-v1)
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))

def gvv_yr4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            res=(t/630.0)*(3*u1**5*(6*(7-21*t+35*t**2-35*t**3+21*t**4-7*t**5+t**6)*v1 + t*(21-70*t+105*t**2-84*t**3+35*t**4-6*t**5)*v2) - 3*u1**4*(30*t**6*u2*(v1-v2) + 210*v1*xs + 42*t**4*(10*u2*v1-6*u2*v2+5*v1*xs-4*v2*xs) + 70*t**2*(5*u2*v1-u2*v2+10*v1*xs-4*v2*xs) - 105*t**3*(5*u2*v1-2*u2*v2+5*v1*xs-3*v2*xs) - 35*t**5*(5*u2*v1-4*u2*v2+v1*xs-v2*xs) - 105*t*(u2*v1+5*v1*xs-v2*xs)) + u1**3*(60*t**6*(3*u2**2+(v1-v2)**2)*(v1-v2) - 70*t**5*(3*u2**2*(4*v1-3*v2) + 6*u2*(v1-v2)*xs + (v1-v2)**2*(6*v1-3*v2-2*ys)) + 420*v1*(v1**2+3*xs**2-2*v1*ys+ys**2) - 105*t**3*(20*v1**3 + 3*u2**2*(4*v1-v2) + 18*u2*(2*v1-v2)*xs - 10*v1**2*(3*v2+2*ys) + 4*v1*(3*v2**2+3*xs**2+6*v2*ys+ys**2) - v2*(v2**2+9*xs**2+6*v2*ys+3*ys**2)) + 140*t**2*(3*u2**2*v1 + 15*v1**3 + 6*u2*(4*v1-v2)*xs - 5*v1**2*(3*v2+4*ys) - v2*(9*xs**2+2*v2*ys+3*ys**2) + v1*(3*v2**2+18*xs**2+16*v2*ys+6*ys**2)) - 210*t*(6*v1**3 + 6*u2*v1*xs - v1**2*(3*v2+10*ys) - v2*(3*xs**2+ys**2) + 4*v1*(3*xs**2+ys*(v2+ys))) + 84*t**4*(9*u2**2*(2*v1-v2) + 6*u2*(4*v1-3*v2)*xs + (v1-v2)*(15*v1**2+3*v2**2+3*xs**2+6*v2*ys+ys**2-5*v1*(3*v2+2*ys)))) + t*u2*(-6*t**5*(3*u2**4+10*u2**2*(v1-v2)**2+15*(v1-v2)**4)*(v1-v2) + 7*t**4*(3*u2**4*v1 + 15*u2**3*(v1-v2)*xs + 30*u2*(v1-v2)**3*xs + 15*(v1-v2)**4*(5*v1-4*ys) + 10*u2**2*(v1-v2)**2*(3*v1-2*ys)) + 315*v1*(v1**2+xs**2-2*v1*ys+ys**2)**2 - 210*t*(v1**2+xs**2-2*v1*ys+ys**2)*(5*v1**3 - v1**2*(5*v2+6*ys) - v2*(xs**2+ys**2) + v1*(2*u2*xs+xs**2+6*v2*ys+ys**2)) - 42*t**3*(3*u2**3*v1*xs + 6*u2*(v1-v2)**2*xs*(3*v1-2*ys) + 2*u2**2*(v1-v2)*(3*v1**2+3*xs**2-4*v1*ys+ys**2) + 6*(v1-v2)**3*(5*v1**2+xs**2-8*v1*ys+3*ys**2)) + 105*t**2*(3*u2*(v1-v2)*xs*(3*v1**2+xs**2-4*v1*ys+ys**2) + u2**2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2) + 3*(v1-v2)**2*(5*v1**3 - 12*v1**2*ys - 2*ys*(xs**2+ys**2) + 3*v1*(xs**2+3*ys**2)))) - 3*u1**2*(60*t**6*u2*(u2**2+(v1-v2)**2)*(v1-v2) + 420*v1*xs*(v1**2+xs**2-2*v1*ys+ys**2) - 70*t**5*(u2**3*(3*v1-2*v2) + 3*u2**2*(v1-v2)*xs + (v1-v2)**3*xs + u2*(v1-v2)**2*(5*v1-2*(v2+ys))) + 84*t**4*(u2**3*(3*v1-v2) + u2**2*(9*v1*xs-6*v2*xs) + (v1-v2)**2*xs*(5*v1-2*(v2+ys)) + u2*(v1-v2)*(10*v1**2+v2**2+3*xs**2+4*v2*ys+ys**2-8*v1*(v2+ys))) - 210*t*(u2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2) + xs*(5*v1**3+3*v1*xs**2+v1*ys*(4*v2+3*ys) - v1**2*(3*v2+8*ys) - v2*(xs**2+ys**2))) + 140*t**2*(3*u2**2*v1*xs + u2*(5*v1**3+9*v1*xs**2+v1*ys*(4*v2+3*ys) - v1**2*(3*v2+8*ys) - v2*(3*xs**2+ys**2)) + xs*(10*v1**3 - 12*v1**2*(v2+ys) + 3*v1*(v2**2+xs**2+4*v2*ys+ys**2) - 2*v2*(xs**2+ys*(v2+ys)))) - 105*t**3*(u2**3*v1 + u2**2*(9*v1*xs-3*v2*xs) + (v1-v2)*xs*(10*v1**2+v2**2+xs**2+4*v2*ys+ys**2-8*v1*(v2+ys)) + u2*(10*v1**3 - 12*v1**2*(v2+ys) + 3*v1*(v2**2+3*xs**2+4*v2*ys+ys**2) - 2*v2*(3*xs**2+ys*(v2+ys))))) + 3*u1*(30*t**6*(u2**2+(v1-v2)**2)**2*(v1-v2) - 35*t**5*(u2**2+(v1-v2)**2)*(u2**2*(2*v1-v2) + 4*u2*(v1-v2)*xs + (v1-v2)**2*(6*v1-v2-4*ys)) + 210*v1*(v1**2+xs**2-2*v1*ys+ys**2)**2 - 105*t*(v1**2+xs**2-2*v1*ys+ys**2)*(6*v1**3 - v1**2*(5*v2+8*ys) - v2*(xs**2+ys**2) + 2*v1*(2*u2*xs+xs**2+3*v2*ys+ys**2)) + 42*t**4*(u2**4*v1 + u2**3*(8*v1*xs-4*v2*xs) + 4*u2*(v1-v2)**2*xs*(4*v1-v2-2*ys) + 2*u2**2*(v1-v2)*(6*v1**2+3*xs**2+ys*(2*v2+ys) - 3*v1*(v2+2*ys)) + (v1-v2)**3*(15*v1**2 - 5*v1*(v2+4*ys) + 2*(xs**2 + ys*(2*v2+3*ys)))) - 105*t**3*(2*u2**3*v1*xs + 2*u2*(v1-v2)*xs*(6*v1**2+xs**2+ys*(2*v2+ys) - 3*v1*(v2+2*ys)) + u2**2*(4*v1**3 + 6*v1*xs**2 + 2*v1*ys*(2*v2+ys) - 3*v1**2*(v2+2*ys) - v2*(3*xs**2+ys**2)) + (v1-v2)**2*(10*v1**3 - 5*v1**2*(v2+4*ys) - 2*ys*(xs**2+ys**2) - v2*(xs**2+3*ys**2) + 4*v1*(xs**2+ys*(2*v2+3*ys)))) + 70*t**2*(2*u2**2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2) + 4*u2*xs*(4*v1**3 - 3*v1**2*(v2+2*ys) - v2*(xs**2+ys**2) + 2*v1*(xs**2+2*v2*ys+ys**2)) + (v1-v2)*(15*v1**4 - 10*v1**3*(v2+4*ys) + (xs**2+ys**2)*(xs**2 + ys*(4*v2+ys)) + 12*v1**2*(xs**2 + ys*(2*v2+3*ys)) - 6*v1*(2*ys*(xs**2+ys**2) + v2*(xs**2+3*ys**2))))) )
            return res*(v2-v1)
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))
def gvv_xyr4(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
    def f(u1, v1, u2, v2, t):
        if v2-v1==0:
            return 0
        else:
            xs=p_x
            ys=p_y
            res=(1/30.0)*(-(5/8)*t**8*(u1-u2)**2*(u1**4-4*u1**3*u2+u2**4+3*u1**2*(2*u2**2+(v1-v2)**2)-2*u1*u2*(2*u2**2+3*(v1-v2)**2)+3*u2**2*(v1-v2)**2+3*(v1-v2)**4)*(v1-v2) + t*u1**2*v1*(5*u1**4-24*u1**3*xs-40*u1*xs*(v1**2+xs**2-2*v1*ys+ys**2)+15*(v1**2+xs**2-2*v1*ys+ys**2)**2+15*u1**2*(v1**2+3*xs**2-2*v1*ys+ys**2)) + (1/7)*t**7*(u1-u2)*(5*u1**5*(7*v1-6*v2)+u1**4*(-5*u2*(29*v1-24*v2)+24*(-v1+v2)*xs)+u1**2*(u2**3*(-170*v1+120*v2)-144*u2**2*(v1-v2)*xs-40*(v1-v2)**3*xs-15*u2*(v1-v2)**2*(17*v1-8*v2-6*ys))+u2*(-5*u2**4*v1-24*u2**3*(v1-v2)*xs-40*u2*(v1-v2)**3*xs-15*(v1-v2)**4*(5*v1-4*ys)-15*u2**2*(v1-v2)**2*(3*v1-2*ys))+u1**3*(10*u2**2*(23*v1-18*v2)+96*u2*(v1-v2)*xs+15*(v1-v2)**2*(7*v1-2*(2*v2+ys)))+u1*(5*u2**4*(11*v1-6*v2)+96*u2**3*(v1-v2)*xs+80*u2*(v1-v2)**3*xs+15*u2**2*(v1-v2)**2*(13*v1-4*v2-6*ys)+15*(v1-v2)**4*(7*v1-2*(v2+2*ys)))) - 0.5*t**2*u1*(5*u1**5*(7*v1-v2)-6*u1**4*(5*u2*v1+24*v1*xs-4*v2*xs)-30*u2*v1*(v1**2+xs**2-2*v1*ys+ys**2)**2+15*u1**3*(7*v1**3-3*v1**2*(v2+4*ys)-v2*(3*xs**2+ys**2)+v1*(8*u2*xs+15*xs**2+4*v2*ys+5*ys**2))+15*u1*(v1**2+xs**2-2*v1*ys+ys**2)*(7*v1**3+8*u2*v1*xs-5*v1**2*(v2+2*ys)-v2*(xs**2+ys**2)+3*v1*(xs**2+ys*(2*v2+ys)))-20*u1**2*(3*u2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2)+2*xs*(6*v1**3-v1**2*(3*v2+10*ys)-v2*(xs**2+ys**2)+4*v1*(xs**2+ys*(v2+ys))))) - (1/6)*t**6*(15*u1**6*(7*v1-5*v2)-6*u1**5*(75*u2*v1-50*u2*v2+24*v1*xs-20*v2*xs)+u2**2*(24*u2**3*v1*xs+40*u2*(v1-v2)**2*xs*(3*v1-2*ys)+15*u2**2*(v1-v2)*(3*v1**2+3*xs**2-4*v1*ys+ys**2)+30*(v1-v2)**3*(5*v1**2+xs**2-8*v1*ys+3*ys**2))+15*u1**4*(10*u2**2*(5*v1-3*v2)+8*u2*(5*v1-4*v2)*xs+(v1-v2)*(21*v1**2+6*v2**2+3*xs**2+8*v2*ys+ys**2-12*v1*(2*v2+ys)))+15*u1**2*(5*u2**4*(3*v1-v2)+16*u2**3*(3*v1-2*v2)*xs+8*u2*(v1-v2)**2*xs*(5*v1-2*(v2+ys))+6*u2**2*(v1-v2)*(10*v1**2+v2**2+3*xs**2+4*v2*ys+ys**2-8*v1*(v2+ys))+(v1-v2)**3*(21*v1**2+v2**2+2*xs**2+8*v2*ys+6*ys**2-12*v1*(v2+2*ys))) -20*u1**3*(15*u2**3*(2*v1-v2)+12*u2**2*(4*v1-3*v2)*xs+2*(v1-v2)**2*xs*(6*v1-3*v2-2*ys)+3*u2*(v1-v2)*(15*v1**2+3*v2**2+3*xs**2+6*v2*ys+ys**2-5*v1*(3*v2+2*ys))) -30*u1*u2*(u2**4*v1+u2**3*(8*v1*xs-4*v2*xs)+4*u2*(v1-v2)**2*xs*(4*v1-v2-2*ys)+2*u2**2*(v1-v2)*(6*v1**2+3*xs**2+ys*(2*v2+ys)-3*v1*(v2+2*ys))+(v1-v2)**3*(15*v1**2-5*v1*(v2+4*ys)+2*(xs**2+ys*(2*v2+3*ys))))) + t**5*(5*u1**6*(7*v1-4*v2)-12*u1**5*(10*u2*v1-5*u2*v2+6*v1*xs-4*v2*xs)+3*u1**4*(35*v1**3+10*u2**2*(5*v1-2*v2)+16*u2*(5*v1-3*v2)*xs-30*v1**2*(2*v2+ys)-4*v2*(v2**2+3*xs**2+3*v2*ys+ys**2)+5*v1*(6*v2**2+3*xs**2+8*v2*ys+ys**2))+u2**2*(8*u2*(v1-v2)*xs*(3*v1**2+xs**2-4*v1*ys+ys**2)+3*u2**2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2)+6*(v1-v2)**2*(5*v1**3-12*v1**2*ys-2*ys*(xs**2+ys**2)+3*v1*(xs**2+3*ys**2))) -4*u1**3*(5*u2**3*(4*v1-v2)+36*u2**2*(2*v1-v2)*xs+2*(v1-v2)*xs*(15*v1**2+3*v2**2+xs**2+6*v2*ys+ys**2-5*v1*(3*v2+2*ys))+3*u2*(20*v1**3-10*v1**2*(3*v2+2*ys)+4*v1*(3*v2**2+3*xs**2+6*v2*ys+ys**2)-v2*(v2**2+9*xs**2+6*v2*ys+3*ys**2))) + 3*u1**2*(5*u2**4*v1+16*u2**3*(3*v1-v2)*xs+8*u2*(v1-v2)*xs*(10*v1**2+v2**2+xs**2+4*v2*ys+ys**2-8*v1*(v2+ys))+(v1-v2)**2*(35*v1**3-30*v1**2*(v2+2*ys)+5*v1*(v2**2+2*xs**2+8*v2*ys+6*ys**2)-4*(v2*xs**2+v2**2*ys+xs**2*ys+3*v2*ys**2+ys**3))+6*u2**2*(10*v1**3-12*v1**2*(v2+ys)+3*v1*(v2**2+3*xs**2+4*v2*ys+ys**2)-2*v2*(3*xs**2+ys*(v2+ys)))) -12*u1*u2*(2*u2**3*v1*xs+2*u2*(v1-v2)*xs*(6*v1**2+xs**2+ys*(2*v2+ys)-3*v1*(v2+2*ys))+u2**2*(4*v1**3+6*v1*xs**2+2*v1*ys*(2*v2+ys)-3*v1**2*(v2+2*ys)-v2*(3*xs**2+ys**2))+(v1-v2)**2*(10*v1**3-5*v1**2*(v2+4*ys)-2*ys*(xs**2+ys**2)-v2*(xs**2+3*ys**2)+4*v1*(xs**2+ys*(2*v2+3*ys))))) + (5/3)*t**3*(3*u1**6*(7*v1-2*v2)-6*u1**5*(6*u2*v1-u2*v2+12*v1*xs-4*v2*xs)+3*u2**2*v1*(v1**2+xs**2-2*v1*ys+ys**2)**2 -6*u1*u2*(v1**2+xs**2-2*v1*ys+ys**2)*(6*v1**3 - v1**2*(5*v2+8*ys) - v2*(xs**2+ys**2) + 2*v1*(2*u2*xs+xs**2+3*v2*ys+ys**2)) + 3*u1**4*(5*u2**2*v1+21*v1**3+8*u2*(5*v1-v2)*xs-6*v1**2*(3*v2+5*ys)-2*v2*(6*xs**2+ys*(v2+2*ys))+v1*(3*v2**2+20*v2*ys+10*(3*xs**2+ys**2))) + 3*u1**2*(21*v1**5-30*v1**4*(v2+2*ys)+6*u2**2*v1*(v1**2+3*xs**2-2*v1*ys+ys**2)+10*v1**3*(v2**2+2*xs**2+8*v2*ys+6*ys**2)-24*v1**2*(v2*xs**2+v2**2*ys+xs**2*ys+3*v2*ys**2+ys**3) - 2*v2*(xs**2+ys**2)*(xs**2+ys*(2*v2+ys)) + 8*u2*xs*(5*v1**3+3*v1*xs**2+v1*ys*(4*v2+3*ys)-v1**2*(3*v2+8*ys)-v2*(xs**2+ys**2)) + 3*v1*(8*v2*ys*(xs**2+ys**2) + (xs**2+ys**2)**2 + 2*v2**2*(xs**2+3*ys**2))) - 4*u1**3*(12*u2**2*v1*xs + 3*u2*(6*v1**3 - v1**2*(3*v2+10*ys) - v2*(3*xs**2+ys**2) + 4*v1*(3*xs**2 + ys*(v2+ys))) + 2*xs*(15*v1**3 - 5*v1**2*(3*v2+4*ys) - v2*(3*xs**2 + ys*(2*v2+3*ys)) + v1*(3*v2**2 + 16*v2*ys + 6*(xs**2 + ys**2))))) - (5/4)*t**4*(5*u1**6*(7*v1-3*v2) - 6*u1**5*(15*u2*v1 - 5*u2*v2 + 16*v1*xs - 8*v2*xs) + 3*u1**4*(35*v1**3 + 5*u2**2*(5*v1 - v2) + 16*u2*(5*v1 - 2*v2)*xs - 5*v1**2*(9*v2 + 8*ys) + 5*v1*(3*v2**2 + 6*xs**2 + 8*v2*ys + 2*ys**2) - v2*(v2**2 + 18*xs**2 + 8*v2*ys + 6*ys**2)) + u2**2*(v1**2+xs**2-2*v1*ys+ys**2)*(15*v1**3 + 8*u2*v1*xs - 3*v1**2*(5*v2+6*ys) - 3*v2*(xs**2+ys**2) + 3*v1*(xs**2 + ys*(6*v2+ys))) + 3*u1**2*(16*u2**3*v1*xs + 6*u2**2*(5*v1**3 + 9*v1*xs**2 + v1*ys*(4*v2+3*ys) - v1**2*(3*v2+8*ys) - v2*(3*xs**2+ys**2)) + (v1-v2)*(35*v1**4 - 40*v1**3*(v2+2*ys) + 8*v2*ys*(xs**2+ys**2) + (xs**2+ys**2)**2 + 2*v2**2*(xs**2+3*ys**2) + 10*v1**2*(v2**2+2*xs**2+8*v2*ys+6*ys**2) - 16*v1*(v2*xs**2 + v2**2*ys + xs**2*ys + 3*v2*ys**2 + ys**3)) + 8*u2*xs*(10*v1**3 - 12*v1**2*(v2+ys) + 3*v1*(v2**2+xs**2+4*v2*ys+ys**2) - 2*v2*(xs**2+ys*(v2+ys)))) - 4*u1**3*(5*u2**3*v1 + 12*u2**2*(4*v1 - v2)*xs + 3*u2*(15*v1**3 - 5*v1**2*(3*v2 + 4*ys) - v2*(9*xs**2 + 2*v2*ys + 3*ys**2) + v1*(3*v2**2 + 18*xs**2 + 16*v2*ys + 6*ys**2)) + 2*xs*(20*v1**3 - 10*v1**2*(3*v2 + 2*ys) + 4*v1*(3*v2**2 + xs**2 + 6*v2*ys + ys**2) - v2*(v2**2 + 6*v2*ys + 3*(xs**2 + ys**2)))) - 6*u1*u2*(2*u2**2*v1*(v1**2 + 3*xs**2 - 2*v1*ys + ys**2) + 4*u2*xs*(4*v1**3 - 3*v1**2*(v2+2*ys) - v2*(xs**2 + ys**2) + 2*v1*(xs**2 + 2*v2*ys + ys**2)) + (v1-v2)*(15*v1**4 - 10*v1**3*(v2+4*ys) + (xs**2+ys**2)*(xs**2 + ys*(4*v2+ys)) + 12*v1**2*(xs**2 + ys*(2*v2+3*ys)) - 6*v1*(2*ys*(xs**2+ys**2) + v2*(xs**2 + 3*ys**2))))))
            return res*(v2-v1)
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))
def gvv_series2_term1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            u = u1 + t*(u2 - u1)-xs
            v = v1 + t*(v2 - v1)-ys
            R2 = u**2 + v**2 
            r  = np.sqrt(R2)
            return (0.5*(-u)*r-2.5*v**2*np.log(r-u))*0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )

def gvv_series2_xterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return ((1/6)*r*(-2*(x+xs)**2+16*(y+ys)**2+xs*(x+xs)+xs**2-32*ys*(y+ys)+16*ys**2)-2.5*xs*y**2*np.log(r-x))*0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )


def gvv_series2_yterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return ((y+ys)*(-0.5*x*r-2.5*y**2*np.log(r-x)))*0.5*(v2-v1)
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )


def gvv_series2_xyterm1(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return    (v2-v1)*0.5* yy*  ( (1/6)*r*(  -2*xx**2+16*yy**2+xs*xx+xs**2-32*ys*yy+16*ys**2)-2.5*xs*y**2*np.log(r-x)        )
    return  ( w1*f(x1,y1,x2,y2,p1)+w2*f(x1,y1,x2,y2,p2)+w3*f(x1,y1,x2,y2,p3)+w4*f(x1,y1,x2,y2,p4) +w5*f(x1,y1,x2,y2,p5)     )   +(w1*f(x2,y2,x3,y3,p1)+w2*f(x2,y2,x3,y3,p2) +w3*f(x2,y2,x3,y3,p3) +w4*f(x2,y2,x3,y3,p4)  +w5*f(x2,y2,x3,y3,p5)     ) + (w1*f(x3,y3,x4,y4,p1)+w2*f(x3,y3,x4,y4,p2)+w3*f(x3,y3,x4,y4,p3)+w4*f(x3,y3,x4,y4,p4) +w5*f(x3,y3,x4,y4,p5)     )+(      w1*f(x4,y4,x1,y1,p1)+w2*f(x4,y4,x1,y1,p2)   +w3*f(x4,y4,x1,y1,p3)    +w4*f(x4,y4,x1,y1,p4)   +w5*f(x4,y4,x1,y1,p5)         )

def gvv_series2_term2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return (-(u1*(1-t) + u2*t)**4/(12*(u2 - u1)) + 2*t*u1*v1**2 + 0.5*t**4*(u2 - u1)*(v1 - v2)**2 - t**2*v1*(3*u1*v1 - u2*v1 - 2*u1*v2) + 2/3*t**3*(v1 - v2)*(3*u1*v1 - 2*u2*v1 - u1*v2))*(v2-v1)
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))

def gvv_series2_xterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return (v2-v1)*( (-(u1*(1-t)+t*u2)**5/(20*(u2-u1)) + t*u1**2*v1**2+ 1/5*t**5*(u1-u2)**2*(v1-v2)**2- t**2*u1*v1*(2*u1*v1 - u2*v1 - u1*v2)- 0.5*t**4*(u1-u2)*(v1-v2)*(2*u1*v1 - u2*v1 - u1*v2)+ 1/3*t**3*(6*u1**2*v1**2 - 6*u1*u2*v1**2 + u2**2*v1**2- 6*u1**2*v1*v2 + 4*u1*u2*v1*v2 + u1**2*v2**2))    )
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+gvv_series2_term2(x1,y1,x2,y2,x3,y3,x4,y4,0,0)*p_x


def gvv_series2_yterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return (v2-v1)*(  (-(1/3)*t*u1**3*v1 + 2*t*u1*v1**3 - 1/15*t**5*(u1-u2)**3*(v1-v2) + 2/5*t**5*(u1-u2)*(v1-v2)**3 + 1/12*t**4*(u1-u2)**2*(4*u1*v1 - u2*v1 - 3*u1*v2) - t**2*v1**2*(4*u1*v1 - u2*v1 - 3*u1*v2) + 1/6*t**2*u1**2*(4*u1*v1 - 3*u2*v1 - u1*v2) - 1/2*t**4*(v1-v2)**2*(4*u1*v1 - 3*u2*v1 - u1*v2) - 1/3*t**3*u1*(u1-u2)*(2*u1*v1 - u2*v1 - u1*v2) + 2*t**3*v1*(v1-v2)*(2*u1*v1 - u2*v1 - u1*v2))    )
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+gvv_series2_term2(x1,y1,x2,y2,x3,y3,x4,y4,0,0)*p_y

def gvv_series2_xyterm2(x1,y1,x2,y2,x3,y3,x4,y4,p_x,p_y):
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
            return (v2-v1)*(   (-(1/4)*t*u1**4*v1 + t*u1**2*v1**3 + 1/24*t**6*(u1-u2)**4*(v1-v2) - 1/6*t**6*(u1-u2)**2*(v1-v2)**3 - 1/20*t**5*(u1-u2)**3*(5*u1*v1 - u2*v1 - 4*u1*v2) + 1/8*t**4*u1*(u1-u2)**2*(5*u1*v1 - 2*u2*v1 - 3*u1*v2) - 1/2*t**2*u1*v1**2*(5*u1*v1 - 2*u2*v1 - 3*u1*v2) - 1/6*t**3*u1**2*(u1-u2)*(5*u1*v1 - 3*u2*v1 - 2*u1*v2) + 1/5*t**5*(u1-u2)*(v1-v2)**2*(5*u1*v1 - 3*u2*v1 - 2*u1*v2) + 1/8*t**2*u1**3*(5*u1*v1 - 4*u2*v1 - u1*v2) + 1/4*t**4*(v2-v1)*(10*u1**2*v1**2 - 12*u1*u2*v1**2 + 3*u2**2*v1**2 - 8*u1**2*v1*v2 + 6*u1*u2*v1*v2 + u1**2*v2**2) + 1/3*t**3*v1*(10*u1**2*v1**2 - 8*u1*u2*v1**2 + u2**2*v1**2 - 12*u1**2*v1*v2 + 6*u1*u2*v1*v2 + 3*u1**2*v2**2))   )
    return  (f(x1,y1,x2,y2,1)-f(x1,y1,x2,y2,0))+(f(x2,y2,x3,y3,1)-f(x2,y2,x3,y3,0)) + (f(x3,y3,x4,y4,1)-f(x3,y3,x4,y4,0))+(f(x4,y4,x1,y1,1)-f(x4,y4,x1,y1,0))+gvv_series2_term2(x1,y1,x2,y2,x3,y3,x4,y4,0,0)*p_y*p_x+p_x*gvv_series2_yterm2(x1,y1,x2,y2,x3,y3,x4,y4,0,0)+p_y*gvv_series2_xterm2(x1,y1,x2,y2,x3,y3,x4,y4,0,0)


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
print(gvv_term1(x1, y1, x2, y2, x3, y3, x4, y4, xs2,ys2))

x1=-1
y1=-1
x2=1
y2=-1
x3=1
y3=1
x4=-1
y4=1
xs1=-3**(-0.5)*0+0.2
ys1=-3**(-0.5)*0+0.2
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
#print(gvv_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1))
#print(gvv_xterm2(x1, y1, x2, y2, x3, y3, x4, y4, xs1, ys1))
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


