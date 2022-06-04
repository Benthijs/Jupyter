#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import abs, angle
import matplotlib.pyplot as plt

def rad2cart(data, r, theta, x, y):
    """
    Bi-linear interpolation from radial grid (r,theta) to cartesian grid (x,y).

    Input: 
        data     - 2d array of size (nr,ntheta) containing values on the radial grid
        r, theta - 1d arrays of size nr and nt containing the polar gridpoints
        x,y      - 1d arrays of size nx and ny containing the cartesian gridpoints

    Output:
        result   - 2d array of size (nx,nt) containing the interpolated values
    """
    new_data = np.zeros((len(x), len(y)))
    dr = r[1] - r[0]
    dth = theta[1] - theta[0]
    norm = dr * dth
    for i in range(len(x)):
        for j in range(len(y)):
            z  = x[i] + 1j* y[j]
            length, ang = abs(z), angle(z)
            i_r = int(length // dr)
            th_r = int(ang // dth)
            print(i_r, th_r, z)
            if 0 <= i_r < len(r) - 1 and 0 <= th_r < len(theta) - 1:
                # surrounding grid point values
                value = 0
                # 4 #ed gridpoints starting (i_r, th_r) counterclockwise
                for k in range(4):
                    iup = int(k == 1 or k == 2)
                    thup = int(k == 2 or k == 3)
                    # add these numbers to get opposite of (i_r+iup, th_r+thup)
                    i_op = int(iup+1)%2
                    th_op = int(thup+1)%2
                    # calculate the weight associated with point i
                    p_i = abs((length - r[i_r+i_op])*(ang - theta[th_r+th_op]))
                    w_i = p_i / norm
                    print(w_i)
                    value += w_i * data[i_r+iup][th_r+thup]
                    new_data[i][j] = value
            else:
                new_data[i][j] = 0
    return new_data

n = 10
dt = 2*np.pi / n
r = [0.1*i for i in range(n)]
theta = [dt * i for i in range(n)]
x = [i*0.1 for i in range(n)]
y = x.copy()
data = [[x_i** 2 + y_i ** 2 for x_i in x] for y_i in y]

# the interpolated discrete cartesian values of function
int_dat = rad2cart(data, r, theta, x, y)

# graphical representation of interpolation
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('r')
ax.set_ylabel(r'$\theta$')
ax.set_zlabel('Z Label')
for i in range(len(x)):
    for j in range(len(y)):
        ax.scatter(r[i], theta[j], data[i][j], marker='*')
for i in range(len(x)):
    for j in range(len(y)):
        z  = x[i] + 1j* y[j]
        length, ang = abs(z), angle(z)
        ax.scatter(length, ang, int_dat[i][j], marker='o')
#perspective     
ax.view_init(35, 55)
plt.show()
