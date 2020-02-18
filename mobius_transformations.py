import cv2, time
import numpy as np
import math as m
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from numpy import pi as PI

### Apply tranformations
def get_abcd_from_3points(z1, z2, z3, w1, w2, w3):
    a = np.linalg.det(np.array([[z1*w1,w1,1],[z2*w2,w2,1],[z3*w3,w3,1]]))
    b = np.linalg.det(np.array([[z1*w1,z1,w1],[z2*w2,z2,w2],[z3*w3,z3,w3]]))
    c = np.linalg.det(np.array([[z1,w1,1],[z2,w2,1],[z3,w3,1]]))
    d = np.linalg.det(np.array([[z1*w1,z1,1],[z2*w2,z2,1],[z3*w3,z3,1]]))
    return a,b,c,d

def _clockwise_twist(x, y):
    z1, z2, z3 = complex(1, 0.5*y), complex(0.5*x , 0.8*y), complex(0.6*x, 0.5*y)
    w1 = complex(0.5*x, y-1)
    w2 = complex(0.5*x+0.3*m.sin(0.4*PI)*y, 0.5*y+0.3*m.cos(0.4*PI)*y)
    w3 = complex(0.5*x+0.1*m.cos(0.1*PI)*y, 0.5*y-0.1*m.sin(0.1*PI)*x)
    return get_abcd_from_3points(z1, z2, z3, w1, w2, w3)

def _clockwise_half_twist(x, y):
    z1, z2, z3 = complex(1, 0.5*y), complex(0.5*x , 0.8*y), complex(0.6*x, 0.5*y)
    w1 = complex(0.5*x, y-1)
    w2 = complex(0.5*x+0.4*y, 0.5*y)
    w3 = complex(0.5*x, 0.5*y-0.1*x)
    return get_abcd_from_3points(z1, z2, z3, w1, w2, w3)

def _spread(x, y):
    z1, z2, z3 = complex(0.3*x, 0.5*y), complex(0.5*x, 0.7*y), complex(0.7*x, 0.5*y)
    w1, w2, w3 = complex(0.2*x, 0.5*y), complex(0.5*x, 0.8*y), complex(0.8*x, 0.5*y)
    return get_abcd_from_3points(z1, z2, z3, w1, w2, w3)

def _spread_twist(x, y):
    z1, z2, z3 = complex(0.3*x, 0.3*y), complex(0.6*x, 0.8*y), complex(0.7*x, 0.3*y)
    w1, w2, w3 = complex(0.2*x, 0.3*y), complex(0.6*x, 0.9*y), complex(0.8*x, 0.2*y)
    return get_abcd_from_3points(z1, z2, z3, w1, w2, w3)

def _counter_clockwise_twist(x, y):
    z1, z2, z3 = complex(1, 0.5*y), complex(0.5*x , 0.8*y), complex(0.6*x, 0.5*y)
    w1 = complex(0.5*x, y-1)
    w2 = complex(0.5*x+0.4*y, 0.5*y)
    w3 = complex(0.5*x, 0.5*y-0.1*x)
    return get_abcd_from_3points(z1, z2, z3, w1, w2, w3)

def _counter_clockwise_half_twist(x, y):
    z1, z2, z3 = complex(1, 0.5*y), complex(0.5*x , 0.8*y), complex(0.6*x, 0.5*y)
    w1 = complex(0.5*x, y-1)
    w2 = complex(0.5*x+0.3*m.sin(0.4*PI)*y, 0.5*y+0.3*m.cos(0.4*PI)*y)
    w3 = complex(0.5*x+0.1*m.cos(0.1*PI)*x, 0.5*y-0.1*m.sin(0.1*PI)*x)
    return get_abcd_from_3points(z1, z2, z3, w1, w2, w3)

def _inverse(x, y):
    z1, z2, z3 = complex(0.3*x, 0.5*y), complex(0.5*x, 0.9*y), complex(x-1, 0.5*y)
    w1, w2, w3 = complex(x-1, 0.5*y), complex(0.5*x, 0.1*y), complex(1, 0.5*y)
    return get_abcd_from_3points(z1, z2, z3, w1, w2, w3)

def _inverse_spread(x, y):
    z1, z2, z3 = complex(0.1*x, 0.5*y), complex(0.5*x, 0.8*y), complex(0.9*x, 0.5*y)
    w1, w2, w3 = complex(x-1, 0.5*y), complex(0.5*x, 0.1*y), complex(1, 0.5*y)
    return get_abcd_from_3points(z1, z2, z3, w1, w2, w3)

def get_in_location_for_out_location(pts_out, a, b, c, d):
    # pts_out shape 2 x (num pixels)
    A = np.array([[a.real], [a.imag]])
    B = np.array([[b.real], [b.imag]])
    C = np.array([[c.real, -c.imag],[c.imag, c.real]])
    D = np.array([[d.real, -d.imag],[d.imag, d.real]])
    pts_in = np.empty_like(pts_out)
    W = pts_out # shape 2 x 
    DW = D@W
    CW = C@W
    B_DW = B - DW
    CW_A = CW - A
    CW_A_2 = (CW_A**2).sum(axis=0)
    O1 = (B_DW*CW_A).sum(axis=0)/CW_A_2
    B_DW_T = -1*B_DW[0,:]
    B_DW[0,:] = B_DW[1,:]
    B_DW[1,:] = B_DW_T
    O2 = (B_DW*CW_A).sum(axis=0)/CW_A_2
    pts_in[0,:] = O1
    pts_in[1,:] = O2
    return pts_in

def get_out_location_for_in_location(pts_in, a, b, c, d):
    # pts_in shape 2 x (num pixels)
    A = np.array([[a.real, -a.imag], [a.imag, a.real]])
    B = np.array([[b.real], [b.imag]])
    C = np.array([[c.real, -c.imag], [c.imag, c.real]])
    D = np.array([[d.real], [d.imag]])
    pts_out = np.empty_like(pts_in)
    Z = pts_in # shape 2 x 
    AZ = A@Z
    CZ = C@Z
    AZ_P_B = AZ + B
    CZ_P_D = CZ + D
    CZ_P_D_2 = (CZ_P_D**2).sum(axis=0)
    O1 = (AZ_P_B*CZ_P_D).sum(axis=0)/CZ_P_D_2
    AZ_P_B_T = -1*AZ_P_B[0,:]
    AZ_P_B[0,:] = AZ_P_B[1,:]
    AZ_P_B[1,:] = AZ_P_B_T
    O2 = (AZ_P_B*CZ_P_D).sum(axis=0)/CZ_P_D_2
    pts_out[0,:] = O1
    pts_out[1,:] = O2
    return pts_out

# 双线性插值矩阵运算，减少计算量
def bilinear(pts_in, pts_out, s_im, o_min, o_max):
    in_size = s_im.shape[:-1]
    i_max = pts_in.max(axis=1)
    i_min = pts_in.min(axis=1)
    X, Y = (pts_in[0, :]-i_min[0]).reshape((pts_in.shape[1],1)), (pts_in[1, :]-i_min[1]).reshape((pts_in.shape[1],1))
    p_h = max(2, -i_max[0]+i_min[0]+in_size[0])
    p_w = max(2, -i_max[1]+i_min[1]+in_size[1])
    p = max(p_h, p_w)
    n_im = np.zeros((int(i_max[0]-i_min[0]+p), int(i_max[1]-i_min[1]+p), 3))
    n_im[int(-i_min[0]):int(-i_min[0] + in_size[0]), int(-i_min[1]):int(-i_min[1] + in_size[1])] = s_im
    t_left_X, t_left_Y = np.floor(X).astype(np.int16), np.floor(Y).astype(np.int16)
    P1 = (t_left_X+1-X)*n_im[t_left_X, t_left_Y, :].reshape(pts_in.shape[1],3) + (X-t_left_X)*n_im[t_left_X+1, t_left_Y, :].reshape(pts_in.shape[1],3)
    P2 = (t_left_X+1-X)*n_im[t_left_X, t_left_Y+1, :].reshape(pts_in.shape[1],3) + (X - t_left_X)*n_im[t_left_X+1, t_left_Y+1, :].reshape(pts_in.shape[1],3)
    V = (t_left_Y+1-Y)*P1 + (Y-t_left_Y)*P2
    o_im = V.reshape((o_max[0]-o_min[0], o_max[1]-o_min[1], 3))
    return o_im

# Receives an np.matrix as the respresentation of the mobius transformation
def apply_SL2C_elt_to_image(src_image, transformation_type):
    s_im = np.atleast_3d(src_image)
    in_size = s_im.shape[:-1]
    a, b, c, d = transformation_type(in_size[0], in_size[1])

    pts_in = np.indices(in_size).reshape((2,-1)) #results in a 2 x (num pixels) array of indices
    pts_out = get_out_location_for_in_location(pts_in, a, b, c, d)
    o_max = pts_out.max(axis=1)
    o_min = pts_out.min(axis=1)

    pts_out = np.indices(((o_max-o_min)[0], (o_max-o_min)[1])).reshape((2,-1)) #results in a 2 x (num pixels) array of indices
    pts_out = pts_out + np.array([[o_min[0]], [o_min[1]]])

    pts_in = get_in_location_for_out_location(pts_out, a, b, c, d)

    o_im = bilinear(pts_in, pts_out, s_im, o_min, o_max)
    return o_im

def main():
    FUNCLIST = [_clockwise_twist, _clockwise_half_twist, _spread, _spread_twist, _counter_clockwise_twist, _counter_clockwise_half_twist, _inverse, _inverse_spread]
    source_image = np.array(cv2.imread('ori.png'),dtype=np.float32)
    for ii, transformation_type in enumerate(FUNCLIST):
        out_image = apply_SL2C_elt_to_image(source_image, transformation_type)
        np.clip(out_image,0,255,out_image)
        cv2.imwrite(str(ii)+'.png', out_image)

if __name__ == '__main__':
    main()