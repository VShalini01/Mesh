import numpy as np
import scipy.optimize
import torch as th
from misc import grad, smoothen, tensor_img
from hyperparam import VERBOSE
from Draw.draw import sanitize
import matplotlib.pyplot as plt
from misc import tensor2numpy
from scipy.optimize import least_squares, minimize
from tools import timer
import math


def make_grads(img):
    from misc import grad, smoothen
    if img.ndim==4:
        mask = th.Tensor((img[:, :, 3] > 100).astype(np.int))
        img = img[:, :, :3]
    else:
        mask=th.ones(img.shape[:2],dtype=int)

    img = smoothen(tensor_img(img))

    Gr, Gg, Gb = grad(img[0], mask, normalize=True), grad(img[1], mask, normalize=True), grad(img[2], mask, normalize=True)
    return th.stack(Gr,2).numpy(), th.stack(Gg,2).numpy(), th.stack(Gb,2).numpy()


def fdiff(fun, x0, eps):
    y0=fun(x0)
    ret=np.zeros(x0.shape+y0.shape)
    post_slice=tuple([slice(None)]*y0.ndim)
    for idx in np.ndindex(x0.shape):
        x1=np.copy(x0)
        x1[idx]+=eps
        y1=fun(x1)
        dif=(y1-y0)/eps
        ret[idx+post_slice]=dif
    return ret

def cdiff(fun, x0, eps):
    y0=fun(x0)
    ret=np.zeros(x0.shape+y0.shape)
    post_slice=tuple([slice(None)]*y0.ndim)
    for idx in np.ndindex(x0.shape):
        xl=np.copy(x0)
        xl[idx]-=eps
        yl=fun(xl)
        xh=np.copy(x0)
        xh[idx]+=eps
        yh=fun(xh)
        dif=(yh-yl)/(2*eps)
        ret[idx+post_slice]=dif
    return ret

def compare_grads(num_grad, anal_grad, tol):
    assert np.isfinite(num_grad).all()
    assert np.isfinite(anal_grad).all()
    dif=num_grad-anal_grad

    norm=np.maximum(np.maximum(np.abs(num_grad),np.abs(anal_grad)),1.)

    rel_dif=np.abs(dif)/norm

    return np.all(rel_dif<tol)

@timer
def concentric_center(img, Gr, Gg, Gb):
    o_init = np.array([10., 15.])
    ortho_mat = np.array([[0, -1], [1, 0]])

    p = np.stack(np.ndindex(img.shape[1:]))
    # flip xy and inverse y because coordinate systems
    p = np.fliplr(p)
    p[:, 1] = img.shape[2] - p[:, 1] - 1
    p = np.tile(p, [3, 1])

    linear_grads = np.concatenate([Gr.reshape(-1, 2), Gg.reshape(-1, 2), Gb.reshape(-1, 2)])
    linear_grads = linear_grads @ ortho_mat

    def loss_fun(x):
        o = x
        dif = o[None, :] - p
        prod = (dif * linear_grads).sum(axis=1)
        return prod

    def jac(x):
        return linear_grads

    res = least_squares(loss_fun, o_init, jac=jac)

    # normalise residuals by radius
    residuals = res.fun.reshape(3, *Gr.shape[:2])
    radii = np.linalg.norm(np.stack(np.ndindex(img.shape[1:])) - res.x, axis=1).reshape(img.shape[1:])

    residuals = residuals / radii[None, :, :]

    return res.x, res.cost, residuals


def solve(Gr, Gg, Gb, P):
    """
    Treating this as a normal equation X * theta = y
    from G.ortho * (p-o)=0
    G.ortho*p=G.ortho*o
    so
    theta = o
    X=G.ortho
    y=G.ortho*p (row-wise dot)

    solution is: theta = (X.T*X).inv() * (X.T*y)
    """
    P=np.tile(P,[3,1])
    G=np.concatenate([serialize_img(Gr), serialize_img(Gg), serialize_img(Gb)],axis=0)

    G_ortho=np.fliplr(G)*np.array([-1,1])[None,:]

    X=G_ortho
    y=np.matmul(G_ortho[:,None,:],P[:,:,None]).squeeze(2).squeeze(1)

    m=X.T@X

    # check that m is invertible
    if np.abs(np.linalg.det(m))<1e-6:
        print("Error: Cannot solve for radial gradient")
        return None

    theta=np.linalg.inv(m)@(X.T@y)

    return theta
    '''
    E = Sum_{all points and color} (Gx*(Py-y) - Gy*(Px-x))^2
    dE/dx = 0 = 2 Sum_{all points and color}  Gx*Gy*Py - Gx*Gy*y - Gy*Gy*Px + Gy*Gy*x
    dE/dy = 0 = 2 Sum_{all points and color} -Gx*Gx*Py + Gx*Gx*y + Gx*Gy*Py - Gx*Gy*x
    '''
    A = Gr[:, :, 1] * Gr[:, :, 0] + Gg[:, :, 1] * Gg[:, :, 0] + Gb[:, :, 1] * Gb[:, :, 0]
    B = Gr[:, :, 0] * Gr[:, :, 0] + Gg[:, :, 0] * Gg[:, :, 0] + Gb[:, :, 0] * Gb[:, :, 0]
    C = Gr[:, :, 1] * Gr[:, :, 1] + Gg[:, :, 1] * Gg[:, :, 1] + Gb[:, :, 1] * Gb[:, :, 1]
    a, b, c = np.sum(A), np.sum(B), np.sum(C)
    ay, ax = np.sum(A * Py), np.sum(A * Px)
    by = np.sum(B * Py)
    cx = np.sum(C * Px)

    T = np.array([
        [-a, c],
        [c, -a]
    ])

    R = np.array([
        [cx - ay],
        [by - ax]
    ])

    det = np.linalg.det(T)
    if det == 0:
        print("Error: Cannot solve for radial gradient")
        return None
    X = np.linalg.inv(T) @ R
    x = X[1, 0]
    y = X[0, 0]
    return x, y


@timer
def pointNearCenter(grads):
    Gr, Gg, Gb=grads
    cols, rows = Gr.shape[0], Gr.shape[1]
    size=np.array([cols, rows])
    p=make_serialized_points(size)

    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(Gr[:,:,0])
    #plt.title("grad x")
    #plt.subplot(1,2,2)
    #plt.imshow(deserialize_img(p,size)[:,:,0])
    #plt.title("px")
#
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(Gr[:,:,1])
    #plt.title("grad y")
    #plt.subplot(1,2,2)
    #plt.imshow(deserialize_img(p,size)[:,:,1])
    #plt.title("py")
#
    #plt.show()

    x, y = solve(Gr, Gg, Gb, p)
    arc_point = np.array([x, y])
    return arc_point

def clip_vec(v, clip_mag):
    mag=np.linalg.norm(v)
    if mag>clip_mag:
        v=(v/mag)*clip_mag
    return v

def soft_unit_clip(v):
    mag=np.linalg.norm(v)
    if mag<1e-12:
        return v
    return v*(np.tanh(mag))/(mag)

@timer
def non_concentric_center_eccentricity(grads, o):
    Gr, Gg, Gb=grads
    cols, rows = Gr.shape[0], Gr.shape[1]
    size=np.array([cols, rows])
    o_init = o
    e_init = np.array([0., 0.])
    ortho_mat = np.array([[0, -1], [1, 0]])

    #Px,Py=np.meshgrid(np.arange(cols),np.arange(rows))
    #Py=np.flipud(Py)
    #p=np.stack([Px.reshape(-1), Py.reshape(-1)],axis=1)
    p=make_serialized_points(size)

    linear_grads = np.concatenate([Gr.reshape(-1, 2), Gg.reshape(-1, 2), Gb.reshape(-1, 2)])
    linear_grads = linear_grads @ ortho_mat

    x0 = np.concatenate([o_init, e_init])

    def calc_rs(o, e):
        e=clip_vec(e,0.999999)
        op = (p-o[None, :])
        a = (1-(e.T @ e))
        b = 2 * (e[None, :] * op).sum(axis=1)
        c = -(op * op).sum(axis=1)
        D=-(4 * a * c) + (b ** 2)

        ret1 = (-b - np.sqrt(D)) / (2 * a)
        ret2 = (-b + np.sqrt(D)) / (2 * a)
        assert np.all(np.isfinite(ret1)) and np.all(np.isfinite(ret2))
        assert np.all(ret1>=0.) or np.all(ret2>=0.)
        return np.maximum(ret1, ret2)

    def unpack_x(x):
        return x[:2], x[2:4]

    def fun(x):
        o, e = unpack_x(x)
        rs = calc_rs(o, e)
        f = (linear_grads * np.tile((p - o[None, :] - rs[:, None] * e[None, :]), [3, 1])).sum(axis=1)
        return f

    # res=scipy.optimize.minimize(fun,x0,method='L-BFGS-B')
    res = least_squares(fun, x0,
                        #bounds=(
                        #    [-np.inf, -np.inf, -np.inf, -0.99999999],
                        #    [np.inf, np.inf, np.inf, 0.99999999]
                        #)
                        )

    o, e = unpack_x(res.x)
    rs = calc_rs(o, e)

    return o, clip_vec(e,0.999999), rs.reshape([rows, cols]), res.cost, res.fun.reshape(3, *[rows, cols])


def center_eccentricity(Gs):
    Gr, Gg, Gb = Gs
    rows, cols = Gr.shape[0], Gr.shape[1]

    #Gr = np.stack([tensor2numpy(Gr[0]), tensor2numpy(Gr[1])], axis=2)
    #Gg = np.stack([tensor2numpy(Gg[0]), tensor2numpy(Gg[1])], axis=2)
    #Gb = np.stack([tensor2numpy(Gb[0]), tensor2numpy(Gb[1])], axis=2)

    # find origin assuming concentric
    o = pointNearCenter(Gs)
    # now try to find e
    o, e, r, oe_cost, oe_residuals = non_concentric_center_eccentricity(Gs, o)

    #return np.array([o[0], cols - o[1]]), np.array([e[0], -e[1]])
    return np.array(o), np.array([e[0], e[1]])

def guess_o(Gs, p):
    gmags=np.stack([
        np.linalg.norm(G,axis=2) for G in Gs
    ])

    wps=p*(gmags.sum(axis=0).flatten()[:,None])

    ret=wps.sum(axis=0)/gmags.sum()

    return ret

def radius_abcs(p, o, e, T):
    q=T.inv_transform(p)
    oq=o-q
    enorm=e.T@e

    c=(np.power(oq,2)).sum(axis=1)
    b=(2*oq@e[:,None]).squeeze(1)
    a=enorm-np.ones_like(c)

    return np.stack([a,b,c])

def radius_dabcs(p, o, e, alpha, theta):
    q=np.matmul(make_T_inv(alpha, theta),p[:,:,None]).squeeze(2)
    oq=o-q

    two_e=2*np.tile(e[None,:],[p.shape[0],1])

    dabcdo = np.stack([
        np.zeros_like(p),
        two_e,
        2*oq
    ],axis=0)

    dabcde = np.stack([
        two_e,
        2*oq,
        np.zeros_like(p)
    ])

    TiT=make_T_inv(alpha, theta).T
    TiTe=2*(TiT@e)

    dabcdT = np.stack([
        np.zeros((p.shape[0],2,2)),
        np.matmul(TiTe[:,None],q[:,None,:]),
        2*np.matmul(np.matmul(TiT, oq[:,:,None]),q[:,None,:])
    ],axis=0)

    return dabcdo.transpose(2,0,1), dabcde.transpose(2,0,1), dabcdT

def quad_solve(a, b, c):
    d=np.sqrt(np.power(b,2)-4*a*c)
    ret=np.stack([(-b-d)/(2*a),(-b+d)/(2*a)],axis=1)

    return ret

def quad_grad(a,b,c):
    d=np.sqrt(np.power(b,2)-4*a*c)
    ret=np.array([
        [(-2*a*c+d*(b-d))/(2*a**2 *d), (2*a*c+d*(b+d))/(2*a**2 *d)],
        [(b-d)/(2*a*d),-(b+d)/(2*a*d)],
        [-1/d,1/d]
    ])
    ret=np.flip(ret,axis=1)
    return ret

def calc_r(p, o, e, T):
    abc=radius_abcs(p,o,e,T).T
    r=quad_solve(abc[:,0],abc[:,1],abc[:,2])

    assert np.all(r[:,0]>=0.)

    return r[:,0]

def calc_dr(p,o,e,alpha,theta):
    abc=radius_abcs(p, o, e, make_T_inv(alpha, theta)).T
    dr=quad_grad(abc[:,0],abc[:,1],abc[:,2])
    dabc=radius_dabcs(p, o, e, alpha, theta)

    dabcdo=np.einsum("arp, oap -> pro",dr,dabc[0])
    dabcde=np.einsum("arp, eap -> pre",dr,dabc[1])
    dabcdT=np.einsum("arp, apij -> prij",dr,dabc[2])

    return dabcdo[:,0,:], dabcde[:,0,:], dabcdT[:,0,:,:]


#def to_euclidean(x):
#    # convert polar vector to euclidean
#    assert x.shape==(2,)
#    phi, r=x
#    return np.array([np.cos(phi), np.sin(phi)])*r

def polar_grad(x):
    # jacobian of the polar to euclidean conversion
    assert x.shape==(2,)
    phi, r=x
    sinphi=np.sin(phi)
    cosphi=np.cos(phi)
    return np.array([
        [-r*sinphi, cosphi],
        [r*cosphi, sinphi]
    ])


def X_r(g, p, o, e, T, r):
    ortho_mat = np.array([[0, -1], [1, 0]])

    ctr_dirs=p-T.fwd_transform(o[None,:]+r[:,None]*e[None,:])
    ctr_dirs=T.fwd_transform(T.inv_transform(ctr_dirs)@ortho_mat)

    p3=np.tile(ctr_dirs,[3,1])

    rhos=np.matmul(g[:,None,:],p3[:,:,None]).squeeze(2).squeeze(1)

    return rhos

def pX_r(g, _, o, e, T, r):
    pXpo=np.matmul(-T.T,g[:,:,None]).squeeze(2)
    pXpe=np.tile(r,[3])[:,None]*np.matmul(-T.T,g[:,:,None]).squeeze(2)
    pXpT=np.matmul(-g[:,:,None],np.tile((o[None,:]+r[:,None]*e[None,:])[:,None,:],[3,1,1]))

    return pXpo, pXpe, pXpT

def pX_pr(g, e, alpha, theta):
    T=make_T(alpha, theta)
    ret=np.matmul(np.matmul(-g[:,None,:],T),e[:,None])

    return ret.squeeze(2).squeeze(1)

def X(g, p, o, ep, T):
    # e is polar, let's convert to euclidean:
    ee=ep#to_euclidean(ep)
    r=calc_r(p, o, ee, T)
    return X_r(g, p, o, ee, T, r)

def dX(g, p, o, ep, T):
    # e is polar, let's convert to euclidean:
    ee=to_euclidean(ep)
    r=calc_r(p, o, ee, T)
    pXpo, pXpe, pXpT=pX_r(g, p, o, ee, T, r)
    dT=T.grad
    pXps=np.tensordot(pXpT,dT[0],2)
    pXptheta=np.tensordot(pXpT,dT[1],2)

    pXpr=pX_pr(g, ee, T)

    drdo, drde, drdT=calc_dr(p, o, ee, T)
    drds=np.tensordot(drdT,dT[0],2)
    drdtheta=np.tensordot(drdT,dT[1],2)

    do=pXpo +         pXpr[:,None] *         np.tile(drdo,[3,1])
    de=pXpe +         pXpr[:,None] *         np.tile(drde,[3,1])
    ds=pXps +         pXpr *         np.tile(drds,[3])
    dtheta=pXptheta + pXpr * np.tile(drdtheta,[3])

    #to account for e being originally polar, we have to multiply by the polar conversion jacobian
    e_polar_grad=polar_grad(ep)
    return do, de@ e_polar_grad, ds[:,None], dtheta[:,None]

def jac(p, linear_grads, o, e, scale, theta):
    grads=dX(linear_grads, p, o, e, scale, theta)

    return np.concatenate(grads,axis=1)

def fun(p, linear_grads, o, e, T):
    return X(linear_grads, p, o, e, T)

def make_serialized_points(sz):
    Px,Py=np.meshgrid(np.arange(sz[0]),np.arange(sz[1]),indexing="ij")
    #Py=np.flipud(Py)
    p_ser=serialize_img(np.stack([Px, Py],axis=2))
    return p_ser

def serialize_img(img):
    assert img.ndim<4 and img.ndim>=2
    if img.ndim==2:
        img=img[:,:,None]
    ret=img.reshape(-1,img.shape[2])
    return ret.squeeze(1) if ret.shape[1]==1 else ret

def deserialize_img(ser_img, sz):
    ret=ser_img.reshape(*sz,*ser_img.shape[1:])

    return ret

class GradientTransform:
    def __init__(self, alpha, theta):
        self.alpha=alpha
        self.theta=theta
        self._T=None
        self._T_inv=None
        self._grad=None

    @property
    def T(self):
        if self._T is None:
            self._T=self.make_T()
        return self._T

    @property
    def T_inv(self):
        if self._T_inv is None:
            self._T_inv=self.make_T_inv()
        return self._T_inv

    @property
    def grad(self):
        if self._grad is None:
            self._grad=self.T_grad()
        return self._grad

    def fwd_transform(self, vector):
        is_1d=(vector.ndim==1)
        if is_1d:
            vector=vector[None,:]
        else:
            assert vector.ndim==2
        ret=np.matmul(vector[:,None,:],self.T).squeeze(1)
        return ret if not is_1d else ret.squeeze(0)

    def inv_transform(self, vector):
        is_1d=(vector.ndim==1)
        if is_1d:
            vector=vector[None,:]
        else:
            assert vector.ndim==2
        ret=np.matmul(vector[:,None,:],self.T_inv).squeeze(1)
        return ret if not is_1d else ret.squeeze(0)

    def make_T(self):
        alpha=self.alpha
        theta=self.theta

        S=np.array([
            [np.exp(alpha), 0.],
            [0., np.exp(-alpha)]
        ])

        R=np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

        return (R@S).T

    def make_T_inv(self):
        alpha=self.alpha
        theta=self.theta

        Sinv=np.array([
            [np.exp(-alpha), 0.],
            [0., np.exp(alpha)]
        ])

        Rinv=np.array([
            [ np.cos(-theta), np.sin(-theta)],
            [-np.sin(-theta), np.cos(-theta)]
        ])

        return (Sinv@Rinv).T

    def T_grad(self):
        alpha=self.alpha
        theta=self.theta

        sint=np.sin(theta)
        cost=np.cos(theta)

        et=np.exp(alpha)
        ent=np.exp(-alpha)

        ds=np.array([
            [et*cost,-et*sint],
            [-ent*sint, -ent*cost]
        ])

        dth=np.array([
            [-et*sint, -et*cost],
            [ ent*cost, -ent*sint]
        ])

        return ds, dth

    def __str__(self):
        return str(self.T)


def elliptic_non_concentric(Gs, use_e=True, use_T=True, init=None):
    print ("G", len(Gs))
    np_Gs=Gs
    Gr, Gg, Gb = np_Gs[0], np_Gs[1], np_Gs[2]
    cols, rows = Gr.shape[0], Gr.shape[1]
    size=np.array([cols, rows])
    ortho_mat = np.array([[0, -1], [1, 0]])

    #Px,Py=np.meshgrid(np.arange(cols),np.arange(rows))
    #Py=np.flipud(Py)
    #p=np.stack([Px.reshape(-1), Py.reshape(-1)],axis=1)
    #p_ser=serialize_img(np.stack([Px, Py],axis=2))
    #assert np.all(p==p_ser)
    p=make_serialized_points(size)

    #p = np.stack(list(np.ndindex((rows, cols))))
    #p = np.fliplr(p)
    #p[:, 1] = rows - p[:, 1] - 1

    e_init = np.array([0.,0.]) #np.random.randn(2)/100
    o_init = np.array([cols/2,rows/2]) #guess_o(np_Gs, p)/size
    s_init=0.
    theta_init=0.

    if init is not None:
        if init[0] is not None:
            o_init=init[0]
        if init[1] is not None:
            e_init=init[1]
        if init[2] is not None:
            s_init, theta_init=init[2]

    print("init transform:",GradientTransform(s_init, theta_init))

    mode={
        "use_e":use_e,
        "use_T":use_T
    }

    def pack_vars(o, e, s, theta, mode):
        return np.concatenate(
            [o/size] + ([e] if mode["use_e"] else [])+ ([np.array([s,theta])] if mode["use_T"] else []))

    def unpack_vars(vars, use_e, use_T):
        o=vars[0:2]*size
        if use_e:
            e=clip_vec(vars[2:4],0.99999)
        else:
            e=np.array([0.,0.])

        if use_T:
            T_idx=4 if use_e else 2
            s=vars[T_idx]
            theta=vars[T_idx+1]
        else:
            s=0.
            theta=0.

        return o, e, GradientTransform(s, theta)

    linear_grads = np.concatenate([Gr.reshape(-1, 2), Gg.reshape(-1, 2), Gb.reshape(-1, 2)])
    #linear_grads = linear_grads @ ortho_mat
    #linear_grads = np.fliplr(linear_grads)

    x0 = pack_vars(o_init, e_init, s_init, theta_init, mode)

    def fun_wrapper(x,**kwargs):
        # print(".",end="")
        #print("fun",x)
        use_e=True if "use_e" not in kwargs else kwargs["use_e"]
        use_T=True if "use_T" not in kwargs else kwargs["use_T"]
        vars=unpack_vars(x,use_e,use_T)
        ret=fun(p,linear_grads,*vars)

        return ret

    def jac_wrapper(x,**kwargs):
        #print("jac",x)
        use_e=True if "use_e" not in kwargs else kwargs["use_e"]
        use_T=True if "use_T" not in kwargs else kwargs["use_T"]
        vars=unpack_vars(x,use_e,use_T)
        ret=jac(p,linear_grads,*vars)

        j_o=ret[:,:2]*size[None,:]
        j_e=ret[:,2:4]
        j_T=ret[:,4:]

        ret=[j_o]+([j_e] if use_e else [])+([j_T] if use_T else [])

        ret=np.concatenate(ret,axis=1)
        #print("grd",ret.sum(axis=0))
        return ret

    low_bounds=np.concatenate(
        [np.array([-np.inf, -np.inf])]+
        ([np.array([-np.inf,-np.inf])] if mode["use_e"] else [])+
        ([np.array([(math.pi/2.)-np.arctan(10**2), -1e38])] if mode["use_T"] else [])
    )
    high_bounds=np.concatenate(
        [np.array([np.inf, np.inf])]+
        ([np.array([np.inf, np.inf])] if mode["use_e"] else [])+
        ([np.array([np.arctan(10**2), 1e38])] if mode["use_T"] else [])
    )

    res = least_squares(
        fun_wrapper,
        x0,
        #jac=jac_wrapper,
        #bounds=(
        #    low_bounds,
        #    high_bounds
        #),
        kwargs=mode
    )

    #def gfwrap(x):
    #    return fun_wrapper(x, **mode)

    #num_grad=fdiff(gfwrap, res.x, 1e-9).T
    #anal_grad=jac_wrapper(res.x, **mode)
    #print("compare grads", compare_grads(num_grad, anal_grad, 1e-4))

    cost=res.cost

    print("Gradient:",res.grad)
    print("Cost:",cost)

    print(res.message)

    o, e, T = unpack_vars(res.x,mode["use_e"],mode["use_T"])
    rs = calc_r(p, o, e, T)

    return o, e, T, T.alpha, T.theta, deserialize_img(rs,size), cost, fun_wrapper(res.x, **mode).reshape(3, *[rows, cols])


def grad(color_img, normalized = True):
    """
    :param color_img: 1 channel Image is numpy array. Y positive is downwards
    :param normalized: Normalize the gradient.
    :return: gradient along x and y axis. y positive is down.
    """
    assert len(color_img.shape) == 2
    gy, gx = np.gradient(color_img) #sobel_op(color_img) #
    g = np.stack([gx.flatten(), gy.flatten()], axis=1)
    mag = np.linalg.norm(g, axis=1) if normalized else np.ones(len(g))
    mag[mag < 1e-8] = 1 # convert small values to 1
    return g / mag[:, None]


def get_parameters(img):
    from misc import grad
    if img is None:
        exit(0)
    assert len(img.shape) == 3

    def crop_img(img, c=10):
        return img[c:-c, c:-c, :]

    img = crop_img(img, c=5)

    Gs = make_grads(img)
    o, e, T, alpha, theta, img_, _, _ = elliptic_non_concentric(Gs)
    print("Rotation:{}".format(np.rad2deg(theta)))
    return o, e, alpha, theta, img



def print_trained_params(f, o, e, s, t, rad):
    s = -1 if s is None else s
    t = 0 if t is None else t
    rad = -1 if rad is None else rad
    print("-------------------------------------"
          "\n\tCenter \t\t :[{:.3f}, {:.3f}] "
          "\n\tFocal  \t\t :[{:.3f}, {:.3f}] "
          "\n\tEccentricity :[{:.3f}, {:.3f}] "
          "\n\tRadius :[{}] "
          "\n\tScale    :{:.3f} "
          "\n\tRotation :{:.3f} deg\n"
          "-------------------------------------".
          format(f[0], f[1], o[0], o[1], e[0], e[1], rad, s, np.rad2deg(t) % 360))


def draw_reconstruction(img, f, o, scale, rotate):
    from misc import get_circle
    plt.subplots()
    plt.imshow(img)
    plt.plot(f[0], f[1], 'o')
    if o is not None:
        plt.plot(o[0], o[1], '+')
        if rotate is not None:
            l = np.max(img.shape)*0.5
            axis1 = np.stack([o, [o[0]+l*np.cos(rotate), o[1]+l*np.sin(rotate)]])
            axis2 = np.stack([o, [o[0] + l * np.cos(rotate+np.pi/2), o[1] + l * np.sin(rotate+np.pi/2)]])
            plt.plot(axis1[:,0], axis1[:,1], '-', color='aqua')
            plt.plot(axis2[:,0], axis2[:,1], '--', color='aqua')
    plt.show()


def main(file):
    from misc import read_img
    img = read_img(file + '.png')
    if img is None:
        print ("ERROR: Image not found.!!")
        exit(1)
    plt.imshow(img)
    plt.show()

    fHat, eHat, scale, rotate, image =get_parameters(img)

    print_trained_params(fHat, fHat, eHat, scale, rotate, -1)
    draw_reconstruction(image, fHat, fHat + eHat, None, rotate)


if __name__ == '__main__':
    main('Con200')

