import torch as th
import numpy as np
import scipy.optimize
from scipy.optimize import least_squares
from misc import grad, smoothen, tensor_img, sanitize
from hyperparam import VERBOSE
from Draw.draw import vector_field
import matplotlib.pyplot as plt


def guess_o(Gs, p):
    gmags=np.stack([
        np.linalg.norm(G,axis=2) for G in Gs
    ])

    wps=p*(gmags.sum(axis=0).flatten()[:,None])

    ret=wps.sum(axis=0)/gmags.sum()

    return ret

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

def calc_rs(p, o, e, T):
    q=p@np.linalg.inv(T).T
    op = (o[None, :] - q)
    a = (e.T @ e - 1)
    b = 2 * (e[None, :] * op).sum(axis=1)
    c = (op * op).sum(axis=1)

    ret = (-b - np.sqrt(np.maximum(-(4 * a * c) + (b ** 2),1e-12))) / (2 * a)

    return ret

def calc_drs(p, o, e, T):
    q=p@np.linalg.inv(T).T
    oq = (o[None, :] - q)
    a = (e.T @ e - 1)
    b = 2 * (e[None, :] * oq).sum(axis=1)
    c = (oq * oq).sum(axis=1)

    dsq=np.maximum(1e-18,-(4 * a * c) + (b ** 2))

    d=np.maximum(np.sqrt(dsq),1e-18)

    drdabc=np.stack([
        (2*a*c+d*(b+d))/(2* a*a * d),
        -(b+d)/(2*a*d),
        1/d
    ],axis=1)

    dabcdo=np.stack([
        np.tile([[0.]],p.shape),
        np.tile(2*e[None,:],[p.shape[0],1]),
        2*oq
    ],axis=1)

    dabcde=np.stack([
        np.tile(2*e[None,:],[p.shape[0],1]),
        2*oq,
        np.tile([[0.]],p.shape)
    ],axis=1)

    ti2=np.linalg.inv(T).T*2
    dabcdT=np.stack([
        np.tile([[0.]],[*p.shape,2]),
        np.matmul((ti2@e[:,None])[None,:,:],q[:,None,:]),
        np.matmul(np.matmul(ti2[None,:,:],oq[:,:,None]),q[:,None,:])
    ],axis=1)
    assert np.all(np.isfinite(drdabc))
    assert np.all(np.isfinite(dabcdo))
    assert np.all(np.isfinite(dabcde))
    assert np.all(np.isfinite(dabcdT))

    return np.matmul(drdabc[:,None,:],dabcdo).squeeze(1), \
           np.matmul(drdabc[:,None,:],dabcde).squeeze(1), \
           np.einsum("ab,abcd->acd",drdabc,dabcdT)


def make_T(s, theta):
    S=np.array([
        [s, 0.],
        [0., 1.]
    ])

    R=np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])

    return S@R

def T_grad(s, theta):

    sint=np.sin(theta)
    cost=np.cos(theta)

    ds=np.array([
        [cost, sint],
        [0.,0.]
    ])

    dth=np.array([
        [-s*sint,-cost],
        [s*cost, -sint]
    ]).T

    return ds, dth

def jac(p, linear_grads, o, e, T, s, theta):
    rs = np.tile(calc_rs(p, o, e, T),[3])
    ttlg=np.matmul(T.T,linear_grads[:,:,None]).squeeze(2)
    pXpo=-ttlg
    pXpe=-rs[:,None]*ttlg
    pXpT=-np.matmul(linear_grads[:,:,None],(o[None,:]+rs[:,None]*e[None,:])[:,None,:])

    pXpr=-np.matmul(linear_grads[:,None,:],T@e).squeeze(1)

    drdo, drde, drdT=calc_drs(p, o, e, T)

    dXdo=pXpo+pXpr[:,None]*np.tile(drdo,[3,1])
    dXde=pXpe+pXpr[:,None]*np.tile(drde,[3,1])
    dXdT=pXpT+pXpr[:,None,None]*np.tile(drdT,[3,1,1])

    dTds, dTdth=T_grad(s, theta)

    dXds=np.einsum("abc,bc->a",dXdT,dTds)
    dXdth=np.einsum("abc,bc->a",dXdT,dTdth)

    assert np.all(np.isfinite(dXdo))
    assert np.all(np.isfinite(dXde))
    assert np.all(np.isfinite(dXds))
    assert np.all(np.isfinite(dXdth))
    return np.concatenate([dXdo, dXde, dXds[:,None], dXdth[:,None]],axis=1)

def fun(p, linear_grads, o, e, T, rows=None, cols=None, img=None):
    rs = calc_rs(p, o, e, T)

    if VERBOSE and rows is not None and cols is not None:
        print("radius shape :=", rs.shape)
        img_rs = rs.reshape(rows, cols)
        img_rs = img_rs / np.max(img_rs)
        plt.imshow(img_rs, cmap='gray')
        plt.title("Distance Scalar Field")
        plt.show()

    # Rotated Red color gradient field
    grad_red = linear_grads[:rows*cols].reshape(rows, cols, 2)
    vector_field(img, None, grad_red[:,:,0], grad_red[:,:,1], label="Rotated Field Red")

    # vector from p -> c field.
    p2c = np.tile(p - o[None, :], [3, 1])
    if VERBOSE:
        p2c_field = p2c[:rows*cols].reshape(rows, cols, 2)
        vector_field(img, None, p2c_field[:, :, 0], p2c_field[:, :, 1], label="Direction to Center")

    print("p2c Shape:", p2c.shape, ", linear_grad:", linear_grads.shape)
    k = linear_grads[:, None, :] @ p2c[:, :, None]

    if VERBOSE and rows is not None and cols is not None:
        print("K :{}. K average:{}".format(k.shape, np.mean(k)))

        fig, axs = plt.subplots(1,4)
        if img is not None:
            axs[0].imshow(img)
        for i in range(3):
            img_k = k[i*rows*cols : (i+1)*rows*cols].reshape(rows, cols)
            img_k = img_k / np.max(img_k)
            axs[i+1].imshow(img_k, cmap='gray')
        plt.show()

    f = np.matmul(
        linear_grads[:,None,:],
        np.tile(
            p - (T@(o[None, :] + rs[:, None] * e[None, :]).T).T,
            [3,1]
        )[:,:,None]
    ) #    gT dot (p - o)
        #(linear_grads * np.tile((p - (o[None, :] + rs[:, None] * e[None, :])@T.T), [3, 1])).sum(axis=1)
    return f.squeeze(2).squeeze(1)

def check_T_grad(s,theta):
    anal_grad=np.stack(T_grad(s,theta))
    num_grad=fdiff(lambda x:make_T(*x),np.array([s,theta]),1e-6)

    dif=num_grad-anal_grad

    assert np.allclose(num_grad,anal_grad,atol=1e-6)

def elliptic_non_concentric(Gs):
    np_Gs=[np.stack([G[0].numpy(),G[1].numpy()], axis=2) for G in Gs]
    Gr, Gg, Gb = np_Gs[0], np_Gs[1], np_Gs[2]
    rows, cols = Gr.shape[0], Gr.shape[1]
    ortho_mat = np.array([[0, -1], [1, 0]])

    linear_grads = np.concatenate([Gr.reshape(-1, 2), Gg.reshape(-1, 2), Gb.reshape(-1, 2)])
    linear_grads = linear_grads @ ortho_mat

    p = np.stack(list(np.ndindex((rows, cols))))
    p = np.fliplr(p)
    # p[:, 1] = cols - p[:, 1] - 1

    size=np.array([rows, cols])
    # size = np.array([cols, rows])

    e_init = np.array([0.,0.]) #np.random.randn(2)/100
    o_init = np.array([rows/2,cols/2]) #guess_o(np_Gs, p)/size #size*0.5 #
    s_init=1.45
    theta_init=0.

    check_T_grad(s_init, theta_init)

    mode={
        "use_e":False,
        "use_T":True
    }

    def pack_vars(o, e, s, theta, mode):
        return np.concatenate(
            [o/size] + ([e] if mode["use_e"] else [])+ ([np.array([s,theta])] if mode["use_T"] else []))

    x0 = pack_vars(o_init, e_init, s_init, theta_init, mode)

    def unpack_vars(vars, use_e, use_T):
        o=vars[0:2]*size
        if use_e:
            e=vars[2:4]
        else:
            e=np.array([0.,0.])

        if use_T:
            T_idx=4 if use_e else 2
            s=vars[T_idx]
            theta=vars[T_idx+1]
        else:
            s=1.
            theta=0.

        return o, e, make_T(s, theta), s, theta


    # flip xy and inverse y because coordinate systems

    #p[:, 1] = cols - p[:, 1] - 1
    # rs=np.linalg.norm(p-o_init[None,:],axis=1)

    def fun_wrapper(x,**kwargs):
        print("fun",x)
        use_e=True if "use_e" not in kwargs else kwargs["use_e"]
        use_T=True if "use_T" not in kwargs else kwargs["use_T"]
        vars=unpack_vars(x,use_e,use_T)
        ret=fun(p,linear_grads,*vars[:3], rows, cols)

        return ret

    def jac_wrapper(x,**kwargs):
        print("jac",x)
        use_e=True if "use_e" not in kwargs else kwargs["use_e"]
        use_T=True if "use_T" not in kwargs else kwargs["use_T"]
        vars=unpack_vars(x,use_e,use_T)
        ret=jac(p,linear_grads,*vars)

        j_o=ret[:,:2]*size[None,:]
        j_e=ret[:,2:4]
        j_T=ret[:,4:]

        ret=[j_o]+([j_e] if use_e else [])+([j_T] if use_T else [])

        ret=np.concatenate(ret,axis=1)
        print(ret.sum(axis=0))
        return ret

    low_bounds=np.concatenate(
        [np.array([-1e38, -1e38])]+
        ([np.array([-1, -1])] if mode["use_e"] else [])+
        ([np.array([1e-6, -1e38])] if mode["use_T"] else [])
    )
    high_bounds=np.concatenate(
        [np.array([1e38, 1e38])]+
        ([np.array([1, 1])] if mode["use_e"] else [])+
        ([np.array([1e38, 1e38])] if mode["use_T"] else [])
    )

    def gfun(x, mode):
        res=fun_wrapper(x, **mode)
        return res.sum()
    num_grad=scipy.optimize.approx_fprime(x0, gfun, 1e-8, mode)
    anal_grad=jac_wrapper(x0,**mode).sum(axis=0)
    dif=num_grad-anal_grad
    rel_dif=np.abs(dif)/np.maximum(np.abs(num_grad),np.abs(anal_grad))
    #assert np.allclose(num_grad,anal_grad,rtol=1e-3)
    res = least_squares(
        fun_wrapper,
        x0,
        jac=jac_wrapper,
        #bounds=(
        #    low_bounds,
        #    high_bounds
        #),
        xtol=1e-15,
        ftol=1e-15,
        kwargs=mode
    )

    num_grad=scipy.optimize.approx_fprime(res.x, gfun, 1e-9, mode)
    anal_grad=jac_wrapper(res.x,**mode).sum(axis=0)
    res_grad=res.grad
    dif=num_grad-anal_grad
    #assert np.allclose(num_grad,anal_grad,rtol=1e-3)

    appr_res_grad=2*fun_wrapper(res.x,**mode)[None,:]@jac_wrapper(res.x,**mode)
    dif=appr_res_grad-res_grad

    #assert np.allclose(appr_res_grad,res_grad,rtol=1e-3,atol=1e-4)

    #def BFGS_fun_wrapper(val):
    #    x=fun_wrapper(val,**mode)
    #    ret=x.T@x
    #    return ret
#
    #def BFGS_jac_wrapper(val):
    #    x=fun_wrapper(val,**mode)
    #    dx=jac_wrapper(val,**mode)
    #    ret=2*x[None,:]@dx
    #    return ret.squeeze(0)
#
    #num_grad=scipy.optimize.approx_fprime(x0, BFGS_fun_wrapper, 1e-8)
    #anal_grad=BFGS_jac_wrapper(x0)
    #dif=num_grad-anal_grad
    #assert np.allclose(num_grad,anal_grad)
#
    #res=minimize(
    #    BFGS_fun_wrapper,
    #    x0,
    #    jac=BFGS_jac_wrapper,
    #    #bounds=(
    #    #    low_bounds,
    #    #    high_bounds
    #    #),
    #    #method="L-BFGS-B"
    #)
#
    #num_grad=scipy.optimize.approx_fprime(res.x, BFGS_fun_wrapper, 1e-8)
    #anal_grad=BFGS_jac_wrapper(res.x)
    #dif=num_grad-anal_grad
    #assert np.allclose(num_grad,anal_grad)

    cost=res.cost

    print("Gradient:",res.grad)

    print(res.message)

    o, e, T, s, theta = unpack_vars(res.x,mode["use_e"],mode["use_T"])
    rs = calc_rs(p, o, e, T)

    return o, e, T, s, theta, rs.reshape([rows, cols]), cost, fun_wrapper(res.x, **mode).reshape(3, *[rows, cols])


def get_grad(img):
    mask = th.Tensor((img[:, :, 3] > 100).astype(np.int)) if len(img.shape) == 4 else th.ones(
        [img.shape[0], img.shape[1]])
    img = img[:, :, :3]
    img = smoothen(tensor_img(img))

    Gr, Gg, Gb = grad(img[0], mask, normalize=True), grad(img[1], mask, normalize=True), grad(img[2], mask, normalize=True)
    if VERBOSE:
        vector_field(img, mask, Gr[0], Gr[1], "Red Gradient Field")
        # vector_field(img, mask, Gg[0], Gg[1], "Green Gradient Field")
        # vector_field(img, mask, Gb[0], Gb[1], "Blue Gradient Field")

    return Gr, Gg, Gb, mask


def check_residue(img, center, eccentricity, scalex):
    Gr, Gg, Gb, mask = get_grad(img)
    np_Gs=[np.stack([G[0].numpy(),G[1].numpy()], axis=2) for G in [Gr, Gg, Gb]]
    Gr, Gg, Gb = np_Gs[0], np_Gs[1], np_Gs[2]
    rows, cols = Gr.shape[0], Gr.shape[1]

    print("Gr shape", Gr.shape)

    ortho_mat = np.array([[0, -1], [1, 0]])
    linear_grads = np.concatenate([Gr.reshape(-1, 2), Gg.reshape(-1, 2), Gb.reshape(-1, 2)])
    linear_grads = linear_grads @ ortho_mat

    p = np.stack(list(np.ndindex((rows, cols)))) + np.array([0.5, 0.5])
    p = np.fliplr(p)
    p[:, 1] = cols - p[:, 1] - 1

    print("{}x{} := {}".format(rows, cols, p.shape))

    res = fun(p, linear_grads, center, eccentricity, make_T(scalex, 0), rows=rows, cols=cols, img=img)
    print("Residue stat, max:{}, min:{}, avg:{}, residue shape :{}".format(np.max(res), np.min(res), np.mean(res), res.shape))

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(img)

    res_img = res.reshape(3, *[rows, cols])

    for r in range(3):
        diff = res_img[r]
        # diff = diff - np.min(diff)

        diff = (diff / np.max(diff))
        axs[r+1].imshow(diff, cmap='gray')
    plt.show()


def reconstruct(img):
    Gr, Gg, Gb, mask = get_grad(img)
    o, e, T, s, theta, _, _, _ = elliptic_non_concentric([Gr, Gg, Gb])

    if VERBOSE:
        import matplotlib.pyplot as plt
        print("origin:({:.2f}, {:.2f}), e:{}, s:{:.2f}, theta:{:.3f}".format(o[0], o[1], e, s, np.rad2deg(theta)))
        l = np.arange(0, 2 * np.pi, np.pi / 50)
        points = np.stack([np.cos(l), np.sin(l)]).T @ T
        plt.imshow(sanitize(img))
        r = 100
        plt.plot(o[0] + r*points[:,0], o[1] + r*points[:,1], '--')
        plt.plot(o[0], o[1], 'o', color='black')
        plt.show()
