import mujoco
import mujoco.viewer
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R_jax
from scipy.spatial.transform import Rotation as R_scipy
from jax import jacfwd, grad, vmap, jit
import time
import matplotlib.pyplot as plt

x_dim = 4
u_dim = 1

_sat = 4.0

dt = 0.1


def step_forward(x_0:np.array, u:np.array):
    # make sure the shape of control is the dimension fo the torque

    assert (u.shape[0] ==  mj_model.nu)
    assert (x_0.shape[0] == mj_model.nq+mj_model.nv)
    # mujoco.mj_resetData(mjx_model, mjx_data)
    mj_data.qpos = x_0[:mj_model.nq]
    mj_data.qvel = x_0[mj_model.nq:]
    mj_data.act = []
    mj_data.ctrl = u.reshape(mj_model.nu,)
    # mjx_data.replace(qpos=x_0[:mj_model.nq])
    # mjx_data.replace(qvel=x_0[mj_model.nq:])
    # mjx_data.replace(act=[])
    # mjx_data.replace(ctrl=u.reshape(mj_model.nu,))

    # mujoco.mj_step(mj_model, mj_data)
    mujoco.mj_step(mj_model, mj_data)

    return np.concatenate((mj_data.qpos, mj_data.qvel),axis=0), x_0


def get_dfdx_dfdu(x: np.array, u: np.array):
    # Here we need to find the partial derivative of the continuous dynamics
    assert (x.shape[0] == mj_model.nq + mj_model.nv)

    mujoco.mj_resetData(mj_model, mj_data)
    # mj_data.qpos =  np.array([0., 0.]) # x[:mj_model.nq]
    # mj_data.qvel =  np.array([0., 0.]) # x[mj_model.nq:]
    #
    mj_data.qpos = x[:mj_model.nq]
    mj_data.qvel = x[mj_model.nq:]
    #
    # mj_data.ctrl = np.array([0.])  # u.reshape(mj_model.nu, )
    mj_data.ctrl = u.reshape(mj_model.nu, )
    # mj_data.act = []

    # mujoco.mj_forward(mj_model, mj_data)

    dfdx_mujoco = np.zeros((2 * mj_model.nv, 2 * mj_model.nv))
    dfdu_mujoco = np.zeros((2 * mj_model.nv, mj_model.nu))
    epsilon = 1e-6
    flg_centered = True

    mujoco.mjd_transitionFD(mj_model, mj_data, epsilon, flg_centered, dfdx_mujoco, dfdu_mujoco, None, None)

    # dfdx_mujoco, dfdu_mujoco
    # dfdx_mujoco*dt + np.identity(mj_model.nq+ mj_model.nv), dfdu_mujoco*dt
    # mid point euler integration
    # e = np.linalg.inv(np.eye(mj_model.nq+ mj_model.nv) - (dfdx_mujoco * dt)/2)
    # new_A = e @ (np.eye(mj_model.nq + mj_model.nv) + (dfdx_mujoco * dt)/2)
    # new_B = e @ dfdu_mujoco * dt


    # new_A, new_B
    # dfdx_mujoco, dfdu_mujoco

    conti_A = (dfdx_mujoco - np.eye(dfdx_mujoco.shape[0])) / mj_model.opt.timestep
    conti_B = dfdu_mujoco / mj_model.opt.timestep
    return  conti_A, conti_B # dfdx_mujoco, dfdu_mujoco # new_A, new_B


def f1(x):
    # th, p, thdot, xdot = x
    p, th, xdot, thdot  = x
    # u_acc = _sat*jnp.tanh(a[0])
    fdot = jnp.array([
                xdot, thdot,
                -0.2*xdot,
                9.81*jnp.sin(th)/1.0-0.2*thdot])
    return fdot
def f2(x):
    # th, p, thdot, xdot = x
    p, th, xdot, thdot = x
    return jnp.array([
        [0.],
        [0.],
        [1.0],
        [jnp.cos(th)/1.0]])

def f(x, u):
    return f1(x) + f2(x)@u

def F_forward(x, u):
    xdot = f(x, u)
    xp = x + dt * xdot
    return xp, x

def F_forward_analytical(x,u):
    return x + f(x,u) * dt

dfdx_jax = jacfwd(f, argnums=0)
dfdu_jax = jacfwd(f, argnums=1)

dfdx_jax_analytical = jacfwd(F_forward_analytical, argnums=0)
dfdu_jax_analytical = jacfwd(F_forward_analytical, argnums=1)
def term_cost(x):
    q,qdot = jnp.split(x,2)
    return -1.0*jnp.cos(q[1]) + q[0]**2 + 0.01* qdot@qdot #+ 0.001*qdot[0]**2 + 0.001*qdot[1]**2

def running_cost(x, u):
    q,qdot = jnp.split(x,2)
    ell = -10.0*jnp.cos(q[1]) + 0.1*q[0]**2 + 0.001*jnp.sum(u**2) + 0.001 * qdot@qdot
    return ell
def drhodt(rho, v, u):
  # -DHx(v, u, rho)
    #
    dfdx_mujoco, dfdu_mujoco = get_dfdx_dfdu(v, u)
    # - Drunning_cost(v,u) - dfdx_mujoco @ rho
    # - Drunning_cost(v,u) - dfdx_jax_analytical(v,u).T @ rho
    return  - Drunning_cost(v,u) - dfdx_mujoco.T @ rho

def F_backward(rho, v):
    x, u = v[:mj_model.nq + mj_model.nv], v[mj_model.nq + mj_model.nv:]
    assert (x.shape[0] == mj_model.nq + mj_model.nv)
    assert (u.shape[0] == mj_model.nu)

    # Here we change the x to satisfy the step cost calculation
    # x_temp = state_quaternion_to_euler(x)
    rhodot = drhodt(rho, x, u)

    rhop = rho - dt * rhodot
    # _du = DHu(x, u, rho)
    return rhop, rhop

forward_sim = lambda x0, u: scan(step_forward, x0, u)
forward_sim_analytical = lambda x0, u: scan(F_forward, x0, u)
backward_sim = lambda rho0, v, u: scan(F_backward, rho0, jnp.concatenate([v[::-1], u[::-1]], axis=1))#, reverse=True)


def scan(f, init, xs, length=None):
  # replacement of jax scan function to make it compatible with
  # the mujoco which is not using jax
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, jnp.stack(ys)


def u_tau(x, rho, udef):
    dfdx_mujoco, dfdu_mujoco = get_dfdx_dfdu(x, udef)

    return np.clip((-rho @ dfdu_mujoco + udef), -1., 1.) # np.clip(-rho @ dfdu(x, udef) + udef, -10, 10)

def get_f(x,u):
    # get continuous version of the dynamics using finite difference
    x_next = step_forward(x,u)[0]

    return (x_next - x) / mj_model.opt.timestep
def dJdlam(rho, x, u2, u1):
    assert(x.shape[0] == mj_model.nq + mj_model.nv)
    assert(rho.shape[0] == 2*mj_model.nv)

    # dfdx1, dfdu1 = get_dfdx_dfdu(x, u1)
    # dfdx2, dfdu2 = get_dfdx_dfdu(x, u2)

    # dfdx1, dfdu1 = dfdx(x,u1),  dfdu(x,u1)
    # dfdx2, dfdu2 = dfdx(x,u2),  dfdu(x,u2)

    # temp = (dfdx2@x).reshape(2*mj_model.nv,1) + (dfdu2@u2).reshape(2*mj_model.nv,1) -(dfdx1@x).reshape(2*mj_model.nv,1) - (dfdu1@u1).reshape(2*mj_model.nv,1)
    # temp = ((dfdx2@x).reshape(2*mj_model.nv,1) + (dfdu2*u2).reshape(2*mj_model.nv,1) -
    #             (dfdx1@x).reshape(2*mj_model.nv,1) - (dfdu1*u1).reshape(2*mj_model.nv,1))
    temp = get_f(x,u2) - get_f(x, u1)

    return rho.T @ temp

# def dJdlam(rho, x, u2, u1):
#     # f2(x) @ (u2-u1)
#     # dfdx(x,u2)@x+dfdu(x,u2) - dfdx(x,u1)@x - dfdu(x,u1)
#     a = f2(x) @ (u2-u1)
#     b = (dfdx(x,u2).T@x).reshape(4,1) + dfdu(x,u2)*u2 -(dfdx(x,u1).T@x).reshape(4,1) - dfdu(x,u1)*u1
#     b = b.reshape(4,)
#     res = rho @ b
#     return res

def SAC(x0, U1):
    U1 = U1.at[:-1, :].set(U1[1:, :])
    U1 = U1.at[-1, :].set(U1[-1, :])


    xf, X = forward_sim(x0, U1)
    # xf, X = forward_sim_analytical(x0, U1)

    rhof = np.array(Dterm_cost(xf))
    # Here rhof is 2*mj_model.nv
    time_now = time.time()
    rho0, rho = backward_sim(rhof, X, U1)
    # print("time for backward_sim", time.time()-time_now)
    # U_tau = u_tau(X, rho[::-1], U1)


    # TODO:
    # check clip first or dJdlam first
    U_tau = np.stack(list(map(lambda x_map, rho_map, u_map: u_tau(x_map, rho_map, u_map), X, rho[::-1], U1)))

    tau_idx = np.argmin(np.stack(list(
        map(lambda rho_map, x_map, u_2_map, u1_map: dJdlam(rho_map, x_map, u_2_map, u1_map), rho[::-1], X, U_tau,
            U1))))  # np.argmin(vmap(dJdlam)(rho[::-1], X, U_tau, U1))

    dJdlam_store.append(np.stack(list(
        map(lambda rho_map, x_map, u_2_map, u1_map: dJdlam(rho_map, x_map, u_2_map, u1_map), rho[::-1], X, U_tau,
            U1))))

    # tau_idx = np.argmin(vmap(dJdlam)(rho[::-1], X, U_tau, U1))

    u_star = U_tau[tau_idx]

    # line search
    J_best = np.inf
    lam_idx_best = 0

    for lam_idx in range(0, T - tau_idx + 1):
        _U2 = U1.at[tau_idx:tau_idx + lam_idx].set(u_star)

        _xf, _X = forward_sim(x0, _U2)
        # _xf, _X = forward_sim_analytical(x0, _U2)

        time_now = time.time()

        J_line = np.mean(np.stack(list(map(lambda X, U: running_cost(X, U), _X, _U2))))
        # J_line = np.mean(vmap(running_cost)(_X, _U2)) + term_cost(_xf)
        if J_line < J_best:
            J_best = J_line
            lam_idx_best = lam_idx
    print("J line search results: ", J_line, lam_idx, J_best, lam_idx_best)
    return U1.at[tau_idx:tau_idx + lam_idx_best].set(u_star)

Dterm_cost = grad(term_cost)
Drunning_cost = grad(running_cost)

xml_path = "./cartpole.xml"

mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = dt

dJdlam_store = []

if __name__ == "__main__":
    th = 0.8
    T = int(th / dt)

    x_0 = np.array([0, 3.1, 0., 0.])
    # set inintial state
    ctrl = jnp.zeros((int(th / dt), 1))
    ctrl = SAC(x_0, ctrl)

    tf = 8
    log = []
    u_log = []
    cost_log = []

    print("nv is ", mj_model.nv)
    print("nu is ", mj_model.nu)
    print("nq is ", mj_model.nq)
    mass = np.sum(mj_model.body_mass)

    for t in range(int(tf/dt)):
        step_start = time.time()

        # x_0 = jnp.concatenate((mj_data.qpos, mj_data.qvel),axis=0)
        # print("start running.......")
        time_sac = time.time()
        log.append(np.array(x_0))
        ctrl = SAC(x_0, ctrl) # jnp.array([0])
        u_log.append(np.array(ctrl[0]))
        print("one iteration for sac is ", time.time() - time_sac)
        print("ctrl is ", ctrl[0])
        # print("ctrl is ", ctrl[0, :])
        # x_0 = step_forward(x_0, ctrl[0, :])[0]
        x_0 = step_forward(x_0, ctrl[0])[0]
        # x_0 = F_forward(x_0, ctrl[0])[0]

    print("finished")
    z = np.stack(log)
    v = np.stack(u_log)
    # lamb = np.array(dJdlam_store).reshape(-1,T-1)
    np.savetxt("z.txt",z, delimiter=',')
    np.savetxt("v.txt", v, delimiter=',')
    # np.savetxt("lam.txt", lamb, delimiter=',')
    # plt.plot(z)
    # plt.stem(v, 'k')
    # plt.show()
