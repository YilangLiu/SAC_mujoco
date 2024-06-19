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

default_pose = jnp.array([0, 1.25, 0, 0, 0, 0])
default_vel = jnp.zeros(6)

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
    if (np.isnan(u).any()):
        # print(u)
        xxxx
    mujoco.mj_step(mj_model, mj_data)

    return np.concatenate((mj_data.qpos, mj_data.qvel),axis=0), x_0


def get_dfdx_dfdu(x: np.array, u: np.array):
    # Here we need to find the partial derivative of the continuous dynamics
    assert (x.shape[0] == mj_model.nq + mj_model.nv)
    # print(x)
    # print(x)
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos = x[:mj_model.nq]
    mj_data.qvel = x[mj_model.nq:]
    # mj_data.act = []
    mj_data.ctrl =  u.reshape(mj_model.nu, )

    # mujoco.mj_forward(mj_model, mj_data)

    dfdx_mujoco = np.zeros((2 * mj_model.nv, 2 * mj_model.nv))
    dfdu_mujoco = np.zeros((2 * mj_model.nv, mj_model.nu))
    epsilon = 1e-6
    flg_centered = True

    mujoco.mjd_transitionFD(mj_model, mj_data, epsilon, flg_centered, dfdx_mujoco, dfdu_mujoco, None, None)

    # dfdx_mujoco, dfdu_mujoco
    # dfdx_mujoco*dt + np.identity(mj_model.nq+ mj_model.nv), dfdu_mujoco*dt
    # mid point euler integration

    conti_A = (dfdx_mujoco - np.eye(dfdx_mujoco.shape[0])) / mj_model.opt.timestep
    conti_B = dfdu_mujoco / mj_model.opt.timestep

    return  conti_A, conti_B

def term_cost(x):
    q,qdot = jnp.split(x, 2)
    return 1* (q[:3] - default_pose[:3])@(q[:3] - default_pose[:3]) + 50 * q[3:]@q[3:] + 0.001 * qdot@qdot

def running_cost(x, u):
    q,qdot = jnp.split(x,2)

    ell = 1* (q[:3] - default_pose[:3])@(q[:3] - default_pose[:3]) + 50 * q[3:]@q[3:] + 0.001 * qdot@qdot + 0.0001 * u@u
    return ell
def drhodt(rho, v, u):
  # -DHx(v, u, rho)
    #
    dfdx_mujoco, dfdu_mujoco = get_dfdx_dfdu(v, u)

    # - Drunning_cost(v,u) - dfdx_mujoco @ rho
    # - Drunning_cost(v,u) - dfdx_jax_analytical(v,u).T @ rho

    # if jnp.isnan(Drunning_cost(v,u)).any():
    #     print("gradient: ",Drunning_cost(v,u))
    #     print("v", v)
    #     print("u", u)
    #     q, qdot = jnp.split(v, 2)
    #     print("q is ", q)
    #     print("qdot is ", qdot)
    #     xxx
    # print("dfdx_mujoco", dfdx_mujoco)
    return  - Drunning_cost(v,u) - dfdx_mujoco.T @ rho

def F_backward(rho, v):
    x, u = v[:mj_model.nq + mj_model.nv], v[mj_model.nq + mj_model.nv:]
    assert (x.shape[0] == mj_model.nq + mj_model.nv)
    assert (u.shape[0] == mj_model.nu)

    # Here we change the x to satisfy the step cost calculation
    # x_temp = state_quaternion_to_euler(x)
    rhodot = drhodt(rho, x, u)
    rhop = rho - dt * rhodot
    if (np.isnan(rho).any()):
        print(x)
        print(u)
        print("rho",rho)
        print("drhodot",drhodt(rho, x, u))
        xxx
    # _du = DHu(x, u, rho)
    return rhop, rhop

forward_sim = lambda x0, u: scan(step_forward, x0, u)

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

    return np.clip((-rho @ dfdu_mujoco + udef), -1, 1) # np.clip(-rho @ dfdu(x, udef) + udef, -10, 10)

def get_f(x,u):
    # get continuous version of the dynamics using finite difference
    x_next = step_forward(x,u)[0]

    return (x_next - x) / mj_model.opt.timestep

def dJdlam(rho, x, u2, u1):
    assert(x.shape[0] == mj_model.nq + mj_model.nv)
    assert(rho.shape[0] == 2*mj_model.nv)

    dfdx1, dfdu1 = get_dfdx_dfdu(x, u1)
    dfdx2, dfdu2 = get_dfdx_dfdu(x, u2)

    # dfdx1, dfdu1 = dfdx(x,u1),  dfdu(x,u1)
    # dfdx2, dfdu2 = dfdx(x,u2),  dfdu(x,u2)

    # temp = (dfdx2@x).reshape(2*mj_model.nv,1) + (dfdu2@u2).reshape(2*mj_model.nv,1) -(dfdx1@x).reshape(2*mj_model.nv,1) - (dfdu1@u1).reshape(2*mj_model.nv,1)
    temp = get_f(x, u2) - get_f(x, u1)

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

    U_tau = np.stack(list(map(lambda x_map, rho_map, u_map: u_tau(x_map, rho_map, u_map), X, rho[::-1], U1)))

    tau_idx = np.argmin(np.stack(list(
        map(lambda rho_map, x_map, u_2_map, u1_map: dJdlam(rho_map, x_map, u_2_map, u1_map), rho[::-1], X, U_tau,
            U1))))  # np.argmin(vmap(dJdlam)(rho[::-1], X, U_tau, U1))

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

xml_path = "./mjmodel.xml"
dt = 0.01

mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = dt

dJdlam_store = []

if __name__ == "__main__":
    th = 0.08
    T = int(th / dt)

    x_0 = np.array([0, 1.2, 0, 0, 0, 0, 0, 0.001, 0, 0, 0, 0])
    # set inintial state
    mujoco.mj_resetData(mj_model, mj_data)

    mj_data.ctrl = np.array([1, 1, 1])

    ctrl = jnp.zeros((int(th / dt), mj_model.nu))
    ctrl = SAC(x_0, ctrl)


    tf = 8
    log = []
    u_log = []
    cost_log = []

    print("nv is ", mj_model.nv) # 6
    print("nu is ", mj_model.nu) # 3
    print("nq is ", mj_model.nq) # 6
    mass = np.sum(mj_model.body_mass)

    with mujoco.viewer.launch_passive(mj_model, mj_data,  show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(mj_model, viewer.cam)
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        mj_model.vis.map.force = 0.01
        i = 0

        while viewer.is_running():
            step_start = time.time()

            # x_0 = jnp.concatenate((mj_data.qpos, mj_data.qvel),axis=0)
            # print("start running.......")
            time_sac = time.time()
            log.append(np.array(x_0))
            ctrl = SAC(x_0, ctrl) #
            # ctrl = jnp.array([0, 0])
            # u_log.append(np.array(ctrl[0]))
            # print("one iteration for sac is ", time.time() - time_sac)

            print("ctrl is ", ctrl[0])
            # print("sim time :", mj_data.time)
            # print("ctrl is ", ctrl[0, :])

            x_0 = step_forward(x_0, ctrl[0, :])[0]
            # x_0 = step_forward(x_0, ctrl[0])[0]
            # x_0 = F_forward(x_0, ctrl[0])[0]

            # mj_data_sim.ctrl = ctrl[0, :]
            # mujoco.mj_step(mj_model_sim, mj_data_sim)
            #
            # x_0 = np.concatenate((mj_data_sim.qpos, mj_data_sim.qvel),axis=0)
            #
            # print("sim time :", mj_data_sim.time)
            # print("current state is ", x_0)
            viewer.sync()
            time.sleep(0.001)

    print("finished")

