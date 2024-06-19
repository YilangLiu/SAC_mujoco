import os
import jax
import jax.scipy.spatial
import jax.scipy.spatial.transform
import mujoco
import mujoco.viewer
# from mujoco import mjx
import numpy as np 
import copy

import jax 
from jax.lax import scan
import jax.numpy as jnp 
from jax.scipy.spatial.transform import Rotation as R_jax
from scipy.spatial.transform import Rotation as R_scipy
from jax import jacfwd, grad, vmap, jit
import time
import matplotlib.pyplot as plt

COM_OFFSET = -jnp.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = jnp.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                        ]) + COM_OFFSET
HIP_OFFSETS = HIP_OFFSETS.reshape(12,1)

DEFAULT_POSE = jnp.array([0, 0, 0.27,
                          0, 0, 0, 1, 
                          0, 0.9, -1.8, 
                          0, 0.9, -1.8, 
                          0, 0.9, -1.8, 
                          0, 0.9, -1.8])

CTRL_DEFAULT = jnp.array([0, 0.9, -1.8,
                          0, 0.9, -1.8,
                          0, 0.9, -1.8,
                          0, 0.9, -1.8])

CTRL_LIMITS_MIN = np.array([-33.5, -33.5, -33.5,
                            -33.5, -33.5, -33.5,
                            -33.5, -33.5, -33.5,
                            -33.5, -33.5, -33.5])

CTRL_LIMITS_MAX = np.array([33.5, 33.5, 33.5,
                            33.5, 33.5, 33.5,
                            33.5, 33.5, 33.5,
                            33.5, 33.5, 33.5])

ctrl_scale = 1.5
# here Q is the weights for the state and R is the weights for control
# Q consists of :
# W_x_euler, W_x_base, W_x_joint_pos, W_x_euler_vel, W_x_base_vel, W_x_joint_vel
# R consist of : W_u

Q_pos = np.diag([0, 0, 0,                            # W_x_euler
                1, 1, 5,                                #  W_x_base
             10, 10,20, 10,10,20, 10,10,20, 10,10,20    # W_x_joint_pos
             ])*2.0

Q_vel = np.diag([10, 10, 10,
                 20, 20, 30,
                 10, 10, 20, 10, 10, 20, 10, 10, 20, 10, 10, 20]) * 0.0001

R = np.diag([10,10,20,
             10,10,20,
             10,10,20,
             10,10,20]) * 0.00001
def step_forward(mj_model, mj_data, x_0: np.array, u: np.array):
    # this function is used to step forward the system while keep the model and data
    assert (u.shape[0] ==  mj_model.nu)
    assert (x_0.shape[0] == mj_model.nq+mj_model.nv)

    # make a cooy of current state and model
    state = get_state(mj_model, mj_data)
    ctrl = mj_data.ctrl 

    mj_data.qpos = x_0[:mj_model.nq]
    mj_data.qvel = x_0[mj_model.nq:]
    mj_data.act = []
    mj_data.ctrl = u.reshape(mj_model.nu,)

    print("prev state qpos: ", mj_data.qpos)

    mujoco.mj_step(mj_model, mj_data)

    print("after state qpos: ", mj_data.qpos)
    next_state = np.concatenate((mj_data.qpos, mj_data.qvel), axis=0)

    # set to previous state
    mujoco.mj_setState(mj_model, mj_data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_setState(mj_model, mj_data, ctrl, mujoco.mjtState.mjSTATE_CTRL)

    # mujoco.mj_forward(mj_model, mj_data)

    print("reset state qpos: ", mj_data.qpos)
    return next_state, x_0

def scan(f, init, xs, mj_model=None, mj_data=None, length=None):
    # replacement of jax scan function to make it compatible with
    # the mujoco which is not using jax
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(mj_model, mj_data,carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)

def get_state(model, data):
  nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  state = np.empty(nstate)
  mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
  return state.reshape((nstate, 1))

def get_dfdx_dfdu(x: np.array, u: np.array):
    # Here we need to find the partial derivative of the continuous dynamics
    assert (x.shape[0] == mj_model.nq + mj_model.nv)
    state = get_state(mj_model, mj_data)
    ctrl = mj_data.ctrl 

    mj_data.qpos = x[:mj_model.nq]
    mj_data.qvel = x[mj_model.nq:]
    mj_data.ctrl =  u.reshape(mj_model.nu, )

    dfdx_mujoco = np.zeros((2 * mj_model.nv, 2 * mj_model.nv))
    dfdu_mujoco = np.zeros((2 * mj_model.nv, mj_model.nu))
    epsilon = 1e-6
    flg_centered = True

    mujoco.mjd_transitionFD(mj_model, mj_data, epsilon, flg_centered, dfdx_mujoco, dfdu_mujoco, None, None)

    conti_A = (dfdx_mujoco - np.eye(dfdx_mujoco.shape[0])) / mj_model.opt.timestep
    conti_B = dfdu_mujoco / mj_model.opt.timestep

    mujoco.mj_setState(mj_model, mj_data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_setState(mj_model, mj_data, ctrl, mujoco.mjtState.mjSTATE_CTRL)

    mujoco.mj_forward(mj_model, mj_data)

    return  conti_A, conti_B

def u_tau(x, rho, udef):
    dfdx_mujoco, dfdu_mujoco = get_dfdx_dfdu(x, udef)
    # np.clip((-rho @ dfdu_mujoco + udef), CTRL_LIMITS_MIN , CTRL_LIMITS_MAX) 
    # -rho @ dfdu_mujoco + udef
    # here we
    # print("rho shape",rho.shape)
    # print("dfdu shape", dfdu_mujoco.shape)
    # print("udef shape ", udef.shape)
    # np.clip((rho @ dfdu_mujoco + udef), CTRL_LIMITS_MIN , CTRL_LIMITS_MAX))
    return np.clip(-np.linalg.inv(R) @ dfdu_mujoco.T @ rho + udef, CTRL_LIMITS_MIN / ctrl_scale, CTRL_LIMITS_MAX / ctrl_scale)

def drhodt(rho, v, u):
    dfdx_mujoco, dfdu_mujoco = get_dfdx_dfdu(v, u)
    v_temp = state_quaternion_to_euler(v)

    return  - Drunning_cost(v_temp, u) - dfdx_mujoco.T @ rho

def F_backward(mj_model, mj_data,rho, v):
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
    # _du = DHu(x, u, rho)
    return rhop, rhop

@jit
def running_cost(x:jax.Array, u:jax.Array):
    # dimension of x is the [base_pos, base_orientation, joint position, joint velocity]
    assert(x.shape[0]==2*mj_model.nv)

    # first build regulating cost
    ref_rot =  R_jax.from_quat(DEFAULT_POSE[3:7]).as_euler("zyx", degrees=False) # jnp.array([0.,0.,0.])
    curr_rot = x[3:6]# R.from_quat(x[3:7]).as_euler("zyx", degrees=False)

    #############################
    #### Regulation Cost ########
    #############################
    regulating_cost = (ref_rot - curr_rot).T@ Q_pos[:3, :3] @ (ref_rot - curr_rot) + \
                      (x[:3] - DEFAULT_POSE[:3]).T @ Q_pos[3:6, 3:6] @ (x[:3] - DEFAULT_POSE[:3]) + \
                      (x[6:18] - DEFAULT_POSE[7:]).T @ Q_pos[6:, 6:] @ (x[6:18] - DEFAULT_POSE[7:]) + \
                      x[18:].T@Q_vel@x[18:] + \
                      u.T @ R @ u
    
    # # (x[:3] - DEFAULT_POSE[:3]).T @ W_x_base @ (x[:3] - DEFAULT_POSE[:3]) + \
    
    # #############################
    # #### foot slip Cost ########
    # #############################

    # c_f = 1 
    # c_1 = -30


    # foot_phi_FL, foot_phi_FR, foot_phi_RL, foot_phi_RR =  get_phi_analytical(x[:18])
    # # foot_phi_FL=mjx_data_temp.geom_xpos[FL_id][2]# .reshape(3,1)
    # # foot_phi_FR=mjx_data_temp.geom_xpos[FR_id][2]# .reshape(3,1)
    # # foot_phi_RL=mjx_data_temp.geom_xpos[RL_id][2]# .reshape(3,1)
    # # foot_phi_RR=mjx_data_temp.geom_xpos[RR_id][2]# .reshape(3,1)

    # foot_vel_FR = analytical_leg_jacobian(x[6:9],-1) @ x[24:27]
    # foot_vel_FL = analytical_leg_jacobian(x[9:12],1) @ x[27:30]
    # foot_vel_RR = analytical_leg_jacobian(x[12:15],-1)@ x[30:33]
    # foot_vel_RL = analytical_leg_jacobian(x[15:18],1) @ x[33:]


    # foot_slip_clearance_cost = c_f * (1/(1+jnp.exp(-c_1*foot_phi_FL))*jnp.linalg.norm(foot_vel_FL[:2])**2 +
    #                                   1/(1+jnp.exp(-c_1*foot_phi_FR))*jnp.linalg.norm(foot_vel_FR[:2])**2 +
    #                                   1/(1+jnp.exp(-c_1*foot_phi_RL))*jnp.linalg.norm(foot_vel_RL[:2])**2 +     
    #                                   1/(1+jnp.exp(-c_1*foot_phi_RR))*jnp.linalg.norm(foot_vel_RR[:2])**2)  # only do x-y plane 
    
    # #############################
    # #### Air time Cost ########
    # #############################
    # c_a = 2e3

    # # air_time_cost = 

    # #############################
    # #### Symmetric Control Cost ########
    # #############################
    # c_s = 1e-2

    # # If it is diagonal pair
    # D = jnp.ones((2,2))
    # D = jnp.concatenate((np.zeros((2,1)), D), axis=1)

    # # assemble C_2 matrix 
    # temp_D_1 = jnp.concatenate((D, jnp.zeros((2,3)), jnp.zeros((2,3)), -D), axis=1)
    # temp_D_2 = jnp.concatenate((jnp.zeros((2,3)), D, -D, jnp.zeros((2,3))), axis=1)
    # C_2_diag = jnp.concatenate((temp_D_1, temp_D_2),axis=0)

    # sys_ctrl_cost = c_s*(jnp.linalg.norm(C_2_diag@u.reshape(mj_model.nu,1)))**2

    temp_loss = regulating_cost # +  sys_ctrl_cost + foot_slip_clearance_cost
    return jnp.squeeze(temp_loss)

@jit
def terminal_cost(x):
    # make sure the shape of x is 2*mode.nv
    assert(x.shape[0]==2*mj_model.nv)

    # ref_rot  = R_jax.from_quat(DEFAULT_POSE[3:7]).as_euler("zyx", degrees=False)
    # curr_rot = x[3:6] # R.from_quat(x[3:7]).as_euler("zyx", degrees=False)

    #############################
    #### Regulation Cost ########
    #############################
    # terminal_cost = (ref_rot - curr_rot).T@ Q_pos[:3, :3] @ (ref_rot - curr_rot) + \
    #                   (x[:3] - DEFAULT_POSE[:3]).T @ Q_pos[3:6, 3:6] @ (x[:3] - DEFAULT_POSE[:3]) + \
    #                   (x[6:18] - DEFAULT_POSE[7:]).T @ Q_pos[6:, 6:] @ (x[6:18] - DEFAULT_POSE[7:]) + \
    #                   x[18:].T@Q_vel@x[18:]
    terminal_cost = (x[:3] - DEFAULT_POSE[:3]).T @ Q_pos[3:6, 3:6] @ (x[:3] - DEFAULT_POSE[:3]) + \
                    (x[6:18] - DEFAULT_POSE[7:]).T @ Q_pos[6:, 6:] @ (x[6:18] - DEFAULT_POSE[7:]) + \
                     x[18:].T @ Q_vel @ x[18:]
    
    return terminal_cost

def get_f(mj_model, mj_data,x,u):
    # get continuous version of the dynamics using finite difference
    x_next = step_forward(mj_model, mj_data,x,u)[0]
    x_temp = state_quaternion_to_euler(x)
    x_next_temp = state_quaternion_to_euler(x_next)
    return (x_next_temp - x_temp) / mj_model.opt.timestep

def dJdlam(mj_model, mj_data, rho, x, u2, u1):
    assert(x.shape[0] == mj_model.nq + mj_model.nv)
    assert(rho.shape[0] == 2*mj_model.nv)
    temp = get_f(mj_model, mj_data, x, u2) - get_f(mj_model, mj_data,x, u1)
    return rho.T @ temp

def state_quaternion_to_euler(x):
    euler_temp = R_jax.from_quat(x[3:7]).as_euler("zyx", degrees=False)
    x_temp = jnp.concatenate((x[:3], euler_temp, x[7:]),axis=0)
    assert(x_temp.shape[0] == 2*mj_model.nv)
    return x_temp

def SAC(mj_model, mj_data, x0, U1):
    U1 = U1.at[:-1, :].set(U1[1:, :])
    U1 = U1.at[-1, :].set(U1[-1, :])
    
    
    xf, X = forward_sim(mj_model, mj_data, x0, U1)

    x_f_temp = state_quaternion_to_euler(xf)
    rhof = np.array(Dterm_cost(x_f_temp))

    rho0, rho = backward_sim(mj_model, mj_data, rhof, X, U1)

    U_tau = np.stack(list(map(lambda x_map, rho_map, u_map: u_tau(x_map, rho_map, u_map), X, rho[::-1], U1)))

    # djdl = np.stack(list(
    #     map(lambda rho_map, x_map, u_2_map, u1_map: dJdlam(mj_model, mj_data,rho_map, x_map, u_2_map, u1_map), rho[::-1], X, U_tau,
    #         U1)))
    # print("##################")
    tau_idx = np.argmin(np.stack(list(
        map(lambda rho_map, x_map, u_2_map, u1_map: dJdlam(mj_model, mj_data,rho_map, x_map, u_2_map, u1_map), rho[::-1], X, U_tau,
            U1))))
    # print("u star is ", U_tau[tau_idx])
    # print("##################")

    # print(np.stack(list(
    #     map(lambda rho_map, x_map, u_2_map, u1_map: dJdlam(mj_model, mj_data,rho_map, x_map, u_2_map, u1_map), rho[::-1], X, U_tau,
    #         U1))))

    # np.clip((-rho @ dfdu_mujoco + udef), CTRL_LIMITS_MIN , CTRL_LIMITS_MAX) 
    # print("U_tau is: ", U_tau)
    u_star = U_tau[tau_idx]
    # u_star = np.clip(U_tau[tau_idx], CTRL_LIMITS_MIN , CTRL_LIMITS_MAX)
    # line search
    J_best = np.inf
    lam_idx_best = 0
    
    for lam_idx in range(0, T - tau_idx + 1):
        _U2 = U1.at[tau_idx:tau_idx + lam_idx].set(u_star)
        _xf, _X = forward_sim(mj_model, mj_data, x0, _U2)
        _xf = state_quaternion_to_euler(_xf)
        _X = np.stack(list(map(lambda X: state_quaternion_to_euler(X),_X)))
        J_line = np.mean(np.stack(list(map(lambda X, U: running_cost(X, U), _X, _U2))))
        # print("J line with starting idx: ", tau_idx, " to ", tau_idx + lam_idx, " the cost is ", J_line)
        if J_line < J_best:
            J_best = J_line
            lam_idx_best = lam_idx

    print("J line search results: ", J_line, lam_idx, J_best, lam_idx_best)

    return U1.at[tau_idx:tau_idx + lam_idx_best].set(u_star)

forward_sim = lambda mj_model, mj_data, x0, u: scan(step_forward, x0, u, mj_model, mj_data)
backward_sim = lambda mj_model, mj_data, rho0, v, u: scan(F_backward, rho0, np.concatenate([v[::-1], u[::-1]], axis=1), mj_model, mj_data)
Dterm_cost = grad(terminal_cost)
Drunning_cost = grad(running_cost)

if __name__ == "__main__":
    xml_path =  "./xml/a1.xml"  # "./unitree_a1/scene.xml" 
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    dt = mj_model.opt.timestep 
    
    th = 0.10
    T = int(th / dt)
    kp = 30
    kd = 1

    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    x_0 = jnp.concatenate((mj_data.qpos, mj_data.qvel),axis=0)
    ctrl = jnp.zeros((T,mj_model.nu))

    with mujoco.viewer.launch_passive(mj_model, mj_data,  show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(mj_model, viewer.cam)
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        mj_model.vis.map.force = 0.01
        i = 0
        while viewer.is_running():
           
            # print("start running.......")
            time_sac = time.time()
            
            q = np.array(mj_data.qpos[7:]) # <-- notice that the first 7 values are the base body position (3) and orientation in quaternains (4)
            v = np.array(mj_data.qvel[6:])
            tau = np.clip(kp * (DEFAULT_POSE[7:]-q) + kd * (-v), CTRL_LIMITS_MIN , CTRL_LIMITS_MAX)
            # tau = kp * (DEFAULT_POSE[7:]-q) + kd * (-v)
            # x_0 = step_forward(mj_model, mj_data, x_0, ctrl[0,:])[0]
            
            x_0 = step_forward(mj_model, mj_data, x_0, tau)[0]

            mj_data.qpos = x_0[:mj_model.nq]
            mj_data.qvel = x_0[mj_model.nq:]
            # mj_data.ctrl = ctrl[0,:] # tau
            mj_data.ctrl = tau
            mujoco.mj_step(mj_model, mj_data)
            # print("ctrl is ", ctrl[0,:])

            # print("qpos: ", mj_data.qpos)
            # print("tau is: ",tau)
            # x_0 = step_forward(x_0, ctrl)[0]
            # mj_data.ctrl = ctrl
            # mujoco.mj_step(mj_model, mj_data)

            viewer.sync()
            # time.sleep(0.05)
            i += 1
