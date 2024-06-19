import os
import jax
import jax.scipy.spatial
import jax.scipy.spatial.transform
import mujoco
import mujoco.viewer
# from mujoco import mjx
import numpy as np 

import jax 
from jax.lax import scan
import jax.numpy as jnp 
from jax.scipy.spatial.transform import Rotation as R_jax
from scipy.spatial.transform import Rotation as R_scipy
from jax import jacfwd, grad, vmap, jit
import time
import matplotlib.pyplot as plt

# from brax.io import html, mjcf, model

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
CTRL_DEFAULT = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])

CTRL_LIMITS_MIN = np.array([-0.802851, -1.0472, -2.69653,
                            -0.802851, -1.0472, -2.69653,
                            -0.802851, -1.0472, -2.69653,
                            -0.802851, -1.0472, -2.69653])

CTRL_LIMITS_MAX = np.array([0.802851, 4.18879, -0.916298,
                            0.802851, 4.18879, -0.916298,
                            0.802851, 4.18879, -0.916298,
                            0.802851, 4.18879, -0.916298])

def foot_position_in_hip_frame(angles: jax.Array, l_hip_sign):
    theta_ab, theta_hip, theta_knee = angles[0], angles[1], angles[2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    leg_distance = jnp.sqrt(l_up**2 + l_low**2 +
                            2 * l_up * l_low * jnp.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * jnp.sin(eff_swing)
    off_z_hip = -leg_distance * jnp.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = jnp.cos(theta_ab) * off_y_hip - jnp.sin(theta_ab) * off_z_hip
    off_z = jnp.sin(theta_ab) * off_y_hip + jnp.cos(theta_ab) * off_z_hip
    return jnp.array([off_x, off_y, off_z]).reshape(3,1)


def jax_fwd_dynamics(motor_angles: jax.Array):
    # FR FL RR RL
    assert(motor_angles.shape[0]==12)
    foot_positions = jnp.concatenate((
        foot_position_in_hip_frame(motor_angles[:3], l_hip_sign=1),
        foot_position_in_hip_frame(motor_angles[3:6],l_hip_sign=-1),
        foot_position_in_hip_frame(motor_angles[6:9],l_hip_sign=1),
        foot_position_in_hip_frame(motor_angles[9:], l_hip_sign=-1))
    )

    return foot_positions + HIP_OFFSETS

def reset(mj_model, mj_data):
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)

def step_forward(x_0:np.array, u:np.array):
    # make sure the shape of control is the dimension fo the torque 

    assert (u.shape[0] ==  mj_model.nu)
    assert (x_0.shape[0] == mj_model.nq+mj_model.nv)
    # mujoco.mj_resetData(mjx_model, mjx_data)
    mj_data.qpos = x_0[:mj_model.nq]
    mj_data.qvel = x_0[mj_model.nq:]
    mj_data.act = []
    mj_data.ctrl = u.reshape(mj_model.nu,)
    mj_model.opt.timestep = dt
    # mjx_data.replace(qpos=x_0[:mj_model.nq])
    # mjx_data.replace(qvel=x_0[mj_model.nq:])
    # mjx_data.replace(act=[])
    # mjx_data.replace(ctrl=u.reshape(mj_model.nu,))

    # mujoco.mj_step(mj_model, mj_data)
    mujoco.mj_step(mj_model, mj_data)
    # xxx
    return np.concatenate((mj_data.qpos, mj_data.qvel),axis=0), x_0

def scan_np(f, init, xs, length=None):
  # replacement of jax scan function to make it compatible with
  # the mujoco which is not using jax 
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)


def drhodt(rho, v, u):
    # - Drunning_cost(v) - dfdx(v, u).T @ rho
    assert(rho.shape[0]==2*mj_model.nv)
    # print("drhodt v shape", v.shape)
    # print("drhodt u shape", u.shape)
    dfdx_mujoco, dfdu_mujoco = get_dfdx_dfdu(v, u)
    
    v_temp = state_quaternion_to_euler(v)
    if (np.isnan((D_running_cost(v_temp,u))).any()):
        print("################### \n nan detected")
    D_cost = np.nan_to_num(np.asarray(D_running_cost(v_temp,u)))
    return -D_cost - dfdx_mujoco@rho


def F_backward(rho, v):
    
    x,u = v[:mj_model.nq+mj_model.nv], v[mj_model.nq+mj_model.nv:]
    assert(x.shape[0]==mj_model.nq+mj_model.nv)
    assert(u.shape[0]==mj_model.nu)

    # Here we change the x to satisfy the step cost calculation 
    # x_temp = state_quaternion_to_euler(x)
    rhodot = drhodt(rho, x, u)

    rhop = rho - dt * rhodot
    # _du = DHu(x, u, rho)
    return rhop, rhop


def get_dfdx_dfdu(x:np.array, u:np.array):
    assert(x.shape[0] == mj_model.nq + mj_model.nv)
    # print(x)
    # print(x)

    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos = x[:mj_model.nq]
    mj_data.qvel = x[mj_model.nq:]
    mj_data.act = []
    mj_data.ctrl = u.reshape(mj_model.nu,)

    # mujoco.mj_forward(mj_model, mj_data)
    # mjx_data = mjx.put_data(mj_model, mj_data)

    # mjx_data_temp = mjx_data.replace(qpos=x[:mj_model.nq])
    # mjx_data_temp = mjx_data_temp.replace(qvel=x[mj_model.nq:])
    # mjx_data_temp = mjx_data_temp.replace(act=[])
    # mjx_data_temp = mjx_data_temp.replace(ctrl=u.reshape(mj_model.nu,))

    # mj_data = mjx.get_data(mj_mo  del, mjx_data_temp)

    dfdx_mujoco = np.zeros((2 * mj_model.nv, 2 * mj_model.nv))
    dfdu_mujoco = np.zeros((2 * mj_model.nv, mj_model.nu))
    epsilon = 1e-6
    flg_centered = True

    mujoco.mjd_transitionFD(mj_model, mj_data, epsilon, flg_centered, dfdx_mujoco, dfdu_mujoco, None, None)

    # re-arrange the dfdx dfdu
    # dfdx_mujoco[:, [6, 8]]   = dfdx_mujoco[:, [8, 6]]
    # dfdx_mujoco[:, [9, 11]]  = dfdx_mujoco[:, [11, 9]]
    # dfdx_mujoco[:, [12, 14]] = dfdx_mujoco[:, [14, 12]]
    # dfdx_mujoco[:, [15, 17]] = dfdx_mujoco[:, [17, 15]]
    #
    # dfdx_mujoco[:, [23, 25]] = dfdx_mujoco[:, [25, 23]]
    # dfdx_mujoco[:, [26, 28]] = dfdx_mujoco[:, [28, 26]]
    # dfdx_mujoco[:, [29, 32]] = dfdx_mujoco[:, [32, 29]]
    # dfdx_mujoco[:, [33, 35]] = dfdx_mujoco[:, [35, 33]]
    #
    # dfdu_mujoco[:, [0, 2]]  = dfdu_mujoco[:, [2, 0]]
    # dfdu_mujoco[:, [3, 5]]  = dfdu_mujoco[:, [5, 3]]
    # dfdu_mujoco[:, [6, 8]]  = dfdu_mujoco[:, [8, 6]]
    # dfdu_mujoco[:, [9, 11]] = dfdu_mujoco[:, [11, 9]]

    e = np.linalg.inv(np.eye(mj_model.nv * 2) - (dfdx_mujoco * dt) / 2)
    new_A = e @ (np.eye(mj_model.nv * 2) + (dfdx_mujoco * dt) / 2)
    new_B = e @ dfdu_mujoco * dt
    # print("new_A is ", new_A)
    return new_A, new_B

def get_phi_analytical(x):
    
    assert(x.shape[0]==mj_model.nv) # quaternion is already converted to euler 

    base_rot = R_jax.from_euler('zyx', x[3:6]).as_matrix()

    foot_pos = jax_fwd_dynamics(x[6:])

    foot_pos_abs_FR = base_rot @ (foot_pos[0:3].reshape(3,1))
    foot_pos_abs_FL = base_rot @ (foot_pos[3:6].reshape(3,1))
    foot_pos_abs_RR = base_rot @ (foot_pos[6:9].reshape(3,1))
    foot_pos_abs_RL = base_rot @ (foot_pos[9:12].reshape(3,1))

    # print("FL pos rel: ",foot_pos[0:3].reshape(3,1),"foot_pos_abs_FL", foot_pos_abs_FL)
    # print("base position: ",x[:3])
    # print("FR pos : ", foot_pos[0:3].T, " FL pos : ", foot_pos[3:6].T, " RR pos : ", foot_pos[6:9].T," RL pos : ", foot_pos[9:12].T)
    
    
    foot_pos_world_FR =  foot_pos_abs_FR + x[:3].reshape(3,1) 
    foot_pos_world_FL =  foot_pos_abs_FL + x[:3].reshape(3,1)   
    foot_pos_world_RR =  foot_pos_abs_RR + x[:3].reshape(3,1)   
    foot_pos_world_RL =  foot_pos_abs_RL + x[:3].reshape(3,1)   
    
    return foot_pos_world_FL[2], foot_pos_world_FR[2], foot_pos_world_RL[2], foot_pos_world_RR[2]


def analytical_leg_jacobian(leg_angles, leg_id):
    """
    Computes the analytical Jacobian.
    Args:
    ` leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
        l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * (-1)**(leg_id + 1)

    t1, t2, t3 = leg_angles[0], leg_angles[1], leg_angles[2]

    l_eff = jnp.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * jnp.cos(t3))
    t_eff = t2 + t3 / 2
    J = jnp.zeros((3, 3))
    J = J.at[0, 0].set(0)
    J = J.at[0, 1].set(-l_eff * jnp.cos(t_eff))
    J = J.at[0, 2].set(l_low * l_up * jnp.sin(t3) * jnp.sin(t_eff) / l_eff - l_eff * jnp.cos(t_eff) / 2)
    J = J.at[1, 0].set(-l_hip * jnp.sin(t1) + l_eff * jnp.cos(t1) * jnp.cos(t_eff))
    J = J.at[1, 1].set(-l_eff * jnp.sin(t1) * jnp.sin(t_eff))
    J = J.at[1, 2].set(-l_low * l_up * jnp.sin(t1) * jnp.sin(t3) * jnp.cos(
        t_eff) / l_eff - l_eff * jnp.sin(t1) * jnp.sin(t_eff) / 2) 
    J = J.at[2, 0].set(l_hip * jnp.cos(t1) + l_eff * jnp.sin(t1) * jnp.cos(t_eff))
    J = J.at[2, 1].set(l_eff * jnp.sin(t_eff) * jnp.cos(t1))
    J = J.at[2, 2].set(l_low * l_up * jnp.sin(t3) * jnp.cos(t1) * jnp.cos(
        t_eff) / l_eff + l_eff * jnp.sin(t_eff) * jnp.cos(t1) / 2)
    return J

@jit
def build_step_loss(x:jax.Array, u:jax.Array):
    # dimension of x is the [base_pos, base_orientation, joint position, joint velocity]
    assert(x.shape[0]==2*mj_model.nv)
    # quad_temp = R_jax.from_euler('zyx', x[3:6], degrees=False).as_quat()
    # x_temp = jnp.concatenate((x[:3],quad_temp,x[6:]),axis=0)
    # assert(x_temp.shape[0]==mj_model.nq+mj_model.nv)


    # mjx_data_temp = mjx_data.replace(qpos=x_temp[:mj_model.nq])
    # mjx_data_temp = mjx_data_temp.replace(qvel=x_temp[mj_model.nq:])
    # mjx_data_temp = mjx_data_temp.replace(act=[])
    # mjx_data_temp = mjx_data_temp.replace(ctrl=u.reshape(mj_model.nu,))
    
    # # mujoco.mj_forward(mjx_model, mjx_data)
    # mjx_data_temp = mjx.forward(mjx_model, mjx_data_temp)
    # print("this line passed")
    # first build regulating cost
    ref_rot =  R_jax.from_quat(DEFAULT_POSE[3:7]).as_euler("zyx", degrees=False) # jnp.array([0.,0.,0.])
    curr_rot = x[3:6]# R.from_quat(x[3:7]).as_euler("zyx", degrees=False)

    # Matrix for euler difference
    W_x_euler = jnp.diag(jnp.array([20, 20 , 100]))*1.0
    W_x_base = jnp.diag(jnp.array([10,10,100])) * 1.0
    W_x_base_vel = jnp.diag(jnp.array([0,0,0]))
    W_x_euler_vel = jnp.diag(jnp.array([0,0,0]))
    W_x_joint_pos = jnp.diag(jnp.array([20,20,40, 20,20,40, 20,20,40, 20,20,40]))
    W_x_joint_vel = jnp.diag(jnp.array([0,0,0, 0,0,0, 0,0,0, 0,0,0]))
    W_u = jnp.diag(jnp.array([0.1,0.1,0.1, 0.1,0.1,0.1, 0.1,0.1,0.1, 0.1,0.1,0.1]))*0.001


    #############################
    #### Regulation Cost ########
    #############################
    regulating_cost = (ref_rot - curr_rot).T@ W_x_euler@ (ref_rot - curr_rot) + \
                      (x[:3] - DEFAULT_POSE[:3]).T @ W_x_base @ (x[:3] - DEFAULT_POSE[:3]) + \
                      (x[6:18] - DEFAULT_POSE[7:]).T @ W_x_joint_pos @ (x[6:18] - DEFAULT_POSE[7:]) + \
                      x[18:21].T@W_x_base_vel@x[18:21] + \
                      x[21:24].T @ W_x_euler_vel @ x[21:24] + \
                      x[24:].T @ W_x_joint_vel @ x[24:] + \
                      u.T @ W_u @ u
    
    # (x[:3] - DEFAULT_POSE[:3]).T @ W_x_base @ (x[:3] - DEFAULT_POSE[:3]) + \
    
    #############################
    #### foot slip Cost ########
    #############################

    c_f = 1 
    c_1 = -30


    foot_phi_FL, foot_phi_FR, foot_phi_RL, foot_phi_RR =  get_phi_analytical(x[:18])
    # foot_phi_FL=mjx_data_temp.geom_xpos[FL_id][2]# .reshape(3,1)
    # foot_phi_FR=mjx_data_temp.geom_xpos[FR_id][2]# .reshape(3,1)
    # foot_phi_RL=mjx_data_temp.geom_xpos[RL_id][2]# .reshape(3,1)
    # foot_phi_RR=mjx_data_temp.geom_xpos[RR_id][2]# .reshape(3,1)

    foot_vel_FR = analytical_leg_jacobian(x[6:9],-1) @ x[24:27]
    foot_vel_FL = analytical_leg_jacobian(x[9:12],1) @ x[27:30]
    foot_vel_RR = analytical_leg_jacobian(x[12:15],-1)@ x[30:33]
    foot_vel_RL = analytical_leg_jacobian(x[15:18],1) @ x[33:]


    foot_slip_clearance_cost = c_f * (1/(1+jnp.exp(-c_1*foot_phi_FL))*jnp.linalg.norm(foot_vel_FL[:2])**2 +
                                      1/(1+jnp.exp(-c_1*foot_phi_FR))*jnp.linalg.norm(foot_vel_FR[:2])**2 +
                                      1/(1+jnp.exp(-c_1*foot_phi_RL))*jnp.linalg.norm(foot_vel_RL[:2])**2 +     
                                      1/(1+jnp.exp(-c_1*foot_phi_RR))*jnp.linalg.norm(foot_vel_RR[:2])**2)  # only do x-y plane 
    
    #############################
    #### Air time Cost ########
    #############################
    c_a = 2e3

    # air_time_cost = 

    #############################
    #### Symmetric Control Cost ########
    #############################
    c_s = 1e-2

    # If it is diagonal pair
    D = jnp.ones((2,2))
    D = jnp.concatenate((np.zeros((2,1)), D), axis=1)

    # assemble C_2 matrix 
    temp_D_1 = jnp.concatenate((D, jnp.zeros((2,3)), jnp.zeros((2,3)), -D), axis=1)
    temp_D_2 = jnp.concatenate((jnp.zeros((2,3)), D, -D, jnp.zeros((2,3))), axis=1)
    C_2_diag = jnp.concatenate((temp_D_1, temp_D_2),axis=0)

    sys_ctrl_cost = c_s*(jnp.linalg.norm(C_2_diag@u.reshape(mj_model.nu,1)))**2

    temp_loss = regulating_cost # +  sys_ctrl_cost + foot_slip_clearance_cost
    return jnp.squeeze(temp_loss)

@jit
def build_terminal_cost(x):
    # make sure the shape of x is 2*mode.nv
    assert(x.shape[0]==2*mj_model.nv)

    ref_rot  = R_jax.from_quat(DEFAULT_POSE[3:7]).as_euler("zyx", degrees=False)
    curr_rot = x[3:6] # R.from_quat(x[3:7]).as_euler("zyx", degrees=False)

    # Matrix for euler difference
    W_x_euler = jnp.diag(jnp.array([20 ,20 ,100]))*1.0
    W_x_base = jnp.diag(jnp.array([10, 10,100]))*1.0
    W_x_euler_vel = jnp.diag(jnp.array([0, 0, 0]))
    W_x_base_vel = jnp.diag(jnp.array([5,5,10]))*0.0
    W_x_joint_pos = jnp.diag(jnp.array([20,20,40, 20,20,40, 20,20,40, 20,20,40]))
    W_x_joint_vel = jnp.diag(jnp.array([0,0,0, 0,0,0, 0,0,0, 0,0,0]))

    #############################
    #### Regulation Cost ########
    #############################
    terminal_cost = (ref_rot - curr_rot).T@ W_x_euler@ (ref_rot - curr_rot) + \
                      (x[:3] - DEFAULT_POSE[:3]).T @ W_x_base @ (x[:3] - DEFAULT_POSE[:3]) + \
                      (x[6:18] - DEFAULT_POSE[7:]).T @ W_x_joint_pos @ (x[6:18] - DEFAULT_POSE[7:]) + \
                      x[18:21].T@W_x_base_vel@x[18:21] + \
                      x[21:24].T @ W_x_euler_vel @ x[21:24] + \
                      x[24:].T @ W_x_joint_vel @ x[24:]
    
    return terminal_cost


def u_tau(x, rho, udef):
    # u_clipped = []
    # for i in range(x.shape[0]):
    #     dfdx,dfdu = get_dfdx_dfdu(x[i,:], udef[i,:])
    #     u_clipped_temp = jnp.clip(-rho[i,:] @ dfdu + udef[i,:], -5,5)
    #     u_clipped.append(u_clipped_temp)
    dfdx_mujoco, dfdu_mujoco = get_dfdx_dfdu(x, udef)

    # print(rho.shape)
    # print(dfdu_mujoco.shape)
    # print(udef.shape)
    # print((-rho.T @ dfdu_mujoco).shape)
    # print(dfdu_mujoco)
    # xxx
    return np.clip((-rho.T @ dfdu_mujoco + udef) * 2.0, CTRL_LIMITS_MIN, CTRL_LIMITS_MAX)  #  jnp.stack(u_clipped)

def dJdlam(rho, x, u2, u1):
    assert(x.shape[0]==mj_model.nq + mj_model.nv)
    assert(rho.shape[0]==2*mj_model.nv)
    euler_temp = R_scipy.from_quat(x[3:7]).as_euler("zyx", degrees=False)
    x_temp = np.concatenate((x[:3],euler_temp,x[7:]),axis=0)
    assert(x_temp.shape[0]==2*mj_model.nv)

    dfdx1, dfdu1 = get_dfdx_dfdu(x, u1)
    dfdx2, dfdu2 = get_dfdx_dfdu(x, u2)
    temp = ((dfdx2@x_temp).reshape(2*mj_model.nv,1) + (dfdu2@u2).reshape(2*mj_model.nv,1) -
            (dfdx1@x_temp).reshape(2*mj_model.nv,1) - (dfdu1@u1).reshape(2*mj_model.nv,1))

    # print("dfdu2@u2 shape: ", (dfdu2@u2).shape)
    # print("temp shape: ", temp.shape)
    # print("rho shape", rho.shape)
    # xxx
    return rho.T @ temp

def state_quaternion_to_euler(x):
    euler_temp = R_jax.from_quat(x[3:7]).as_euler("zyx", degrees=False)
    x_temp = jnp.concatenate((x[:3],euler_temp,x[7:]),axis=0)
    assert(x_temp.shape[0] == 2*mj_model.nv)
    return x_temp


def SAC(x0, U1):

    U1 = U1.at[:-1,:].set(U1[1:,:])
    U1 = U1.at[-1,:].set(U1[-1,:]) # here the shape is (time_steps, mj_model,nu)

    xf, X = forward_sim(x0, U1)

    x_temp = state_quaternion_to_euler(xf)
    rhof = np.array(D_terminal_cost(x_temp))
    # Here rhof is 2*mj_model.nv
    time_now = time.time()
    rho0, rho = backward_sim(rhof, X, U1)
    # print("time for backward_sim", time.time()-time_now)
    # U_tau = u_tau(X, rho[::-1], U1)

    U_tau = np.stack(list(map(lambda x_map, rho_map, u_map: u_tau(x_map, rho_map, u_map), X, rho[::-1], U1)))

    tau_idx = np.argmin(np.stack(list(
            map(lambda rho_map, x_map, u_2_map, u1_map: dJdlam(rho_map, x_map, u_2_map, u1_map), rho[::-1], X, U_tau,
            U1)))) # np.argmin(vmap(dJdlam)(rho[::-1], X, U_tau, U1))

    u_star = U_tau[tau_idx]
    # print("U_tau is ",U_tau)
    # print("idx is ", tau_idx)
    # print("u_star is: ",u_star)

    # line search
    J_best = np.inf
    lam_idx_best = 0
    # lam_idxs = jnp.arange(0, T - tau_idx+1)
    # J_line = vmap(linesearch, in_axes=(0, None, None, None))(lam_idxs, tau_idx, U1, u_star)
    # lam_idx_best = lam_idxs[np.argmin(J_line)]
    
    for lam_idx in range(0, T-tau_idx+1):
        _U2 = U1.at[tau_idx:tau_idx+lam_idx,:].set(u_star)
        # plt.plot(_U2)
        # plt.plot(u) vb
        # plt.pause(0.0001)
        # input()
        _xf, _X = forward_sim(x0, _U2)
        _xf = state_quaternion_to_euler(_xf)

        _X = np.stack(list(map(lambda X: state_quaternion_to_euler(X),_X))) # vmap(state_quaternion_to_euler)(_X)
        time_now = time.time()
        # print("_X shape",_X.shape)
        # print("_X init",_X[0,:])
        # print("_X", _X[-1,:])
        # xxx
        J_line = np.mean(np.stack(list(map(lambda X, U: build_step_loss(X,U), _X, _U2))))
        # print("time for J_line", time.time()-time_now)
        # J_line = np.mean(vmap(build_step_loss)(_X, _U2)) + build_terminal_cost(_xf)
        if J_line < J_best:
            J_best = J_line
            lam_idx_best = lam_idx
    print("J line search results: ",J_line, lam_idx, J_best, lam_idx_best)
    if lam_idx_best != 0 :
        print("u star is: ", u_star)
    return U1.at[tau_idx:tau_idx+lam_idx_best,:].set(u_star)


D_running_cost = grad(build_step_loss, argnums=0)
D_terminal_cost = grad(build_terminal_cost, argnums=0)
forward_sim = lambda x0, u:scan_np(step_forward, x0,u)
backward_sim = lambda rho0, v, u: scan_np(F_backward, rho0, np.concatenate([v[::-1], u[::-1]], axis=1))

if __name__ == "__main__":
    xml_path = "./unitree_a1/scene.xml"
    dt =0.01

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)


    # sys = mjcf.loads("scene_brax.xml", asset_path="/home/yilangliu/Research/CI_MUJOCO_JAX/mujoco_menagerie/unitree_a1")
    # mjx_model = mjx.device_put(mj_model)
    # mjx_model = mjx.put_model(mj_model)
    # mjx_data = mjx.put_data(mj_model, mj_data)

    # start configuring the model
    FL_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'FL')
    FR_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'FR')
    RL_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'RL')
    RR_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'RR')
    FL_geom = mj_model.geom("FL")
    FR_geom = mj_model.geom("FR")
    RL_geom = mj_model.geom("RL")
    RR_geom = mj_model.geom("RR")
    FL_calf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'FL_calf')
    FR_calf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'FR_calf')
    RL_calf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'RL_calf')
    RR_calf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'RR_calf')


    # Jacobian matrices
    jac_com = np.zeros((3, mj_model.nv))
    jac_foot_FL = np.zeros((3, mj_model.nv))
    jac_foot_FR = np.zeros((3, mj_model.nv))
    jac_foot_RL = np.zeros((3, mj_model.nv))
    jac_foot_RR = np.zeros((3, mj_model.nv))

    # Previous jacobian matrices
    jac_foot_FL_prev = np.zeros((3, mj_model.nv))
    jac_foot_FR_prev = np.zeros((3, mj_model.nv))
    jac_foot_RL_prev = np.zeros((3, mj_model.nv))
    jac_foot_RR_prev = np.zeros((3, mj_model.nv))

    # Derivative of the jacobian matrices
    jac_foot_FL_dot = np.zeros((3, mj_model.nv))
    jac_foot_FR_dot = np.zeros((3, mj_model.nv))
    jac_foot_RL_dot = np.zeros((3, mj_model.nv))
    jac_foot_RR_dot = np.zeros((3, mj_model.nv))

    # Previous foot positions
    position_foot_FL_prev = np.zeros((3, ))
    position_foot_FR_prev = np.zeros((3, ))
    position_foot_RL_prev = np.zeros((3, ))
    position_foot_RR_prev = np.zeros((3, ))

    # Torque vectors
    tau_FL = np.zeros((mj_model.nv, 1))
    tau_FR = np.zeros((mj_model.nv, 1))
    tau_RL = np.zeros((mj_model.nv, 1))
    tau_RR = np.zeros((mj_model.nv, 1))


    print("nv is ", mj_model.nv)
    print("nu is ", mj_model.nu)
    print("nq is ", mj_model.nq)

    # starting simulation loop 
    reset(mj_model, mj_data)
    mj_model.opt.timestep = dt

    # step_forward(mj_model, mj_data, np.zeros(mj_model.nu))

    # position_foot_FL = mj_data.geom_xpos[FL_id].reshape(3,1)
    # position_foot_FR = mj_data.geom_xpos[FR_id].reshape(3,1)
    # position_foot_RL = mj_data.geom_xpos[RL_id].reshape(3,1)
    # position_foot_RR = mj_data.geom_xpos[RR_id].reshape(3,1)



    # motor_ang, base_pos = jnp.asarray(mj_data.qpos)[7:], jnp.asarray(mj_data.qpos)[:3]


    # print(jax_fwd(motor_ang))

    # print(np.concatenate((position_foot_FL,position_foot_FR, position_foot_RL, position_foot_RR)))

    # foot_pos = jax_fwd(motor_ang)

    # print("FL: {}, FR {}, RL {}, RR{}".format(FL_id, FR_id, RL_id, RR_id))
    # print("FL: {}, FR {}, RL {}, RR{}".format(base_pos[2]+ foot_pos[2], base_pos[2]+ foot_pos[5], base_pos[2]+ foot_pos[8], base_pos[2]+ foot_pos[11]))

    # visualize the robot Optional
    # setup SAC configuration:

    x_0 = jnp.concatenate((mj_data.qpos, mj_data.qvel),axis=0)

    prediction_horizon = 0.05
    T = int(prediction_horizon/dt)
    ctrl = jnp.zeros((T,mj_model.nu))

    ctrl = SAC(x_0, ctrl)

    ## test jacobian
    # gravity_support = jnp.array([0,0.9,-1.8]).reshape(3,1)
    # print("RL angle is: ",x_0[7:10])
    # print(analytical_leg_jacobian(x_0[7:10],-1).T@gravity_support)
    # xxx



    # Run some tests for the cost function

    # print(state_quaternion_to_euler(x_0))
    # print(build_step_loss(state_quaternion_to_euler(x_0).at[7].set(0.4), ctrl[0,:]))5
    # print(build_terminal_cost(state_quaternion_to_euler(x_0)))
    with mujoco.viewer.launch_passive(mj_model, mj_data,  show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(mj_model, viewer.cam)
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        i = 0

        while viewer.is_running():
            step_start = time.time()

            # x_0 = jnp.concatenate((mj_data.qpos, mj_data.qvel),axis=0)    
            # print("start running.......")
            time_sac = time.time()
            
            ctrl =  SAC(x_0, ctrl) # jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
            # FR FL RR RL
            # ctrl = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
            print("one iteration for sac is ", time.time()-time_sac)
            print("ctrl is ", ctrl[0,:])

            x_0 = step_forward(x_0, ctrl[0,:])[0]
            # x_0 = step_forward(x_0, ctrl)[0]
            
            # print("finished one iteration")
            
            # x_0 = step_forward(x_0, ctrl)[0]

            # print("motor angle: ", jnp.asarray(mj_data.qpos))
            # motor_ang, base_pos = jnp.asarray(mj_data.qpos)[7:], jnp.asarray(mj_data.qpos)[:3]
            # foot_pos_FL,foot_pos_FR,foot_pos_RL,foot_pos_RR = get_phi_analytical(jnp.asarray(mj_data.qpos))
            # print("FL_bullet: {}, FR_bullet {}, RL_bullet {}, RR_bullet{}".format(base_pos[2]+ foot_pos[2], base_pos[2]+ foot_pos[5], base_pos[2]+ foot_pos[8], base_pos[2]+ foot_pos[11]))
            
            # print("FL_bullet: {}, FR_bullet {}, RL_bullet {}, RR_bullet{}".format(foot_pos_FL[2], foot_pos_FR[2] , foot_pos_RL[2], foot_pos_RR[2]))
            
            # position_foot_FL = mj_data.geom_xpos[FL_id].reshape(3,1)
            # position_foot_FR = mj_data.geom_xpos[FR_id].reshape(3,1)
            # position_foot_RL = mj_data.geom_xpos[RL_id].reshape(3,1)
            # position_foot_RR = mj_data.geom_xpos[RR_id].reshape(3,1)
            # print("FL_mujoco: {}, FR_mujoco {}, RL_mujoco {}, RR_mujoco{}".format(position_foot_FL[2], position_foot_FR[2], position_foot_RL[2], position_foot_RR[2]))

            # mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            # time.sleep(0.01)

            i += 1
