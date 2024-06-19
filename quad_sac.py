import numpy as np
import jax.numpy as jnp
from jax import jit
import jax
from jax import random
import mujoco
import mujoco.viewer
import time 

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



if __name__ == "__main__":
    xml_path = "./unitree_a1/scene.xml"
    dt =0.01

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    print("nv is ", mj_model.nv)
    print("nu is ", mj_model.nu)
    print("nq is ", mj_model.nq)

    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    mujoco.mj_step(mj_model, mj_data)
    print(mj_data.qpos)