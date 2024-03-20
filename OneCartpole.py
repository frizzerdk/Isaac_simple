import numpy as np
import os
from isaacgym import gymapi, gymutil, gymtorch
import torch
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Isaac Gym
gym = gymapi.acquire_gym()

# Parse arguments (for standalone applications)
args = gymutil.parse_arguments(description="CartPole Visualization in Isaac Gym")

# Set physics engine and device
sim_params = gymapi.SimParams()
# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

if sim is None:
    raise ValueError("Failed to create sim")

# Create a viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError("Failed to create viewer")

# Set the asset root to the directory containing your URDF file
asset_root = os.path.dirname(os.path.abspath(__file__))  # Adjust this to the correct folder
cartpole_urdf_file = "cartpole.urdf"  # Ensure this is the correct filename
cartpole_urdf_path = os.path.join(asset_root, cartpole_urdf_file)

# Make sure the URDF file exists at the specified path
if not os.path.exists(cartpole_urdf_path):
    raise FileNotFoundError(f"Could not find URDF file at {cartpole_urdf_path}")

# Load the CartPole URDF
asset_options = gymapi.AssetOptions()
# fiix to environment
asset_options.fix_base_link = True
cartpole_asset = gym.load_asset(sim, asset_root, cartpole_urdf_file, asset_options)
num_dof=gym.get_asset_dof_count(cartpole_asset)

spacing = 2.0

lower = gymapi.Vec3(-spacing, 0, spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
# Create the environment
# set up the env grid
num_envs = 1000
envs_per_row = 30
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    height = random.uniform(1.0, 2.5)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, height, 1.0)

    actor_handle = gym.create_actor(env, cartpole_asset, pose, "MyActor", i, 1,0)
    
    actor_handles.append(actor_handle)
    
    #other
    # configure the joints for effort control mode (once)
    props = gym.get_actor_dof_properties(env, actor_handle)
    props["driveMode"][0]=(gymapi.DOF_MODE_NONE)
    props["driveMode"][1]=(gymapi.DOF_MODE_POS)
    #props["stiffness"].fill(1000.0)
    #props["damping"].fill(200.0)
    gym.set_actor_dof_properties(env, actor_handle, props)
# Main simulation loop
while not gym.query_viewer_has_closed(viewer):
     
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # actions_tensor = torch.zeros(num_envs * num_dof, device='cpu', dtype=torch.float)
    # actions_tensor[::num_dof] = random
    # forces = gymtorch.unwrap_tensor(actions_tensor)
    # gym.set_dof_actuation_force_tensor(sim, forces)
    

    # Apply actions to each cart
    for i, env in enumerate(envs):
        actor_handle = actor_handles[i]
        random_target = random.uniform(-4,4)
        targets = np.repeat([0,random_target],1).astype('f')
        gym.set_actor_dof_position_targets(env, actor_handle, targets)
        

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.query_viewer_action_events(viewer)
    gym.sync_frame_time(sim)


# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


