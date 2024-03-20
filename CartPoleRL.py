import numpy as np
import os
from isaacgym import gymapi, gymutil, gymtorch
import torch
import random
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def __main__():
    
    sim, gym, device, args = setup_isaac_gym(gpu_pipeline=False, use_gpu=True)
    add_ground_to_simulation(sim, gym)
    cartpole_asset, asset_options, num_dof = make_cartpole_asset(sim, gym)
    environments,actor_handles, num_env= setup_environments_and_actors(sim, gym, args, device, num_dof, cartpole_asset, asset_options)
    viewer = create_viewer(sim, gym,environments[0])
    dof_state_buffer,pos_buffer,vel_buffer = make_buffers(gym,sim,num_dof,num_env, device)
    # Simulation loop
    while not gym.query_viewer_has_closed(viewer):
#        torch.cuda.empty_cache()
        gym.refresh_dof_state_tensor(sim)
        apply_actions_vec(gym,sim,pos_buffer,vel_buffer,dof_state_buffer,device)
        
        # apply_actions(gym,environments, actor_handles,pos_buffer,vel_buffer)
        
        simulation_step(sim, gym)
        
        interface_step(sim,gym,viewer)




############################################################################################################
    #Functions
############################################################################################################
def make_buffers(gym,sim,num_dof,num_env, device):
    dof_state_buffer_tensor  = gym.acquire_dof_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    dof_state_buffer = gymtorch.wrap_tensor(dof_state_buffer_tensor)
    pos_view = dof_state_buffer.view(num_env, num_dof, 2)[..., 0]
    vel_view = dof_state_buffer.view(num_env, num_dof, 2)[..., 1]

    return dof_state_buffer, pos_view, vel_view

def get_simulation_state(sim, gym):
    sim_state = gym.get_sim_state(sim)
    observation = gym.get_sim_observation(sim)
    reward = gym.get_sim_reward(sim)
    return sim_state,observation,reward

def apply_actions_vec(gym,sim,pos_buffer,vel_buffer,dof_state_buffer,device,num_envs=100,num_dof=2):
    # actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
    #     actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
    #     forces = gymtorch.unwrap_tensor(actions_tensor)
    #     self.gym.set_dof_actuation_force_tensor(self.sim, forces)
    # move variables to device
    state=dof_state_buffer
    noise = torch.randn(num_envs, num_dof, device=device)*10
    noise[:,1] = 0
    controller_tensoer=torch.zeros(num_envs,num_dof, device=device, dtype=torch.float)
    controller_tensoer[:,1] =  (pos_buffer[:,1]*-0.1+vel_buffer[:,1]*-0.01)*1000
    action_tensor = noise +controller_tensoer
    #action_tensor[:,0] =  (pos_buffer[:,0]*-0.1+vel_buffer[:,0]*-0.01)*1000
    force_vector = gymtorch.unwrap_tensor(action_tensor)
    gym.set_dof_actuation_force_tensor(sim, force_vector)

def apply_actions(gym,environments, actor_handles):
    actor_efforts = np.zeros((len(environments)))
    for i, env in enumerate(environments):
        # get dof positions and velocities
        actor_handle = actor_handles[i]
        dof_handle = gym.get_actor_dof_handle(env,actor_handle,1)
        dof_pos = gym.get_dof_position(env, dof_handle)
        dof_vel = gym.get_dof_velocity(env, dof_handle)
        random_effort = random.uniform(-4,4)*0.1
        actor_efforts[i] += random_effort
        effort = actor_efforts[i]*1 +dof_vel*-0.0+dof_pos*-5
        efforts = np.repeat([0,effort],1).astype('f')
        gym.apply_actor_dof_efforts(env, actor_handle, efforts)

def interface_step(sim,gym,viewer):
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.query_viewer_action_events(viewer)
    gym.sync_frame_time(sim)

def clean_up(gym, sim, viewer):
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    
def simulation_step(sim, gym):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    


def setup_environments_and_actors(sim, gym, args, device, num_dof, cartpole_asset, asset_options):
    # set up the env grid
    num_envs = 100
    envs_per_row = 10
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

        cartpole_len = random.uniform(1.0, 2.5)
        cartpole_len = 1.0
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(1.0, cartpole_len, 1.0)

        actor_handle = gym.create_actor(env, cartpole_asset, pose, "MyActor", i, 1,0)
        
        actor_handles.append(actor_handle)
        
        #other
        # configure the joints for effort control mode (once)
        props = gym.get_actor_dof_properties(env, actor_handle)
        props["driveMode"][0]=(gymapi.DOF_MODE_EFFORT)
        props["driveMode"][1]=(gymapi.DOF_MODE_EFFORT)
        props["stiffness"].fill(0.0)
        props["damping"].fill(0.0)
        gym.set_actor_dof_properties(env, actor_handle, props)

    gym.prepare_sim(sim)
    return envs, actor_handles, num_envs


def add_ground_to_simulation(sim, gym):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0
    gym.add_ground(sim, plane_params)

def create_viewer(sim, gym,env):
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    # set directon of the camera
    gym.viewer_camera_look_at(viewer, env,gymapi.Vec3(-3, -3, 3), gymapi.Vec3(0, 0, 1))
    if viewer is None:
        raise ValueError("Failed to create viewer")
    return viewer
    
def make_cartpole_asset(sim, gym):
    asset_root = os.path.dirname(os.path.abspath(__file__))  # Adjust this to the correct folder
    cartpole_urdf_file = "cartpole.urdf"  # Ensure this is the correct filename
    cartpole_urdf_path = os.path.join(asset_root, cartpole_urdf_file)

    # Make sure the URDF file exists at the specified path
    if not os.path.exists(cartpole_urdf_path):
        raise FileNotFoundError(f"Could not find URDF file at {cartpole_urdf_path}")
    
    # Set asset options
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True

    # Load the CartPole URDF
    cartpole_asset = gym.load_asset(sim, asset_root,cartpole_urdf_file, asset_options)
    num_dof=gym.get_asset_dof_count(cartpole_asset)
    return cartpole_asset, asset_options, num_dof

def setup_isaac_gym(gpu_pipeline, use_gpu):
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu and gpu_pipeline else "cpu") 
    
    
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
    sim_params.use_gpu_pipeline = torch.cuda.is_available() and gpu_pipeline and use_gpu
    sim_params.physx.use_gpu = torch.cuda.is_available() and use_gpu
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
   
    if sim is None:
        raise ValueError("Failed to create sim")

    return sim, gym, device, args


if __name__ == "__main__":
    __main__()