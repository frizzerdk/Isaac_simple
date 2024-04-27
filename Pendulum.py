import numpy as np
import os
import cv2
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
from SimpleRlAgent import SimpleRlAgent
from PID_agent import PIDAgent
import torch
import random
import time
import wandb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def __main__():
    
    print(torch.cuda.is_available())
    
    sim, gym, device, pipeline_device, args= setup_isaac_gym(gpu_pipeline=False, use_gpu=True)
    add_ground_to_simulation(sim, gym)
    pendulum_asset, asset_options, num_dof = make_pendulum_asset(sim, gym)
    environments, _ , num_env= setup_environments_and_actors(sim, gym, args, device, num_dof, pendulum_asset, asset_options)
    viewer = create_viewer(sim, gym,environments[0])
    dof_state_buffer,state_buffer,dof_full_view = make_buffers(gym,sim,num_dof,num_env,pipeline_device)
    progress_buf , reset_buf , truncate_buf= torch.zeros(num_env,1,device=device),torch.zeros(num_env,1,device=device),torch.zeros(num_env,1,device=device)

    agent = SimpleRlAgent(num_dof*2, 1, torch.tensor([[-300,300]]),device=device)    
    #agent = PIDAgent(num_dof*2, 1, torch.tensor([[-300,300]]),device=device)
    counter = 0

    # Simulation loop
    while not gym.query_viewer_has_closed(viewer):
        time_start = time.time()
        gym.refresh_dof_state_tensor(sim)
        states= get_simulation_states(state_buffer,device) # Tensor n_envs x n_dof*2
        observations = get_state_observation(states) # Tensor n_envs x n_dof*2
        actions=get_agent_actions(observations,agent) # Tensor n_envs x n_action_dim
        effort_vector = action2forceVec(actions,num_dof,num_env,device) # Tensor n_envs x n_dof
        #effort_vector=base_controller(pos_buffer,vel_buffer,device,num_env,num_dof) # Tensor n_envs x n_dof
        
        s1,a1,o1 = states,actions,observations
        
        apply_actions_vec(gym,sim,effort_vector,pipeline_device,noise_level=10)
        simulation_step(sim, gym, progress_buf)
        reset_failed_envs(sim,gym,state_buffer,dof_state_buffer,progress_buf,reset_buf,pipeline_device)
        truncate_envs(sim, gym, state_buffer,dof_state_buffer,progress_buf,truncate_buf,pipeline_device)

        s2 = get_simulation_states(state_buffer,device)
        o2 = get_state_observation(s2)
        r1 = calculate_rewards(s1,a1,s2,num_env,reset_buf)
        agent.add_experience(o1,a1,r1,o2,reset_buf)
        reset_buf[:] = 0
        truncate_buf[:] = 0
        for i in range(2):
            agent.learn()
        counter += 1
        if counter > 60:
            counter = 0
            show_value_function(device,value_function=lambda states : compute_value_function_agent(states,agent,device),name='Agent Value Function',window_position=(850,1000))
            show_value_function(device,value_function=lambda states : compute_policy_function_agent(states,agent,device),name='Agent Policy Function',window_position=(450,1000))
            show_value_function(device,value_function=lambda states : compute_q_function_agent(states,agent,device),name='Agent Q Function',window_position=(50,1000))
                                
        
        interface_step(sim,gym,viewer)
        time_end = time.time()
        #print without making new line
        #print(f"\rTime taken: {time_end-time_start}",end="")
        




############################################################################################################
    #Functions
############################################################################################################
def truncate_envs(sim, gym, states,dof_state_buffer,progress_buf,truncate_buf,pipeline_device):
    # Truncate envs that have reached the max number of steps
    max_steps = 500
    truncate_buf[progress_buf >= max_steps] = 1
    truncate_indices = torch.where(truncate_buf)[0].tolist()
    if len(truncate_indices) > 0:
        print(f"Truncating {len(truncate_indices)} envs")
        reset_idx(sim,gym,torch.tensor(truncate_indices,device=pipeline_device),dof_state_buffer,states,pipeline_device)
        progress_buf[truncate_indices] = 0
        
def reset_failed_envs(sim,gym,states,dof_state_buffer,progress_buf,reset_buf,pipeline_device):
    state_lims = torch.tensor([[-3,3],[-100,100]],device=pipeline_device)
    failed_envs = torch.any(torch.logical_or(states<state_lims[:,0],states>state_lims[:,1]),dim=1)
    failed_indices = torch.where(failed_envs)[0].tolist()
    if len(failed_indices)>0:
        reset_idx(sim,gym,torch.tensor(failed_indices,device=pipeline_device),dof_state_buffer,states,pipeline_device)
        #print(f"Resetting {len(failed_indices)} failed envs")
        reset_buf[failed_indices] = 1
        progress_buf[failed_indices] = 0
                          
def reset_idx(sim,gym, env_ids,dof_state_buffer,states,device):
    #device ='cpu'
    env_ids = env_ids.to(device)
    states = states.to(device)
    state_init_dist = torch.tensor([[-0.5, 0.5], [-0.01, 0.01]], device=device) # [min, max] for each state dimension
    reset_states = torch.rand((len(env_ids), state_init_dist.size(0)), device=device) * (state_init_dist[:, 1] - state_init_dist[:, 0]) + state_init_dist[:, 0]
    #reset_states = torch.ones_like(reset_states)
    states[env_ids] = reset_states

    env_ids_int32 = env_ids.to(dtype=torch.int32).to(device)
    gym.set_dof_state_tensor_indexed(sim,
                                    gymtorch.unwrap_tensor(dof_state_buffer),
                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

def compute_value_function_dummy(states,counter=[0]):
    # Example value function: cos(p1) * cos(v1) * p2 * v2
    # States are [p1, v1, p2, v2]
    counter[0] += 1
    p1, v1, p2, v2 = states[:, 0]+counter[0]*0.0001, states[:, 1], states[:, 2], states[:, 3]
    values = torch.cos(p1) * torch.cos(v1) * p2 * v2
    return values

def compute_q_function_agent(states,agent,device):
    states = states.clone().detach()*torch.tensor([[300,0,3,20]],device=device)
    actions = states[:,0].clone().detach().unsqueeze(1)
    states = states.detach()
    states[:,0] = 0
    states = states[:,2:4]
    values= agent.estimate_Q(states,actions)-agent.estimate_Q(states,actions*0)
    return values
def compute_value_function_agent(states,agent,device):
    states = states.clone().detach()*torch.tensor([[0,0,3,20]],device=device)
    states = states[:,2:4]
    actions = agent.select_actions(states,exploration=False)
    values= agent.estimate_Q(states,actions)
    return values
def compute_policy_function_agent(states,agent,device):
    states = states.clone().detach()*torch.tensor([[0,0,3,20]],device=device)
    states = states[:,2:4]
    actions = agent.select_actions(states,exploration=False)
    return actions


def render_tensor_with_cv(tensor, scale_factor=1, name='Value Function', window_position=(100, 100)):
    min_val, max_val = tensor.min().item(), tensor.max().item()
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor * 255.0
    tensor = tensor.cpu().detach().numpy().astype(np.uint8)
    color_mapped_tensor = cv2.applyColorMap(tensor, cv2.COLORMAP_JET)
    
    # Resize the image
    height, width = color_mapped_tensor.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))  # New dimensions
    resized_image = cv2.resize(color_mapped_tensor, new_dimensions, interpolation=cv2.INTER_AREA)
    
    # Add descriptions (or any text) to the image
    # Note: This will overlay the text on your image. Adjust position as needed.
    # min value
    cv2.putText(resized_image, 'Min: {:.3f}'.format(min_val), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(resized_image, 'Max: {:.3f}'.format(max_val), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the window at specified position before showing the image
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, window_position[0], window_position[1])
    
    cv2.imshow(name, resized_image)
    cv2.waitKey(1)

    #cv2.destroyAllWindows()

def show_value_function(device, value_function=None, limits=None, index_map=None, name='Value Function',window_position = None):
    window_position = (100,100)if window_position is None else window_position
    limits = torch.tensor([[-1, 1], [-1, 1], [-1, 1], [-1, 1]], device=device) if limits is None else limits
    value_function = compute_value_function_dummy if value_function is None else value_function
    state_map = [2, 3, 0, 1] if index_map is None else index_map
    
    img_grid_dim = 3
    img_dim = 50
    value_function_grid = torch.zeros((img_grid_dim * img_dim, img_grid_dim * img_dim), device=device)

    for i in range(img_grid_dim):
        for j in range(img_grid_dim):
            # Map states based on the index_map
            mapped_p2 = torch.linspace(limits[state_map[2], 0], limits[state_map[2], 1], img_grid_dim, device=device)[i]
            mapped_v2 = torch.linspace(limits[state_map[3], 0], limits[state_map[3], 1], img_grid_dim, device=device)[j]
            
            mapped_p1_space = torch.linspace(limits[state_map[0], 0], limits[state_map[0], 1], img_dim, device=device)
            mapped_v1_space = torch.linspace(limits[state_map[1], 0], limits[state_map[1], 1], img_dim, device=device)
            
            mapped_p1_grid, mapped_v1_grid = torch.meshgrid(mapped_p1_space, mapped_v1_space, indexing='ij')

            # Rearrange states according to the state_map for value function computation
            states = torch.stack([
                mapped_v1_grid.flatten(), 
                mapped_p1_grid.flatten(), 
                mapped_v2.repeat(img_dim * img_dim), 
                mapped_p2.repeat(img_dim * img_dim)
            ], dim=1)[:, state_map]  # Reorder columns based on state_map
            
            values = value_function(states).view(img_dim, img_dim)
            value_function_grid[i * img_dim:(i + 1) * img_dim, j * img_dim:(j + 1) * img_dim] = values

    #print('name: {}, min: {:.3f}, max: {:.3f}'.format(name, value_function_grid.min().item(), value_function_grid.max().item()))

    render_tensor_with_cv(value_function_grid,name=name,window_position=window_position)


def calculate_rewards(s1,a1,s2,num_env,reset_buf):
    # first dimension is env index
    # negative reward for distance from 0 postion and speed accross all states
    p1, v1= s2[:, 0], s2[:, 1]
    #state_rewards =s2[:,0].abs()*-0+s2[:,1].abs()*-0.000+s2[:,2].abs()*s2[:,2].abs()*2+s2[:,3].abs()*-0.0001
    state_rewards = p1.pow(2)*-1 - p1.abs()\
                    +v1.abs()*-0.0001\

    action_rewards = a1[:,0].pow(2)*-0.00001
    reset_rewards = reset_buf.squeeze()*-10
    total_rewards = state_rewards+action_rewards+reset_rewards
    # print min and max rewards
    #print(f"Min reward: {total_rewards.min().item()}, Max reward: {total_rewards.max().item()}")
    return total_rewards

def action2forceVec(actions,num_dof,num_env,device):
    force_vector = torch.zeros(num_env,num_dof,device=device,dtype=torch.float)
    force_vector[:,0] = actions.squeeze()
    return force_vector

def get_state_observation(states):
    return states

def get_agent_actions(observations:torch.tensor,agent)->torch.tensor: # Tensor n_envs x n_obs_dim
    
    observations = observations.to(agent.device)
    exploration_mask = torch.ones(observations.shape[0],1,device=agent.device)
    exploration_mask[0] = 0
    actions = agent.select_actions(observations,exploration=True,exploration_mask = exploration_mask) # Tensor n_envs x n_action_dim
    return actions


def make_buffers(gym,sim,num_dof,num_env, device):
    # Make buffers on the pipeline device
    dof_state_buffer_raw  = gym.acquire_dof_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    dof_state_buffer = gymtorch.wrap_tensor(dof_state_buffer_raw)
    dof_state_buffer.to(device) # Tensor n_envs * n_dof x 2
    state_buffer = dof_state_buffer.view(num_env, num_dof*2) # Tensor n_envs x n_dof*2
    dof_full_view = dof_state_buffer.view(num_env, num_dof, 2) # Tensor n_envs x n_dof x 2

    return dof_state_buffer,state_buffer,dof_full_view

def get_simulation_states(state_buffer,device):
    # Get simulation states on agent device
    sim_state = state_buffer.to(device)
    return sim_state

def base_controller(pos_buffer,vel_buffer,device,num_envs=100,num_dof=2)->torch.tensor:
    controller_tensoer=torch.zeros(num_envs,num_dof, device=device, dtype=torch.float)
    controller_tensoer[:,1] =  (pos_buffer[:,1]*-0.1+vel_buffer[:,1]*-0.00)*100
    action_tensor = controller_tensoer
    return action_tensor.to(device)

def apply_actions_vec(gym,sim,action_tensor,pipeline_device,noise_level=0):
    #noise with same dimension as action_tensor
    action_tensor = action_tensor.to(pipeline_device)
    noise = torch.randn_like(action_tensor,device=pipeline_device)*noise_level
    total_force = action_tensor+noise
    force_vector = gymtorch.unwrap_tensor(total_force) # to gymtorchtensor: will be on pipeline_device
    
    gym.set_dof_actuation_force_tensor(sim, force_vector)

def apply_actions(gym,environments, actor_handles,actions,noise_level=100,noise_mask=[1,0]):
    actor_efforts = actions
    noise_mask_tensor = torch.tensor(noise_mask, dtype=torch.float32, device=actions.device)
    
    for i, env in enumerate(environments):
        # get dof positions and velocities
        actor_handle = actor_handles[i]
        dof_handle = gym.get_actor_dof_handle(env,actor_handle,1)
        random_effort = random.uniform(-1,1)*noise_level*noise_mask_tensor
        actor_efforts[i,:] += random_effort
        effort = actor_efforts[i]*1  # Tensor 2x1
        #efforts = effort
        gym.apply_actor_dof_efforts(env, actor_handle, effort)
    return actor_efforts

def interface_step(sim,gym,viewer):
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.query_viewer_action_events(viewer)
    gym.sync_frame_time(sim)

def clean_up(gym, sim, viewer):
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    
def simulation_step(sim, gym, progress_buf):
    progress_buf[:] = progress_buf[:] + 1
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    


def setup_environments_and_actors(sim, gym, args, device, num_dof, cartpole_asset, asset_options):
    # set up the env grid
    num_envs = 16
    envs_per_row = num_envs
    env_spacing = 0.2
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
    
def make_pendulum_asset(sim, gym):
    asset_root = os.path.dirname(os.path.abspath(__file__))  # Adjust this to the correct folder
    pendulum_urdf_file = "pendulum.urdf"  # Ensure this is the correct filename
    pendulum_urdf_path = os.path.join(asset_root, pendulum_urdf_file)

    # Make sure the URDF file exists at the specified path
    if not os.path.exists(pendulum_urdf_path):
        raise FileNotFoundError(f"Could not find URDF file at {pendulum_urdf_path}")
    
    # Set asset options
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True

    # Load the CartPole URDF
    pendulum_asset = gym.load_asset(sim, asset_root,pendulum_urdf_file, asset_options)
    num_dof=gym.get_asset_dof_count(pendulum_asset)
    return pendulum_asset, asset_options, num_dof

def setup_isaac_gym(gpu_pipeline, use_gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu") 
    pipeline_device = torch.device("cuda" if torch.cuda.is_available() and gpu_pipeline and use_gpu else "cpu")
    
    # Initialize Isaac Gym
    gym = gymapi.acquire_gym()

    # Parse arguments (for standalone applications)
    args = gymutil.parse_arguments(description="Pendulum Visualization in Isaac Gym")
    

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

    return sim, gym, device, pipeline_device, args

def render_tensor(tensor):
    # Step 1: Convert the tensor to a NumPy array
    # Ensure tensor is on CPU and convert to numpy
    np_array = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    
    # Step 2: Normalize the tensor to 0-255
    # Assuming the tensor's values range from min to max, normalize to 0-255
    normalized_array = cv2.normalize(np_array, None, alpha=0, beta=255, 
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Step 3: Apply a colormap
    # cv2.applyColorMap supports various colormaps like COLORMAP_JET, COLORMAP_HOT, etc.
    heatmap = cv2.applyColorMap(normalized_array, cv2.COLORMAP_JET)
    
    # Display the heatmap
    cv2.imshow('Heatmap', heatmap)
    cv2.waitKey(1)  # Use a small delay so the window is responsive; adjust as necessary
if __name__ == "__main__":
    __main__()