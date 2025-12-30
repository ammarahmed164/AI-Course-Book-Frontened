# Isaac Sim Navigation Setup
# This script demonstrates setting up navigation in Isaac Sim

import omni
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.viewports import set_active_viewport_camera
import numpy as np

def setup_navigation_environment():
    """Setup environment for navigation simulation."""
    # Initialize the world
    my_world = World(stage_units_in_meters=1.0)
    
    # Add ground plane
    from omni.isaac.core.objects import GroundPlane
    my_world.scene.add(GroundPlane(prim_path="/World/Ground", size=100.0))
    
    # Add robot (using a simple differential drive robot)
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("Could not find Isaac Sim assets. Please enable Isaac Sim USD asset extension")
        return None
    
    # Add a simple cuboid robot
    my_world.scene.add(
        Robot(
            prim_path="/World/Robot",
            name="robot",
            translation=np.array([0.0, 0.0, 0.1]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
    )
    
    # Add some obstacles
    from omni.isaac.core.objects import DynamicCuboid
    obstacle1 = my_world.scene.add(
        DynamicCuboid(
            prim_path="/World/Obstacle1",
            name="obstacle1",
            position=np.array([2.0, 0.0, 0.5]),
            size=0.5,
            color=np.array([0.8, 0.1, 0.1])
        )
    )
    
    # Add dome light
    from omni.isaac.core.utils.lighting import add_dome_light
    add_dome_light("/World/DomeLight", intensity=1000)
    
    return my_world

def main():
    # Setup the environment
    world = setup_navigation_environment()
    
    if world is None:
        return
    
    # Reset the world to apply changes
    world.reset()
    
    # Run for a few steps to see the environment
    for i in range(100):
        world.step(render=True)
    
    print("Navigation environment created successfully!")

if __name__ == "__main__":
    main()