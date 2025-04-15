#!/usr/bin/env python3

# Copyright 2024 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Simple mission for a swarm of drones."""

__authors__ = 'Rafael Perez-Segui, Miguel Fernandez-Cortizas'
__copyright__ = 'Copyright (c) 2024 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'

# Configuration macros
STAGE2_METHOD = "lineY"  # Options: "sequential" or "lineY"

import argparse
import sys
from typing import List, Optional, Dict, Tuple
from math import radians, cos, sin, pi
from itertools import cycle, islice
import random
import rclpy
import yaml
import numpy as np
import math
import time
from as2_msgs.msg import YawMode
from as2_msgs.msg import BehaviorStatus
from as2_python_api.drone_interface import DroneInterface
from as2_python_api.behavior_actions.behavior_handler import BehaviorHandler

from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, Point

# Constants for formation and movement
LAYER_OFFSET = 0.5  # Vertical separation between drones during transitions
FORMATION_DISTANCE = 1.0  # Distance between drones in formation
CIRCLE_POINTS = 36  # Number of points to discretize the circle
FLIGHT_SPEED = 0.5  # Speed for drone movement (m/s)
FORMATION_CHANGE_INTERVAL = 20  # Number of waypoints before changing formation

def read_config_from_yaml(file_path: str) -> dict:
    """Read config from yaml file"""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


class FormationManager:
    """Manages different drone formations"""

    @staticmethod
    def line_formation(num_drones: int, distance: float = FORMATION_DISTANCE) -> List[List[float]]:
        """Generate a line formation along X-axis with specified number of drones
        
        Args:
            num_drones: Number of drones in the formation
            distance: Distance between adjacent drones
            
        Returns:
            List of [x, y] offsets for each drone relative to the leader
        """
        return FormationManager.lineX_formation(num_drones, distance)

    @staticmethod
    def lineX_formation(num_drones: int, distance: float = FORMATION_DISTANCE) -> List[List[float]]:
        """Generate a line formation along X-axis with specified number of drones
        
        Args:
            num_drones: Number of drones in the formation
            distance: Distance between adjacent drones
            
        Returns:
            List of [x, y] offsets for each drone relative to the leader
        """
        offsets = []
        
        if num_drones == 1:
            return [[0.0, 0.0]]
            
        # Leader (drone0) is at the center of the line
        offsets.append([0.0, 0.0])  # Leader at center
        
        # Calculate positions for followers
        left_side = []
        right_side = []
        
        for i in range(1, (num_drones + 1) // 2):
            # Drones to the left of leader
            left_side.append([-i * distance, 0.0])
            
        for i in range(1, num_drones // 2 + 1):
            # Drones to the right of leader
            right_side.append([i * distance, 0.0])
            
        # Combine all positions with leader at index 0
        offsets.extend(left_side[::-1])  # Reverse left side to maintain order
        offsets.extend(right_side)
            
        return offsets

    @staticmethod
    def lineY_formation(num_drones: int, distance: float = FORMATION_DISTANCE) -> List[List[float]]:
        """Generate a line formation along Y-axis with specified number of drones
        
        Args:
            num_drones: Number of drones in the formation
            distance: Distance between adjacent drones
            
        Returns:
            List of [x, y] offsets for each drone relative to the leader
        """
        offsets = []
        
        if num_drones == 1:
            return [[0.0, 0.0]]
            
        # Leader (drone0) is at the front of the line
        offsets.append([0.0, 0.0])  # Leader at front
        
        # Calculate positions for followers (all behind the leader)
        for i in range(1, num_drones):
            # Drones behind the leader along Y axis
            offsets.append([0.0, -i * distance])
            
        return offsets

    @staticmethod
    def v_formation(num_drones: int, distance: float = FORMATION_DISTANCE, angle: float = 30.0) -> List[List[float]]:
        """Generate a V formation with specified number of drones
        
        Args:
            num_drones: Number of drones in the formation
            distance: Distance between adjacent drones
            angle: Angle of the V formation in degrees
            
        Returns:
            List of [x, y] offsets for each drone relative to the leader
        """
        offsets = []
        
        if num_drones == 1:
            return [[0.0, 0.0]]
            
        # Leader is at the front of the V
        offsets.append([0.0, 0.0])
        
        # Convert angle to radians
        angle_rad = math.radians(angle)
        
        # Place followers in V formation
        left_side = []
        right_side = []
        
        for i in range(1, (num_drones + 1) // 2):
            # Left side of V
            x = -i * distance * math.cos(angle_rad)
            y = i * distance * math.sin(angle_rad)
            left_side.append([x, y])
            
        for i in range(1, num_drones // 2 + 1):
            # Right side of V
            x = -i * distance * math.cos(angle_rad)
            y = -i * distance * math.sin(angle_rad)
            right_side.append([x, y])
            
        # Combine all positions with leader at index 0
        offsets.extend(left_side)
        offsets.extend(right_side)
            
        return offsets

    @staticmethod
    def get_formation_offsets(formation_type: str, num_drones: int, distance: float = FORMATION_DISTANCE) -> List[List[float]]:
        """Get offsets for the specified formation type
        
        Args:
            formation_type: Type of formation ("line", "lineX", "lineY", "v", etc.)
            num_drones: Number of drones in the formation
            distance: Distance between drones in the formation
            
        Returns:
            List of [x, y] offsets for each drone relative to the leader
        """
        if formation_type.lower() == "line" or formation_type.lower() == "linex":
            return FormationManager.lineX_formation(num_drones, distance)
        elif formation_type.lower() == "liney":
            return FormationManager.lineY_formation(num_drones, distance)
        elif formation_type.lower() == "v":
            return FormationManager.v_formation(num_drones, distance)
        else:
            # Default to line formation if unknown type
            print(f"Unknown formation type: {formation_type}. Using line formation.")
            return FormationManager.lineX_formation(num_drones, distance)


class CircularTrajectoryGenerator:
    """Generates circular trajectory points"""
    
    @staticmethod
    def generate_circle_points(center: List[float], diameter: float, num_points: int = CIRCLE_POINTS) -> List[List[float]]:
        """Generate points along a circular trajectory
        
        Args:
            center: [x, y] coordinates of the circle center
            diameter: Diameter of the circle
            num_points: Number of points to generate
            
        Returns:
            List of [x, y, z] points along the circle
        """
        radius = diameter / 2.0
        points = []
        
        for i in range(num_points):
            angle = 2.0 * pi * i / num_points
            x = center[0] + radius * cos(angle)
            y = center[1] + radius * sin(angle)
            # Keep the same height as the center
            z = 1.5  # Fixed height for the circular trajectory
            
            points.append([x, y, z])
            
        return points


class Dancer(DroneInterface):
    """Drone Interface extended with path to perform and async behavior wait"""

    def __init__(self, namespace: str, verbose: bool = False,
                 use_sim_time: bool = False):
        super().__init__(namespace, verbose=verbose, use_sim_time=use_sim_time)

        self.__speed = FLIGHT_SPEED
        self.__yaw_mode = YawMode.PATH_FACING
        self.__yaw_angle = None
        self.__frame_id = "earth"

        self.current_behavior: Optional[BehaviorHandler] = None
        self.led_pub = self.create_publisher(ColorRGBA, f"/{namespace}/leds/control", 10)
        
        # Drone's layer for collision avoidance during transitions
        self.layer = 0
        self.is_leader = False

    def change_led_colour(self, colour):
        """Change the colours

        Args:
            colour (tuple): The LED RGB Colours (0-255)
        """
        msg = ColorRGBA()
        msg.r = colour[0]/255.0
        msg.g = colour[1]/255.0
        msg.b = colour[2]/255.0
        self.led_pub.publish(msg)

    def change_leds_random_colour(self):
        self.change_led_colour([random.randint(0, 255) for _ in range(3)])

    def do_behavior(self, beh, *args) -> None:
        """Start behavior and save current to check if finished or not"""
        self.current_behavior = getattr(self, beh)
        self.current_behavior(*args)

    def goal_reached(self) -> bool:
        """Check if current behavior has finished"""
        if not self.current_behavior:
            return False

        if self.current_behavior.status == BehaviorStatus.IDLE:
            return True
        return False

    def go_to_layered(self, target_position: List[float]) -> None:
        """Go to target position using layered approach to avoid collisions
        
        Args:
            target_position: [x, y, z] target position
        """
        # Get current position
        current_position = self.position
        
        # Calculate intermediate waypoints for layered approach
        layer_height = target_position[2] + LAYER_OFFSET * self.layer
        
        # Create a path with three waypoints for smooth movement
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.__frame_id
        
        # 1. Current position
        pose1 = PoseStamped()
        pose1.pose.position.x = current_position[0]
        pose1.pose.position.y = current_position[1]
        pose1.pose.position.z = current_position[2]
        
        # 2. Go up to layer height
        pose2 = PoseStamped()
        pose2.pose.position.x = current_position[0]
        pose2.pose.position.y = current_position[1]
        pose2.pose.position.z = layer_height
        
        # 3. Move horizontally to target position at layer height
        pose3 = PoseStamped()
        pose3.pose.position.x = target_position[0]
        pose3.pose.position.y = target_position[1]
        pose3.pose.position.z = layer_height
        
        # 4. Descend to target height
        pose4 = PoseStamped()
        pose4.pose.position.x = target_position[0]
        pose4.pose.position.y = target_position[1]
        pose4.pose.position.z = target_position[2]
        
        # Add all poses to the path
        path.poses = [pose1, pose2, pose3, pose4]
        
        # Follow the path with path_facing yaw mode for smooth movement
        self.do_behavior("follow_path", 
                        path, 
                        self.__speed,
                        YawMode.PATH_FACING, 
                        0.0, 
                        self.__frame_id, 
                        False)  # Don't wait - let the caller check for completion


class SwarmConductor:
    """Swarm Conductor for centralized control of multiple drones"""

    def __init__(self, drones_ns: List[str], verbose: bool = False,
                 use_sim_time: bool = False):
        self.drones: Dict[int, Dancer] = {}
        
        # Initialize drones
        for index, name in enumerate(drones_ns):
            self.drones[index] = Dancer(name, verbose, use_sim_time)
            self.drones[index].layer = index + 1  # Assign unique layer for collision avoidance
            
            # First drone is the leader
            if index == 0:
                self.drones[index].is_leader = True
                self.drones[index].change_led_colour((255, 0, 0))  # Leader is red
            else:
                self.drones[index].change_led_colour((0, 0, 255))  # Followers are blue
        
        self.leader = self.drones[0]
        self.current_formation = "line"  # Default formation
        self.formation_offsets = []
        self.circle_trajectory = []
        self.current_waypoint_index = 0

    def shutdown(self):
        """Shutdown all drones in swarm"""
        for drone in self.drones.values():
            drone.shutdown()

    def wait_all_drones(self):
        """Wait until all drones have reached their goals"""
        all_finished = False
        while not all_finished:
            all_finished = True
            for drone in self.drones.values():
                all_finished = all_finished and drone.goal_reached()
                if not all_finished:
                    break
            time.sleep(0.1)  # Small sleep to avoid busy waiting

    def get_ready(self) -> bool:
        """Arm and offboard for all drones in swarm"""
        success = True
        for drone in self.drones.values():
            # Arm
            success_arm = drone.arm()

            # Offboard
            success_offboard = drone.offboard()
            success = success and success_arm and success_offboard
        return success

    def takeoff(self):
        """Takeoff swarm and wait for all drones"""
        for drone in self.drones.values():
            drone.do_behavior("takeoff", 1.5, 0.7, False)
            drone.change_led_colour((0, 255, 0))  # Green during takeoff
        self.wait_all_drones()

    def land(self):
        """Land swarm and wait for all drones"""
        for drone in self.drones.values():
            drone.do_behavior("land", 0.4, False)
        self.wait_all_drones()

    def change_formation(self, new_formation: str):
        """Change the formation of the swarm
        
        Args:
            new_formation: New formation type
        """
        print(f"Changing formation to {new_formation}")
        
        # Check if we're already in the requested formation
        if hasattr(self, 'current_formation') and self.current_formation == new_formation:
            # Check if drones are already in the correct positions
            if self.is_formation_achieved():
                print(f"Formation {new_formation} is already achieved, skipping formation change")
                return
        
        # Get formation offsets for the new formation
        self.formation_offsets = FormationManager.get_formation_offsets(
            new_formation, len(self.drones), FORMATION_DISTANCE)
        
        # Apply the new formation at the current leader position
        self.apply_formation_at_current_position(use_layered=True)
        
        # Update current formation
        self.current_formation = new_formation
        print(f"Formation changed to {new_formation}")

    def apply_formation_at_current_position(self, use_layered: bool = False):
        """Apply the current formation at the current leader position
        
        Args:
            use_layered: Whether to use layered approach for collision avoidance
        """
        # Get leader position
        leader_pos = self.leader.position
        
        # Move swarm to maintain formation at current position
        self._move_swarm_to_position(leader_pos, use_layered=use_layered)

    def initialize_circle_trajectory(self, center: List[float], diameter: float):
        """Initialize the circular trajectory
        
        Args:
            center: [x, y] coordinates of the circle center
            diameter: Diameter of the circle
        """
        self.circle_trajectory = CircularTrajectoryGenerator.generate_circle_points(
            center, diameter)
        self.current_waypoint_index = 0
        
    def move_to_next_segment(self, segment_size: int = 10):
        """Move the swarm along a segment of the circular trajectory using follow_path
        
        Args:
            segment_size: Number of waypoints to include in one segment
        """
        if not self.circle_trajectory:
            print("Circle trajectory not initialized!")
            return
        
        # Create a segment of waypoints starting from current index
        segment_waypoints = []
        for i in range(segment_size):
            idx = (self.current_waypoint_index + i) % len(self.circle_trajectory)
            segment_waypoints.append(self.circle_trajectory[idx])
        
        # Create path for leader
        leader_path = Path()
        leader_path.header.stamp = self.leader.get_clock().now().to_msg()
        leader_path.header.frame_id = "earth"
        
        # Add waypoints to leader path
        for waypoint in segment_waypoints:
            pose = PoseStamped()
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            pose.pose.position.z = waypoint[2]
            leader_path.poses.append(pose)
        
        # Command leader to follow the path
        self.leader.do_behavior("follow_path", 
                               leader_path, 
                               FLIGHT_SPEED,
                               YawMode.PATH_FACING, 
                               0.0, 
                               "earth", 
                               False)
        
        # For each follower, create a path that maintains formation with leader
        for i, drone in self.drones.items():
            if i == 0:  # Skip leader
                continue
            
            # Get offset for this drone in the formation
            offset = self.formation_offsets[i]
            
            # Create path for follower
            follower_path = Path()
            follower_path.header.stamp = drone.get_clock().now().to_msg()
            follower_path.header.frame_id = "earth"
            
            # Add waypoints to follower path with appropriate offset
            for waypoint in segment_waypoints:
                pose = PoseStamped()
                pose.pose.position.x = waypoint[0] + offset[0]
                pose.pose.position.y = waypoint[1] + offset[1]
                pose.pose.position.z = waypoint[2]
                follower_path.poses.append(pose)
            
            # Command follower to follow the path
            drone.do_behavior("follow_path", 
                             follower_path, 
                             FLIGHT_SPEED,
                             YawMode.PATH_FACING, 
                             0.0, 
                             "earth", 
                             False)
        
        # Wait for all drones to complete their paths
        self.wait_all_drones()
        
        # Update waypoint index for next segment
        self.current_waypoint_index = (self.current_waypoint_index + segment_size) % len(self.circle_trajectory)
        
        # Change formation periodically
        if (self.current_waypoint_index % FORMATION_CHANGE_INTERVAL) < segment_size:
            # Alternate between line and V formations
            new_formation = "v" if self.current_formation == "line" else "line"
            self.change_formation(new_formation)

    def execute_stage1(self, config: dict):
        """Execute stage 1 - circular trajectory with changing formations
        
        Args:
            config: Configuration dictionary from YAML file
        """
        # Extract stage1 configuration
        stage1_config = config.get('stage1', {})
        stage_center = stage1_config.get('stage_center', [0.0, 0.0])
        trajectory_config = stage1_config.get('trajectory', {})
        diameter = trajectory_config.get('diameter', 3.0)
        formations = stage1_config.get('formations', ["line", "v"])
        
        print(f"Executing Stage 1 with center={stage_center}, diameter={diameter}")
        
        # Initialize with the first formation
        self.change_formation(formations[0])
        
        # Initialize circular trajectory
        self.initialize_circle_trajectory(stage_center, diameter)
        
        # Execute circular trajectory with formation changes using segments
        # We'll do two complete circles
        segment_size = 10  # Number of waypoints per segment
        num_segments = (2 * len(self.circle_trajectory)) // segment_size
        
        for _ in range(num_segments):
            self.move_to_next_segment(segment_size)
            
            # Small delay for visualization
            time.sleep(0.1)

    def execute_stage2(self, config: dict):
        """Execute stage 2 - window traversal
        
        Args:
            config: Configuration dictionary from YAML file
        """
        # Extract stage2 configuration
        stage2_config = config.get('stage2', {})
        stage_center = stage2_config.get('stage_center', [0.0, -6.0])
        room_height = stage2_config.get('room_height', 5.0)
        windows = stage2_config.get('windows', {})
        
        # Get traversal method from global macro
        traversal_method = STAGE2_METHOD
        
        print(f"Executing Stage 2 with center={stage_center}, method={traversal_method}")
        
        # Define window parameters
        window1 = windows.get('1', {})
        window1_center = window1.get('center', [-0.5, 1.5])
        window1_width = window1.get('gap_width', 2.0)
        window1_height = window1.get('height', 2.0)
        window1_floor = window1.get('distance_floor', 1.0)
        window1_thickness = window1.get('thickness', 0.3)
        
        window2 = windows.get('2', {})
        window2_center = window2.get('center', [1.0, -2.5])
        window2_width = window2.get('gap_width', 1.0)
        window2_height = window2.get('height', 1.5)
        window2_floor = window2.get('distance_floor', 3.0)
        window2_thickness = window2.get('thickness', 0.5)
        
        # Calculate absolute positions based on stage center
        window1_abs_center = [stage_center[0] + window1_center[0], 
                              stage_center[1] + window1_center[1]]
        window2_abs_center = [stage_center[0] + window2_center[0], 
                              stage_center[1] + window2_center[1]]
        
        # Define waypoints for the path
        start_pos = [window1_abs_center[0], window1_abs_center[1] + 3.0, 1.5]
        pre_window1_pos = [window1_abs_center[0], window1_abs_center[1] + 1.0, 
                           window1_floor + window1_height/2]
        post_window1_pos = [window1_abs_center[0], window1_abs_center[1] - window1_thickness - 1.0, 
                            window1_floor + window1_height/2]
        pre_window2_pos = [window2_abs_center[0], window2_abs_center[1] + 1.0, 
                           window2_floor + window2_height/2]
        post_window2_pos = [window2_abs_center[0], window2_abs_center[1] - window2_thickness - 1.0, 
                            window2_floor + window2_height/2]
        end_pos = [window2_abs_center[0], window2_abs_center[1] - 3.0, 1.5]
        
        print(f"Start position: {start_pos}")
        print(f"Pre window 1 position: {pre_window1_pos}")
        print(f"Post window 1 position: {post_window1_pos}")
        print(f"Pre window 2 position: {pre_window2_pos}")
        print(f"Post window 2 position: {post_window2_pos}")
        print(f"End position: {end_pos}")
        
        # Choose traversal method based on configuration
        if traversal_method.lower() == 'sequential':
            self._execute_stage2_sequential(
                start_pos, pre_window1_pos, post_window1_pos, 
                pre_window2_pos, post_window2_pos, end_pos)
        elif traversal_method.lower() == 'liney':
            self._execute_stage2_lineY(
                start_pos, pre_window1_pos, post_window1_pos, 
                pre_window2_pos, post_window2_pos, end_pos)
        else:
            print(f"Unknown traversal method: {traversal_method}. Using lineY method.")
            self._execute_stage2_lineY(
                start_pos, pre_window1_pos, post_window1_pos, 
                pre_window2_pos, post_window2_pos, end_pos)

    def _execute_stage2_sequential(self, start_pos, pre_window1_pos, post_window1_pos, 
                                  pre_window2_pos, post_window2_pos, end_pos):
        """Execute stage 2 using sequential traversal method
        
        Args:
            start_pos: Starting position
            pre_window1_pos: Position before window 1
            post_window1_pos: Position after window 1
            pre_window2_pos: Position before window 2
            post_window2_pos: Position after window 2
            end_pos: Final position
        """
        print("Stage 2: Using sequential traversal method")
        
        # Initialize with V formation
        self.change_formation("v")
        
        # Move the swarm to the starting position in V formation
        self._move_swarm_to_position(start_pos)
        
        print("Stage 2: Sequential window traversal starting")
        
        # Store the final positions for each drone
        final_positions = {}
        
        # Calculate final positions for all drones based on formation offsets
        for i in range(len(self.drones)):
            if i == 0:
                # Leader's final position
                final_positions[i] = end_pos
            else:
                # Follower's final position based on formation offset
                offset = self.formation_offsets[i]
                final_positions[i] = [
                    end_pos[0] + offset[0],
                    end_pos[1] + offset[1],
                    end_pos[2]
                ]
        
        # Sequential traversal for each drone
        for i, drone in self.drones.items():
            print(f"Stage 2: Drone {i} traversing windows")
            
            # Drone-specific waypoints
            if i == 0:
                # Leader follows the main path
                waypoints = [
                    pre_window1_pos,
                    post_window1_pos,
                    pre_window2_pos,
                    post_window2_pos,
                    end_pos
                ]
            else:
                # Calculate offset waypoints for followers
                offset = self.formation_offsets[i]
                
                # Adjust height for each drone to avoid collisions
                height_offset = 0.3 * i
                
                # Create waypoints with appropriate offsets
                drone_pre_window1 = [
                    pre_window1_pos[0],  # Same x as leader for window traversal
                    pre_window1_pos[1],
                    pre_window1_pos[2] + height_offset  # Different height
                ]
                
                drone_post_window1 = [
                    post_window1_pos[0],  # Same x as leader for window traversal
                    post_window1_pos[1],
                    post_window1_pos[2] + height_offset  # Different height
                ]
                
                drone_pre_window2 = [
                    pre_window2_pos[0],  # Same x as leader for window traversal
                    pre_window2_pos[1],
                    pre_window2_pos[2] + height_offset  # Different height
                ]
                
                drone_post_window2 = [
                    post_window2_pos[0],  # Same x as leader for window traversal
                    post_window2_pos[1],
                    post_window2_pos[2] + height_offset  # Different height
                ]
                
                waypoints = [
                    drone_pre_window1,
                    drone_post_window1,
                    drone_pre_window2,
                    drone_post_window2,
                    final_positions[i]
                ]
            
            # Create a path for the drone to follow
            path = Path()
            path.header.stamp = drone.get_clock().now().to_msg()
            path.header.frame_id = "earth"
            
            # Add current position as first waypoint
            current_pos = drone.position
            pose_current = PoseStamped()
            pose_current.pose.position.x = current_pos[0]
            pose_current.pose.position.y = current_pos[1]
            pose_current.pose.position.z = current_pos[2]
            path.poses.append(pose_current)
            
            # Add all waypoints to the path
            for waypoint in waypoints:
                pose = PoseStamped()
                pose.pose.position.x = waypoint[0]
                pose.pose.position.y = waypoint[1]
                pose.pose.position.z = waypoint[2]
                path.poses.append(pose)
            
            # Command drone to follow the path
            drone.do_behavior("follow_path", 
                             path, 
                             FLIGHT_SPEED,
                             YawMode.PATH_FACING, 
                             0.0, 
                             "earth", 
                             True)  # Wait for this drone to complete before next drone starts
            
            print(f"Stage 2: Drone {i} completed window traversal")
        
        print("Stage 2: All drones have traversed the windows")

    def _execute_stage2_lineY(self, start_pos, pre_window1_pos, post_window1_pos, 
                             pre_window2_pos, post_window2_pos, end_pos):
        """Execute stage 2 using lineY formation traversal method
        
        Args:
            start_pos: Starting position
            pre_window1_pos: Position before window 1
            post_window1_pos: Position after window 1
            pre_window2_pos: Position before window 2
            post_window2_pos: Position after window 2
            end_pos: Final position
        """
        print("Stage 2: Using lineY formation traversal method")
        
        # Start directly with lineY formation with reduced distance for window traversal
        WINDOW_FORMATION_DISTANCE = 0.4  # Reduced distance for window traversal
        self.formation_offsets = FormationManager.get_formation_offsets(
            "lineY", len(self.drones), WINDOW_FORMATION_DISTANCE)
        self.current_formation = "lineY"
        
        # Move the swarm to the starting position in lineY formation
        self._move_swarm_to_position(start_pos, use_layered=True)
        
        print("Stage 2: LineY formation window traversal starting")
        
        # Define all waypoints for the path
        waypoints = [
            start_pos,
            pre_window1_pos,
            post_window1_pos,
            pre_window2_pos,
            post_window2_pos,
            end_pos
        ]
        
        # Create paths for each drone to follow
        for i, drone in self.drones.items():
            # Create a path for this drone
            path = Path()
            path.header.stamp = drone.get_clock().now().to_msg()
            path.header.frame_id = "earth"
            
            # Add waypoints to the path with appropriate offsets
            for waypoint in waypoints:
                pose = PoseStamped()
                
                if i == 0:  # Leader follows the main path
                    pose.pose.position.x = waypoint[0]
                    pose.pose.position.y = waypoint[1]
                    pose.pose.position.z = waypoint[2]
                else:  # Followers maintain formation
                    # Get offset for this drone
                    offset = self.formation_offsets[i]
                    
                    # Apply offset to waypoint
                    pose.pose.position.x = waypoint[0] + offset[0]
                    pose.pose.position.y = waypoint[1] + offset[1]
                    pose.pose.position.z = waypoint[2]
                
                path.poses.append(pose)
            
            # Command drone to follow the path
            print(f"Stage 2: Drone {i} following window traversal path")
            drone.do_behavior("follow_path", 
                             path, 
                             FLIGHT_SPEED,
                             YawMode.PATH_FACING, 
                             0.0, 
                             "earth", 
                             False)  # Don't wait - we'll wait for all drones together
        
        # Wait for all drones to complete their paths
        self.wait_all_drones()
        
        # Keep lineY formation at the end (no need to change)
        print("Stage 2: LineY formation traversal completed")

    def _move_swarm_to_position(self, leader_position: List[float], use_layered: bool = False):
        """Move the entire swarm to a new position while maintaining formation
        
        Args:
            leader_position: [x, y, z] target position for the leader
            use_layered: Whether to use layered approach for collision avoidance
        """
        # Command leader to move to the position
        self.leader.do_behavior("go_to", 
                               leader_position[0],
                               leader_position[1],
                               leader_position[2],
                               FLIGHT_SPEED,
                               YawMode.PATH_FACING, 
                               0.0, 
                               "earth", 
                               False)  # Don't wait - we'll wait for all drones together
        
        # Move followers to maintain formation
        for i, drone in self.drones.items():
            if i == 0:  # Skip leader
                continue
            
            # Get offset for this drone in the formation
            offset = self.formation_offsets[i]
            
            # Calculate target position based on formation offset
            target_x = leader_position[0] + offset[0]
            target_y = leader_position[1] + offset[1]
            target_z = leader_position[2]
            
            # Move follower to its position
            if use_layered:
                # Use layered approach for collision avoidance during formation changes
                drone.go_to_layered([target_x, target_y, target_z])
            else:
                # Direct movement when formation is already established
                drone.do_behavior("go_to", 
                                 target_x,
                                 target_y,
                                 target_z,
                                 FLIGHT_SPEED,
                                 YawMode.PATH_FACING, 
                                 0.0, 
                                 "earth", 
                                 False)
        
        # Wait for all drones to reach their positions
        self.wait_all_drones()

    def is_formation_achieved(self) -> bool:
        """Check if the current formation matches the desired formation
        
        Returns:
            bool: True if formation is achieved, False otherwise
        """
        FORMATION_TOLERANCE = 0.1  # 10cm tolerance
        
        # Skip check if we don't have formation offsets yet
        if not hasattr(self, 'formation_offsets') or not self.formation_offsets:
            return False
        
        # Get leader position
        leader_pos = self.leader.position
        
        # Check each follower's position against expected position
        for i, drone in self.drones.items():
            if i == 0:  # Skip leader
                continue
            
            # Get current position
            current_pos = drone.position
            
            # Get expected position based on formation offset
            offset = self.formation_offsets[i]
            expected_x = leader_pos[0] + offset[0]
            expected_y = leader_pos[1] + offset[1]
            expected_z = leader_pos[2]
            
            # Calculate distance to expected position
            dx = current_pos[0] - expected_x
            dy = current_pos[1] - expected_y
            dz = current_pos[2] - expected_z
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # If any drone is not in position, formation is not achieved
            if distance > FORMATION_TOLERANCE:
                return False
        
        # All drones are in position
        return True


def confirm(msg: str = 'Continue') -> bool:
    """Confirm message"""
    confirmation = input(f"{msg}? (y/n): ")
    if confirmation == "y":
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Multi-drone formation flight mission')

    parser.add_argument('-n', '--namespaces',
                        type=str,
                        nargs='+',
                        default=['drone0', 'drone1', 'drone2'],
                        help='Namespaces of the drones to be used in the mission')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time',
                        action='store_true',
                        default=True,
                        help='Use simulation time')
    parser.add_argument('-c', '--config',
                        type=str,
                        default='src/challenge_multi_drone/scenarios/scenario1.yaml',
                        help='Path to the config file')

    args = parser.parse_args()
    drones_namespace = args.namespaces
    verbosity = args.verbose
    use_sim_time = args.use_sim_time
    config_path = args.config   
    config = read_config_from_yaml(config_path)

    rclpy.init()
    swarm = SwarmConductor(
        drones_namespace,
        verbose=verbosity,
        use_sim_time=use_sim_time)

    if confirm("Takeoff"):
        swarm.get_ready()
        swarm.takeoff()

        if confirm("Stage 1"):
            swarm.execute_stage1(config)
            
        if confirm("Stage 2"):
            swarm.execute_stage2(config)

        confirm("Land")
        swarm.land()

    print("Shutdown")
    swarm.shutdown()
    rclpy.shutdown()

    sys.exit(0)


if __name__ == '__main__':
    main()
