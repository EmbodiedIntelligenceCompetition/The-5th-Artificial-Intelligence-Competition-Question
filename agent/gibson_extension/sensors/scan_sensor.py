import cv2
import numpy as np
import pybullet as p
from transforms3d.quaternions import quat2mat

from igibson.sensors.dropout_sensor_noise import DropoutSensorNoise
from igibson.sensors.sensor_base import BaseSensor
from igibson.utils.constants import OccupancyGridState

class ScanSensor(BaseSensor):
    """
    1D LiDAR scanner sensor and occupancy grid sensor
    """

    def __init__(self, env, modalities, rear=False):
        super(ScanSensor, self).__init__(env)
        self.modalities = modalities
        self.scan_noise_rate = self.config.get("scan_noise_rate", 0.0)
        self.n_horizontal_rays = self.config.get("n_horizontal_rays", 128)
        self.n_vertical_beams = self.config.get("n_vertical_beams", 1)
        assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
        self.laser_linear_range = self.config.get("laser_linear_range", 10.0)
        self.laser_angular_range = self.config.get("laser_angular_range", 180.0)
        self.min_laser_dist = self.config.get("min_laser_dist", 0.05)
        self.laser_link_name = (
            self.config.get("laser_link_name", "scan_link")
            if not rear
            else self.config.get("laser_link_rear_name", "scan_link")
        )
        self.noise_model = DropoutSensorNoise(env)
        self.noise_model.set_noise_rate(self.scan_noise_rate)
        self.noise_model.set_noise_value(1.0)
        self.rear = rear

        self.laser_position, self.laser_orientation = (
            env.robots[0].links[self.laser_link_name].get_position_orientation()
        )
        self.base_position, self.base_orientation = env.robots[0].base_link.get_position_orientation()
        

        if "occupancy_grid" in self.modalities:
            self.grid_resolution = self.config.get("grid_resolution", 128)
            self.occupancy_range = self.config.get("occupancy_range", 5)  # m
            self.robot_footprint_radius = self.config.get("robot_footprint_radius", 0.32)
            self.robot_footprint_radius_in_map = int(
                self.robot_footprint_radius / self.occupancy_range * self.grid_resolution
            )

            # Initialize global occupancy grid
            self.global_grid_resolution = self.config.get("global_grid_resolution", 512)
            self.global_occupancy_grid = np.zeros((self.global_grid_resolution, self.global_grid_resolution)).astype(np.uint8)
            self.global_occupancy_grid.fill(int(OccupancyGridState.UNKNOWN * 2.0))

    def get_local_occupancy_grid(self, scan, target_pos, cur_pos, cur_ori):
        """
        Get local occupancy grid based on current 1D scan

        :param scan: 1D LiDAR scan
        :return: local occupancy grid
        """
        laser_linear_range = self.laser_linear_range
        laser_angular_range = self.laser_angular_range
        min_laser_dist = self.min_laser_dist

        laser_angular_half_range = laser_angular_range / 2.0

        angle = np.arange(
            -np.radians(laser_angular_half_range),
            np.radians(laser_angular_half_range),
            np.radians(laser_angular_range) / self.n_horizontal_rays,
        )
        unit_vector_laser = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])

        scan_laser = unit_vector_laser * (scan * (laser_linear_range - min_laser_dist) + min_laser_dist)

        laser_translation = self.laser_position
        laser_rotation = quat2mat(
            [self.laser_orientation[3], self.laser_orientation[0], self.laser_orientation[1], self.laser_orientation[2]]
        )
        scan_world = laser_rotation.dot(scan_laser.T).T + laser_translation

        base_translation = self.base_position
        base_rotation = quat2mat(
            [self.base_orientation[3], self.base_orientation[0], self.base_orientation[1], self.base_orientation[2]]
        )
        scan_local = base_rotation.T.dot((scan_world - base_translation).T).T
        scan_local = scan_local[:, :2]
        scan_local = np.concatenate([np.array([[0, 0]]), scan_local, np.array([[0, 0]])], axis=0)

        # Flip y axis
        scan_local[:, 1] *= -1

        occupancy_grid = np.zeros((self.grid_resolution, self.grid_resolution)).astype(np.uint8)
        occupancy_grid.fill(int(OccupancyGridState.UNKNOWN * 2.0))
        
        scan_local_in_map = scan_local / self.occupancy_range * self.grid_resolution + (self.grid_resolution / 2)
        scan_local_in_map = scan_local_in_map.reshape((1, -1, 1, 2)).astype(np.int32)
        for i in range(scan_local_in_map.shape[1]):
            cv2.circle(
                img=occupancy_grid,
                center=(scan_local_in_map[0, i, 0, 0], scan_local_in_map[0, i, 0, 1]),
                radius=2,
                color=int(OccupancyGridState.OBSTACLES * 2.0),
                thickness=-1,
            )
        cv2.fillPoly(
            img=occupancy_grid, pts=scan_local_in_map, color=int(OccupancyGridState.FREESPACE * 2.0), lineType=1
        )
        cv2.circle(
            img=occupancy_grid,
            center=(self.grid_resolution // 2, self.grid_resolution // 2),
            radius=int(self.robot_footprint_radius_in_map),
            color=int(OccupancyGridState.OBSTACLES * 2.0),
            thickness=-1,
        )

        # Draw the agent's heading direction
        center = (self.grid_resolution // 2, self.grid_resolution // 2)
        heading_vector = np.array([1.0, 0.0, 0.0])
        heading_world = base_rotation.dot(heading_vector)
        heading_local = base_rotation.T.dot(heading_world)
        heading_local = heading_local[:2]
        heading_local[1] *= -1  # Flip y axis
        heading_local = heading_local / np.linalg.norm(heading_local) * (self.grid_resolution // 15)  # Adjust the scale here
        end_point = (int(center[0] + heading_local[0]), int(center[1] + heading_local[1]))

        cv2.arrowedLine(occupancy_grid, center, end_point, color=int(OccupancyGridState.UNKNOWN * 2.0), thickness=2)

        # Draw target position on the occupancy grid
        cur_base_rotation = quat2mat(
            [cur_ori[3], cur_ori[0], cur_ori[1], cur_ori[2]]
        )
        target_pos_local = target_pos - cur_pos
        target_pos_local = cur_base_rotation.T.dot(target_pos_local)
        target_pos_local = target_pos_local[:2]
        target_pos_local[1] *= -1  # Flip y axis
        target_pos_local = target_pos_local / self.occupancy_range * self.grid_resolution + (self.grid_resolution / 2)
        cv2.circle(
            img=occupancy_grid,
            center=(int(target_pos_local[0]), int(target_pos_local[1])),
            radius=2,
            color=int(OccupancyGridState.OBSTACLES * 2.0),
            thickness=-1,
        )


        return occupancy_grid[:, :, None].astype(np.float32) / 2.0

    def get_obs(self, env):
        """
        Get current LiDAR sensor reading and occupancy grid (optional)

        :return: LiDAR sensor reading and local occupancy grid, normalized to [0.0, 1.0]
        """
        laser_angular_half_range = self.laser_angular_range / 2.0
        if self.laser_link_name not in env.robots[0].links:
            raise Exception(
                "Trying to simulate LiDAR sensor, but laser_link_name cannot be found in the robot URDF file. Please add a link named laser_link_name at the intended laser pose. Feel free to check out assets/models/turtlebot/turtlebot.urdf and examples/configs/turtlebot_p2p_nav.yaml for examples."
            )
        laser_position, laser_orientation = env.robots[0].links[self.laser_link_name].get_position_orientation()
        angle = np.arange(
            -laser_angular_half_range / 180 * np.pi,
            laser_angular_half_range / 180 * np.pi,
            self.laser_angular_range / 180.0 * np.pi / self.n_horizontal_rays,
        )
        unit_vector_local = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])
        transform_matrix = quat2mat(
            [laser_orientation[3], laser_orientation[0], laser_orientation[1], laser_orientation[2]]
        )  # [x, y, z, w]
        unit_vector_world = transform_matrix.dot(unit_vector_local.T).T

        start_pose = np.tile(laser_position, (self.n_horizontal_rays, 1))
        start_pose += unit_vector_world * self.min_laser_dist
        end_pose = laser_position + unit_vector_world * self.laser_linear_range
        results = p.rayTestBatch(start_pose, end_pose, numThreads=6)  # numThreads = 6

        # hit fraction = [0.0, 1.0] of self.laser_linear_range
        hit_fraction = np.array([item[2] for item in results])
        hit_fraction = self.noise_model.add_noise(hit_fraction)
        scan = np.expand_dims(hit_fraction, 1)

        state = {}
        state["scan" if not self.rear else "scan_rear"] = scan.astype(np.float32)
        if "occupancy_grid" in self.modalities:
            # self.update_global_occupancy_grid(scan)
            # state["global_occupancy_grid"] = self.global_occupancy_grid.astype(np.float32) / 2.0
            state["global_occupancy_grid"] = self.get_local_occupancy_grid(scan, env.task.target_pos, *env.robots[0].base_link.get_position_orientation())
        
        return state


    def update_global_occupancy_grid(self, scan):
        """
        Update global occupancy grid based on current 1D scan

        :param: 1D LiDAR scan
        """
        local_grid = self.get_local_occupancy_grid(scan)

        base_position, base_orientation = self.base_position, self.base_orientation
        base_rotation = quat2mat([base_orientation[3], base_orientation[0], base_orientation[1], base_orientation[2]])
        base_rotation_inv = base_rotation.T

        # 机器人在全局占用图中的位置
        global_center_x = int(base_position[0] / self.occupancy_range * self.global_grid_resolution + self.global_grid_resolution / 2)
        global_center_y = int(base_position[1] / self.occupancy_range * self.global_grid_resolution + self.global_grid_resolution / 2)

        # 将局部栅格图合并到全局栅格图
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                if local_grid[i, j, 0] != int(OccupancyGridState.UNKNOWN * 2.0):
                    local_x = i - self.grid_resolution // 2
                    local_y = j - self.grid_resolution // 2
                    global_x = int(base_rotation_inv[0, 0] * local_x + base_rotation_inv[0, 1] * local_y + global_center_x)
                    global_y = int(base_rotation_inv[1, 0] * local_x + base_rotation_inv[1, 1] * local_y + global_center_y)
                    if 0 <= global_x < self.global_grid_resolution and 0 <= global_y < self.global_grid_resolution:
                        self.global_occupancy_grid[global_x, global_y] = local_grid[i, j, 0]