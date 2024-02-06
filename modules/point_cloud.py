import open3d as o3d
import numpy as np
import os

class PointCloud:
    def __init__(self, config, kp1=None, kp2=None, current_depth=None, target_depth=None, pc1=None, pc2=None, points1=None, points2=None, colors1=None, colors2=None):
            self.config = config
            self.kp1 = kp1
            self.kp2 = kp2
            self.current_depth = current_depth
            self.target_depth = target_depth
            self.pc1 = pc1
            self.pc2 = pc2
            self.points1 = points1
            self.points2 = points2
            self.colors1 = colors1
            self.colors2 = colors2
            self.vm_id = 0
            self.count_steps = 0
            self.K = np.array([[256, 0, 128],[0, 256, 128],[0, 0, 1]])

    def get_3d_points(self, kp1, kp2, current_depth, target_depth, current_color, target_color, vm_id, count_steps):
        if self.config['feature_matching']['descriptor'] == 'SuperGlue':
            if self.config['mode']['color']:
                points1, colors1, points2, colors2 = self.sg_get_3d_points_with_color(kp1, kp2, current_depth, target_depth, current_color, target_color)
                self.points1 = points1
                self.points2 = points2
                self.colors1 = colors1
                self.colors2 = colors2
                filtered_points1, filtered_colors1 = self.filter_invalid_points_with_color(points1, colors1)
                filtered_points2, filtered_colors2 = self.filter_invalid_points_with_color(points2, colors2)
                pc1 = self.create_colored_point_cloud(filtered_points1, filtered_colors1)
                pc2 = self.create_colored_point_cloud(filtered_points2, filtered_colors2)
            else:
                points1, points2 = self.sg_get_3d_points(kp1, kp2, current_depth, target_depth)
                self.points1 = points1
                self.points2 = points2
                points1 = self.filter_invalid_points(points1.reshape(-1, 3))
                points2 = self.filter_invalid_points(points2.reshape(-1, 3))
                pc1 = o3d.geometry.PointCloud()
                pc1.points = o3d.utility.Vector3dVector(points1)
                pc2 = o3d.geometry.PointCloud()
                pc2.points = o3d.utility.Vector3dVector(points2)

        # Other methods to implement with ORB, BRISK, etc.
        self.pc1 = pc1
        self.pc2 = pc2
        self.vm_id = vm_id
        self.count_steps = count_steps
        self.save_pc()
        return self.pc1, self.pc2
    
    def save_pc(self): # Not yet implemented
        if self.config['logs']['pc']:
            pc_dir = self.config['paths']['LOGS_DIR'] + 'point_clouds'
            os.makedirs(pc_dir, exist_ok=True)
            print("Saving point clouds to: ", pc_dir)
            pc1_path = pc_dir + f"/pc_current_{self.count_steps:04d}.ply"
            pc2_path = pc_dir + f"/pc_target_{self.count_steps:04d}.ply"
            o3d.io.write_point_cloud(pc1_path, self.pc1)
            o3d.io.write_point_cloud(pc2_path, self.pc2) # not sure if colored pc or not 

    
    def filter_invalid_points(self, points):
        valid_indices = np.where(points[:, 2] > 0)[0]
        return points[valid_indices]

    def create_colored_point_cloud(self, points, colors):
        """
        Create an Open3D PointCloud object from points and colors.

        Args:
        points (np.ndarray): Array of points (shape Nx3).
        colors (np.ndarray): Array of colors corresponding to the points (shape Nx3).

        Returns:
        open3d.geometry.PointCloud: Colored point cloud in Open3D format.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def filter_invalid_points_with_color(self, points, colors):
        """
        Filter out points with invalid depth values and their corresponding colors.

        Args:
        points (np.ndarray): Array of points (shape Nx3).
        colors (np.ndarray): Array of colors corresponding to the points (shape Nx3).

        Returns:
        np.ndarray, np.ndarray: Filtered arrays of points and colors.
        """
        valid_indices = np.where(points[:, 2] > 0)[0]
        return points[valid_indices], colors[valid_indices]
    
    def sg_get_3d_points_with_color(self, kp1, kp2, depth_img1, depth_img2, color_img1, color_img2):
        points1 = []
        colors1 = []
        points2 = []
        colors2 = []
        len_matches = len(kp1)
        K = self.K

        for m in range(len_matches):
            # Image 1
            u1, v1 = int(kp1[m][0]), int(kp1[m][1])
            z1 = depth_img1[v1, u1]
            if isinstance(z1, np.ndarray):
                z1 = z1[0] if z1.shape[0] > 0 else None
            color1 = color_img1[v1, u1] / 255.0  # Normalize color to [0, 1]

            # Image 2
            u2, v2 = int(kp2[m][0]), int(kp2[m][1])
            z2 = depth_img2[v2, u2]
            if isinstance(z2, np.ndarray):
                z2 = z2[0] if z2.shape[0] > 0 else None
            color2 = color_img2[v2, u2] / 255.0  # Normalize color to [0, 1]

            if z1 is not None and z1 > 0 and z2 is not None and z2 > 0:
                x1 = (u1 - K[0, 2]) * z1 / K[0, 0]
                y1 = (v1 - K[1, 2]) * z1 / K[1, 1]
                points1.append([x1, y1, z1])
                colors1.append(color1)

                x2 = (u2 - K[0, 2]) * z2 / K[0, 0]
                y2 = (v2 - K[1, 2]) * z2 / K[1, 1]
                points2.append([x2, y2, z2])
                colors2.append(color2)

        return np.array(points1), np.array(colors1), np.array(points2), np.array(colors2)
    
    def sg_get_3d_points(self, kp1, kp2, depth_img1, depth_img2):
        points1 = []
        points2 = []
        len_matches = len(kp1)
        K = self.K
        for m in range(len_matches):
            # Image 1
            u1, v1 = int(kp1[m][0]), int(kp1[m][1])
            z1 = depth_img1[v1, u1]
            if isinstance(z1, np.ndarray):
                z1 = z1[0] if z1.shape[0] > 0 else None

            # Image 2
            u2, v2 = int(kp2[m][0]), int(kp2[m][1])
            z2 = depth_img2[v2, u2]
            if isinstance(z2, np.ndarray):
                z2 = z2[0] if z2.shape[0] > 0 else None

            if z1 is not None and z1 > 0 and z2 is not None and z2 > 0:
                x1 = (u1 - K[0, 2]) * z1 / K[0, 0]
                y1 = (v1 - K[1, 2]) * z1 / K[1, 1]
                points1.append([x1, y1, z1])

                x2 = (u2 - K[0, 2]) * z2 / K[0, 0]
                y2 = (v2 - K[1, 2]) * z2 / K[1, 1]
                points2.append([x2, y2, z2])

        return np.array(points1), np.array(points2)


    def merge_and_recolor_point_clouds(self):
        # Define red and blue colors
        red = np.array([1, 0, 0])  # RGB for red
        blue = np.array([0, 0, 1])  # RGB for blue

        # Assign red color to all points in pc1 regardless of existing color
        colors_pc1 = np.tile(red, (len(self.pc1.points), 1))
        self.pc1.colors = o3d.utility.Vector3dVector(colors_pc1)

        # Assign blue color to all points in pc2 regardless of existing color
        colors_pc2 = np.tile(blue, (len(self.pc2.points), 1))
        self.pc2.colors = o3d.utility.Vector3dVector(colors_pc2)

        # Merge the recolored point clouds
        merged_pc = self.pc1 + self.pc2

        # Save the merged and recolored point cloud
        self.save_merged_and_recolored_pc(merged_pc)

    def save_merged_and_recolored_pc(self, point_cloud):
        # Define the directory and filename for saving the point cloud
        pc_dir = self.config['paths']['LOGS_DIR'] + 'merged_recolored_point_clouds'
        os.makedirs(pc_dir, exist_ok=True)
        pc_path = pc_dir + f"/merged_recolored_pc_{self.count_steps:04d}.ply"
        
        # Save the point cloud as a .ply file
        o3d.io.write_point_cloud(pc_path, point_cloud)
        print(f"Merged and recolored point cloud saved to: {pc_path}")
