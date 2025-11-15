import numpy as np
import open3d as o3d
import trimesh
import kaolin.ops.mesh as mesh_ops
import torch

def sample_mesh_to_pointcloud(mesh, num_points, sampling_method="uniform"):
    """
    将单个 mesh 转换为点云。

    参数：
      mesh: 包含 mesh.vertices 和 mesh.faces 的对象（例如 MeshExtractResult）。
      num_points: 采样的点数。
      sampling_method: 采样方法，支持 "uniform", "random", "poisson_disk"，
                       "uniform" 使用 trimesh 面积加权采样，
                       "random" 对三角面均匀采样，
                       "poisson_disk" 使用 open3d 的泊松盘采样。

    返回：
      points: numpy 数组，形状为 (num_points, 3)。
    """

    
    if sampling_method == "uniform":
        # 使用 trimesh 的面积加权采样
        # trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # points, _ = trimesh_mesh.sample(num_points, return_index=True)
        return kaolin_sample_point_cloud(mesh, num_points)
    else:
        # 将 vertices 和 faces 转为 numpy 数组
        if hasattr(mesh.vertices, "cpu"):
            vertices = mesh.vertices.cpu().numpy()
        else:
            vertices = np.array(mesh.vertices)
            
        if hasattr(mesh.faces, "cpu"):
            faces = mesh.faces.cpu().numpy()
        else:
            faces = np.array(mesh.faces)
            
        if sampling_method == "random":
            # 随机采样：对所有三角面以相同概率进行采样（不考虑面面积）
            trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            num_triangles = len(trimesh_mesh.faces)
            # 随机选择三角面索引（不加权）
            triangle_indices = np.random.randint(0, num_triangles, size=num_points)
            sampled_points = []
            for idx in triangle_indices:
                tri_indices = trimesh_mesh.faces[idx]
                v0, v1, v2 = trimesh_mesh.vertices[tri_indices]
                # 生成两个随机数
                r1, r2 = np.random.rand(2)
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                # 根据重心坐标采样
                point = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
                sampled_points.append(point)
            return np.array(sampled_points)
        elif sampling_method == "poisson_disk":
            # 使用 open3d 进行泊松盘采样
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
            o3d_mesh.compute_vertex_normals()
            pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=num_points)
            points = np.asarray(pcd.points)
            return points
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}") 
    
    
    
def kaolin_sample_point_cloud(mesh, num_points=10000):
    """
    从mesh表面采样点云，并渲染保存为图片

    参数:
        mesh: 输入的mesh对象 (MeshExtractResult 类型)
        num_points: 要采样的点数，默认100000
        output_file: 输出图片路径，默认"point_cloud.png"
        use_normals: 是否使用法线信息，默认True
        multi_view: 是否渲染多个视角，默认True
    """
    # 从mesh中提取顶点和面
    vertices = mesh.vertices
    faces = mesh.faces

    # 确保数据类型正确
    device = vertices.device
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, device=device)
    if not isinstance(faces, torch.Tensor):
        faces = torch.tensor(faces, device=device)

    # 确保faces是长整型
    faces = faces.long()

    # 使用kaolin从mesh表面采样点云
    vertices_batch = vertices.unsqueeze(0)  # [1, N, 3]

    # 采样点云
    points, face_indices = mesh_ops.sample_points(vertices_batch, faces, num_points)
    points = points.squeeze(0)

    return points