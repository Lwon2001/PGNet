import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 虽然不直接使用，但能保证 3D 投影正常工作
from typing import Union

def point_cloud_render(point_cloud: Union[torch.Tensor, np.ndarray], output_file: str = "point_cloud_matplotlib.png"):
    """
    使用 Matplotlib 将 (N, 3) 点云数据以 3D 散点图形式保存为图片（参考示例代码改进）
    
    参数:
      point_cloud (torch.Tensor): 点云数据，形状为 (N, 3)
      output_file (str): 图片保存路径，默认 "point_cloud_matplotlib.png"
    """
    # 如果输入是 torch.Tensor，转换为 numpy 数组
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().detach().numpy()
    
    # 创建图形和 3D 坐标轴，图形大小设为正方形
    fig = plt.figure(figsize=(8, 8))
    try:
        ax = fig.gca(projection='3d', adjustable='box')
    except Exception:
        ax = fig.add_subplot(111, projection='3d', adjustable='box')
    
    # 隐藏坐标轴显示
    ax.axis('off')
    
    # 交换点云坐标顺序，参考代码中 x, z, y 的赋值顺序
    x, z, y = point_cloud.transpose(1, 0)
    
    # 设置视角
    ax.view_init(30, 45)
    
    # 根据点云全局的最小值和最大值设置坐标轴范围
    pt_min = np.min(point_cloud)
    pt_max = np.max(point_cloud)
    ax.set_xbound(pt_min, pt_max)
    ax.set_ybound(pt_min, pt_max)
    ax.set_zbound(pt_min, pt_max)
    
    # 绘制散点图，使用 x 值作为颜色映射依据，点大小设为1
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet', s=1)
    
    # 保存图片
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

# 示例使用：
if __name__ == "__main__":
    import os
    pc = torch.load("/home/ubuntu/data/projects/TRELLIS/data/ShapeNetViPC-Dataset/ShapeNetViPC-PointCloud/04090263/8181c6ea706bff129f433921788191f3/08.pt")
    
    output_file = "./output/point_cloud_matplotlib.png"
    point_cloud_render(pc, output_file)
    
    # 打印绝对路径
    abs_path = os.path.abspath(output_file)
    print(f"点云图像已保存到: {abs_path}")