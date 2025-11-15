import os
import torch
import numpy as np
from tqdm import tqdm

class PointCloudCache:
    """
    高效的点云数据缓存类，用于加载和缓存点云数据
    """
    def __init__(self, pc_path, num_samples, preload=False, verbose=True):
        """
        初始化点云缓存
        
        Args:
            pc_path: 预采样点云的保存路径
            num_samples: 所需的点数
            preload: 是否在初始化时预加载所有数据（内存充足时推荐）
            verbose: 是否显示加载进度
        """
        self.pc_path = pc_path
        self.num_samples = num_samples
        self.cache = {}  # 点云数据缓存
        self.verbose = verbose
        
        # 验证路径
        if not os.path.exists(pc_path):
            raise ValueError(f"点云路径不存在: {pc_path}")
            
        # 如果需要预加载，加载路径下的所有点云文件
        if preload:
            self._preload_all_point_clouds()
    
    def _preload_all_point_clouds(self):
        """预加载所有点云数据"""
        if self.verbose:
            print(f"预加载点云数据从: {self.pc_path}")
            print(f"检查路径存在性: {os.path.exists(self.pc_path)}")
            
        # 递归查找所有.pt文件
        pt_files = []
        if os.path.exists(self.pc_path):
            for root, dirs, files in os.walk(self.pc_path):
                for file in files:
                    if file.endswith('.pt'):
                        full_path = os.path.join(root, file)
                        # 获取相对于pc_path的路径
                        rel_path = os.path.relpath(full_path, self.pc_path)
                        pt_files.append(rel_path)
        
        if self.verbose:
            print(f"找到 {len(pt_files)} 个.pt文件")
            # 显示一些找到的文件示例
            if len(pt_files) > 0:
                print("文件示例:")
                for i, file in enumerate(pt_files[:5]):
                    print(f"  - {file}")
                if len(pt_files) > 5:
                    print(f"  - ...还有 {len(pt_files) - 5} 个文件")
        
        # 如果没有找到文件，检查目录结构
        if len(pt_files) == 0 and self.verbose:
            print("\n检查目录结构:")
            if os.path.exists(self.pc_path):
                print(f"目录 {self.pc_path} 内容:")
                for item in os.listdir(self.pc_path):
                    item_path = os.path.join(self.pc_path, item)
                    item_type = "目录" if os.path.isdir(item_path) else "文件"
                    print(f"  - {item} ({item_type})")
                    
                    # 如果是目录，继续检查一级子目录
                    if os.path.isdir(item_path):
                        subdir_items = os.listdir(item_path)
                        if len(subdir_items) > 0:
                            print(f"    子目录 {item} 内容 (显示最多5项):")
                            for i, subitem in enumerate(subdir_items[:5]):
                                subitem_path = os.path.join(item_path, subitem)
                                subitem_type = "目录" if os.path.isdir(subitem_path) else "文件"
                                print(f"      - {subitem} ({subitem_type})")
                            if len(subdir_items) > 5:
                                print(f"      - ...还有 {len(subdir_items) - 5} 项")
        
        if len(pt_files) > 0:
            # 预加载找到的点云文件
            with tqdm(total=len(pt_files), desc="预加载点云", disable=not self.verbose) as pbar:
                for rel_path in pt_files:
                    # 将文件路径转换为任务ID
                    # 替换操作系统路径分隔符为统一的'/'
                    task_id = rel_path.replace('\\', '/').replace('.pt', '')
                    try:
                        self._load_single_pc(task_id)
                        pbar.update(1)
                    except Exception as e:
                        if self.verbose:
                            print(f"加载文件 {rel_path} 失败: {str(e)}")
                        continue
        
        if self.verbose:
            print(f"预加载完成，共加载 {len(self.cache)} 个点云文件")
            
    def _load_single_pc(self, task_id):
        """加载单个点云文件"""
        pc_file = os.path.join(self.pc_path, f"{task_id}.pt")
        try:
            if not os.path.exists(pc_file):
                # 尝试在任务ID的子目录中查找
                alt_pc_file = os.path.join(self.pc_path, task_id + ".pt")
                if os.path.exists(alt_pc_file):
                    pc_file = alt_pc_file
                else:
                    raise FileNotFoundError(f"点云文件不存在: {pc_file} 或 {alt_pc_file}")
                
            pc = torch.load(pc_file)
            # 如果点数不匹配，进行采样或补充
            if pc.shape[0] != self.num_samples:
                if pc.shape[0] > self.num_samples:
                    # 随机采样到指定数量（确保使用相同的种子以便结果可重现）
                    with torch.random.fork_rng():
                        torch.manual_seed(0)  # 固定种子
                        indices = torch.randperm(pc.shape[0])[:self.num_samples]
                    pc = pc[indices]
                else:
                    # 如果点数不够，通过重复补充
                    repeat_times = self.num_samples // pc.shape[0] + 1
                    pc_repeated = pc.repeat(repeat_times, 1)
                    pc = pc_repeated[:self.num_samples]
            
            # 存入缓存
            self.cache[task_id] = pc
            return pc
        except Exception as e:
            raise FileNotFoundError(f"无法加载点云文件: {pc_file}, 错误: {str(e)}")
    
    def load_batch(self, task_ids, device=None):
        """
        加载一批点云数据
        
        Args:
            task_ids: 任务ID列表，每个ID的格式为 "cat_id/model_id/view_id"
            device: 设备（如果需要直接加载到特定设备）
            
        Returns:
            pytorch tensor 形状为 [batch_size, num_samples, 3]
        """
        batch_pcs = []
        for task_id in task_ids:
            # 从缓存中获取，或者按需加载
            try:
                if task_id in self.cache:
                    pc = self.cache[task_id]
                else:
                    pc = self._load_single_pc(task_id)
                batch_pcs.append(pc)
            except Exception as e:
                print(f"警告: 无法加载任务ID: {task_id}, 错误: {str(e)}")
                # 创建一个空的点云作为替代
                empty_pc = torch.zeros((self.num_samples, 3))
                batch_pcs.append(empty_pc)
        
        # 将列表转换为批量张量
        batch_pcs_tensor = torch.stack(batch_pcs, dim=0)
        
        # 如果指定了设备，则移动到该设备
        if device is not None:
            batch_pcs_tensor = batch_pcs_tensor.to(device)
            
        return batch_pcs_tensor

# 便捷函数：返回常用扩展名的所有PC文件的任务ID列表
def get_all_pc_task_ids(pc_folder, extension='.pt'):
    """获取指定文件夹中所有点云文件的任务ID列表"""
    task_ids = []
    for filename in os.listdir(pc_folder):
        if filename.endswith(extension):
            task_id = filename[:-len(extension)]
            task_ids.append(task_id)
    return task_ids 