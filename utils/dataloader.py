import os
from turtle import Turtle
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm
import rembg
import trimesh
import json
from pathlib import Path


# 旋转Z轴函数，用于绕Z轴旋转点云数据
def rotation_z(pts, theta):
    cos_theta = np.cos(theta)  # 计算旋转角度的余弦值
    sin_theta = np.sin(theta)  # 计算旋转角度的正弦值
    # 构造绕Z轴旋转的3x3旋转矩阵
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                 [sin_theta, cos_theta, 0.0],
                                 [0.0, 0.0, 1.0]])
    # 将点云数据与旋转矩阵的转置相乘，实现旋转操作
    return pts @ rotation_matrix.T


# 旋转Y轴函数，用于绕Y轴旋转点云数据
def rotation_y(pts, theta):
    cos_theta = np.cos(theta)  # 计算旋转角度的余弦值
    sin_theta = np.sin(theta)  # 计算旋转角度的正弦值
    # 构造绕Y轴旋转的3x3旋转矩阵
    rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                 [0.0, 1.0, 0.0],
                                 [sin_theta, 0.0, cos_theta]])
    # 将点云数据与旋转矩阵的转置相乘，实现旋转操作
    return pts @ rotation_matrix.T


# 旋转X轴函数，用于绕X轴旋转点云数据
def rotation_x(pts, theta):
    cos_theta = np.cos(theta)  # 计算旋转角度的余弦值
    sin_theta = np.sin(theta)  # 计算旋转角度的正弦值
    # 构造绕X轴旋转的3x3旋转矩阵
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                 [0.0, cos_theta, -sin_theta],
                                 [0.0, sin_theta, cos_theta]])
    # 将点云数据与旋转矩阵的转置相乘，实现旋转操作
    return pts @ rotation_matrix.T

class PCDataLoader(Dataset):
    def __init__(self, filepath, data_path, status, pc_input_num=3500, view_align=False, category='all', sampling_method='poisson_disk', num_points=16384, normalize_gen=True):
        """
        数据加载器，在ViPCDataLoader基础上增加了生成点云的加载
        
        Args:
            filepath: 文件列表路径
            data_path: 数据集根目录
            status: 数据集状态 ('train' 或 'test')
            pc_input_num: 部分点云的输入点数量
            view_align: 是否对齐视图和部分点云
            category: 要加载的类别 ('all' 或特定类别名称)
            sampling_method: 点云采样方法 (默认为 'poisson_disk')
            num_points: 生成点云的点数量 (默认为 16384)
        """
        # 调用父类构造函数
        super(PCDataLoader, self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align
        self.sampling_method = sampling_method
        self.num_points = num_points
        self.normalize_gen = normalize_gen
        # 类别名称与对应ID的映射字典
        self.cat_map = {
            'plane': '02691156',
            'bench': '02828884',
            'cabinet': '02933112',
            'car': '02958343',
            'chair': '03001627',
            'monitor': '03211117',
            'lamp': '03636649',
            'speaker': '03691459',
            'firearm': '04090263',
            'couch': '04256520',
            'table': '04379243',
            'cellphone': '04401088',
            'watercraft': '04530566'
        }
        # 读取包含文件路径的文件，每行一个文件路径，存入filelist列表
        with open(filepath, 'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()
        
        # 构造不同数据类型的根路径
        self.imcomplete_path = os.path.join(data_path, 'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path, 'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path, 'ShapeNetViPC-View')
        

        gen_root = os.path.join(data_path, 'ShapeNetViPC-Gen')
        trellis_gen_path = os.path.join(gen_root, 'trellis', sampling_method, f'num_points_{num_points}')
        if os.path.exists(trellis_gen_path):
            self.gen_path = trellis_gen_path
        else:
            raise FileNotFoundError(f"Trellis gen point cloud path not found: {trellis_gen_path}")

        # 根据传入的数据类别筛选文件
        for key in self.filelist:
            if category != 'all':
                # 如果当前文件所属类别与指定类别不匹配，则跳过
                if key.split('/')[0] != self.cat_map[category]:
                    continue
            # 记录类别和键值
            self.cat.append(key.split('/')[0])
            self.key.append(key)

        # 定义图像的预处理操作：缩放到224像素大小并转换为Tensor
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.key)}')
        print(f'Gen point cloud path: {self.gen_path}')

    def __getitem__(self, idx):
        # 根据索引获取当前样本对应的文件键
        key = self.key[idx]
        # 构造部分点云数据文件的路径，并添加文件后缀"dat"
        pc_part_path = os.path.join(self.imcomplete_path, key.replace('\n', '')) + '.dat'
        
        # 根据view_align参数确定选择哪个文件名作为参考
        if self.view_align:
            # 如果视图与部分点云要求对齐，则直接使用key
            ran_key = key        
        else:
            # 否则，随机选择一个视图，替换文件名后三位中的数字（视角编号）
            ran_key = key[:-3] + str(random.randint(0, 23)).rjust(2, '0')
            # 如果os.path.join(self.gen_path, ran_key.replace('\n', '')) + '.pt'不存在，则重新选择
            while not os.path.exists(os.path.join(self.gen_path, ran_key.replace('\n', '')) + '.pt'):
                # print(f"Warning: Gen point cloud not found for {ran_key}. Re-selecting view.")
                ran_key = key[:-3] + str(random.randint(0, 23)).rjust(2, '0')
        
        # 构造完整点云数据文件的路径
        pc_path = os.path.join(self.gt_path, ran_key.replace('\n', '')) + '.dat'
        # 构造生成点云数据文件的路径
        gen_pc_path = os.path.join(self.gen_path, ran_key.replace('\n', '')) + '.pt'
        # 构造渲染视图图像的路径，目录结构包含类别、子类别及渲染文件夹
        view_path = os.path.join(self.rendering_path, 
                                 ran_key.split('/')[0] + '/' + 
                                 ran_key.split('/')[1] + '/rendering/' + 
                                 ran_key.split('/')[-1].replace('\n', '') + '.png')

        # 载入视图图像文件，并通过预处理转换为Tensor
        views = self.transform(Image.open(view_path))
        views = views[:3, :, :]  # 截取前三个通道（通常为RGB）

        # 载入完整点云数据
        with open(pc_path, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)
            
        # 载入生成点云数据
        gen_pc = torch.load(gen_pc_path, weights_only=True)

            
        # 载入部分点云数据
        with open(pc_part_path, 'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        
        # 如果部分点云的点数小于预设数量，则重复点集合以达到所需数量
        if pc_part.shape[0] < self.pc_input_num:
            pc_part = np.repeat(pc_part, (self.pc_input_num // pc_part.shape[0]) + 1, axis=0)[0:self.pc_input_num]
        
        # 提取视图和部分点云对应的视角编号，用于获取元数据
        image_view_id = view_path.split('.')[0].split('/')[-1]
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        # 载入描述视角信息的元数据文件（包含Theta和Phi角度）
        view_metadata = np.loadtxt(view_path[:-6] + 'rendering_metadata.txt')

        # 获取部分点云视角对应的Theta和Phi角，并转换为弧度制
        theta_part = math.radians(view_metadata[int(part_view_id), 0])
        phi_part = math.radians(view_metadata[int(part_view_id), 1])

        # 获取视图图像对应的Theta和Phi角，并转换为弧度制
        theta_img = math.radians(view_metadata[int(image_view_id), 0])
        phi_img = math.radians(view_metadata[int(image_view_id), 1])

        # 先对部分点云进行旋转调整，使其初步对齐：
        # 先绕X轴旋转（负的phi_part），再绕Y轴旋转（加上π和theta_part）
        pc_part = rotation_y(rotation_x(pc_part, -phi_part), np.pi + theta_part)
        # 再依据视图图像的角度分别绕Y轴和X轴旋转做进一步调整
        pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

        # 对完整点云进行归一化处理：
        # 计算完整点云的质心（平均值），并用以平移点云
        gt_mean = pc.mean(axis=0)
        pc = pc - gt_mean
        # 计算归一化因子：所有点到质心的最大欧氏距离
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        # 使用归一化因子归一化完整点云
        pc = pc / pc_L_max

        # 同样对部分点云做平移和归一化处理
        pc_part = pc_part - gt_mean
        pc_part = pc_part / pc_L_max
        
        # 对生成点云单独做平移和归一化处理
        if self.normalize_gen:
            gen_mean   = gen_pc.mean(dim=0, keepdim=True)   # [1,3]
            gen_L_max  = torch.norm(gen_pc - gen_mean, dim=1).max()  # 标量
            gen_pc     = (gen_pc - gen_mean) / gen_L_max

        # 为了与ViPCDataLoader保持一致的接口，添加任务ID
        task_id = ran_key.replace('\n', '')
        
        # 返回视图、完整点云、生成点云和部分点云，均转为Tensor格式
        return task_id, views.float(), torch.from_numpy(pc).float(), gen_pc, torch.from_numpy(pc_part).float()

    def __len__(self):
        # 返回数据集中样本的总数量
        return len(self.key)


# 数据加载器类，继承自torch.utils.data.Dataset
class ViPCDataLoader(Dataset):
    def __init__(self,filepath,data_path,status,pc_input_num=3500,view_align=False,category='all'):
        super(ViPCDataLoader,self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'plane':'02691156',
            'bench': '02828884',
            'cabinet':'02933112',
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459',
            'firearm': '04090263',
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088',
            'watercraft':'04530566'
        }
        with open(filepath,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()
        
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')

        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split('/')[0])
            self.key.append(key)

        self.transform = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.key)}')


    def __getitem__(self, idx):
        key = self.key[idx]
        pc_part_path = os.path.join(self.imcomplete_path,key.replace('\n',''))+'.dat'
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')
        pc_path = os.path.join(self.gt_path,ran_key.replace('\n',''))+'.dat'
        view_path = os.path.join(self.rendering_path,ran_key.split('/')[0]+'/'+ran_key.split('/')[1]+'/rendering/'+ran_key.split('/')[-1].replace('\n','')+'.png')

        # load view 
        views = self.transform(Image.open(view_path))
        views = views[:3,:,:]
        # load partial points
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        # load gt
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        # incase some item point number less than 3500 
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]
        # load the view metadata
        image_view_id = view_path.split('.')[0].split('/')[-1]
        part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')

        theta_part = math.radians(view_metadata[int(part_view_id),0])
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = rotation_y(rotation_x(pc_part, - phi_part),np.pi + theta_part)
        pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float()

    def __len__(self):
        return len(self.key)




def preprocess_image(input: Image.Image) -> Image.Image:
    """
    Preprocess the input image.
    
    处理逻辑：
        1. 如果图像有 Alpha 通道且包含透明像素，则直接使用原图。
        2. 如果没有 Alpha 通道或 Alpha 通道没有透明像素，则进行背景去除：
            - 先转换为 RGB 模式
            - 根据最大尺寸调整图像缩放比例，限制最大边长为 1024 像素
            - 使用 rembg 进行背景去除
        3. 计算非透明区域的边界框（BBox），基于该框进行裁剪，使主体尽可能居中。
        4. 将裁剪后的图像调整到 518x518 像素，并归一化 Alpha 通道的影响，使主体透明区域保留。
        5. 返回最终处理后的图像。
    """
    # if has alpha channel, use it directly; otherwise, remove background
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if has_alpha:
        output = input
    else:
        # convert to RGB and remove background
        input = input.convert('RGB')
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if getattr(self, 'rembg_session', None) is None:
            self.rembg_session = rembg.new_session('u2net')  # remove background using u2net
        output = rembg.remove(input, session=self.rembg_session)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output

def custom_collate(batch):
    """
    自定义collate函数，处理任务ID、图像和部分点云
    """
    task_ids = [item[0] for item in batch]  # 任务ID列表
    
    # 处理图像列表，图像可能为None
    if batch[0][1] is None:  # 检查第一个样本的图像是否为None
        images = None
    else:
        images = [item[1] for item in batch]    # PIL图像列表
    
    # 堆叠GT点云、生成点云和部分点云    
    pcs_gt = torch.stack([item[2] for item in batch])
    pcs_gen = torch.stack([item[3] for item in batch])
    pc_parts = torch.stack([item[4] for item in batch])
    
    return task_ids, images, pcs_gt, pcs_gen, pc_parts


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ViPCDataset = ViPCDataLoader('train_list.txt',
                                data_path='/home/ubuntu/data/projects/TRELLIS/data/ShapeNetViPC/ShapeNetViPC-Dataset',
                                status='train', view_align=False, category='all')
    train_loader = DataLoader(ViPCDataset,
                              batch_size=16 ,
                              num_workers=1,
                              shuffle=False,
                              drop_last=True,
                              collate_fn=custom_collate)
    # for batch, (views, gt, partial) in enumerate(tqdm(train_loader)):
    #     img = views[0]
    #     img.save(f'view{batch}.png')
    #     preprocessed_img = preprocess_image(img)
    #     preprocess_image.save(f'preprocessed_view{batch}.png')
