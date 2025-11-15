import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch.multiprocessing as mp
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import imageio

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils.point_cloud import sample_mesh_to_pointcloud
from trellis.renderers.pc_render import point_cloud_render
from trellis.utils import render_utils

# 设置环境变量
os.environ['SPCONV_ALGO'] = 'native'  # 速度更快的设置

# 创建全局线程池，用于点云采样
SAMPLING_THREAD_POOL = None
SAMPLING_FUTURES = []
SAMPLING_LOCK = threading.Lock()


def init_sampling_thread_pool(max_workers=16):
    """初始化全局采样线程池"""
    global SAMPLING_THREAD_POOL
    if SAMPLING_THREAD_POOL is None:
        SAMPLING_THREAD_POOL = ThreadPoolExecutor(max_workers=max_workers)
    return SAMPLING_THREAD_POOL

def sample_mesh_to_pc(mesh, save_path, sampling_method="poisson_disk", num_points=16384, visualize=False):
    """在单独的线程中采样点云并保存"""
    try:
        # 将mesh转换为点云
        points = sample_mesh_to_pointcloud(
            mesh,
            num_points,
            sampling_method
        )
        
        # 将点云转换为tensor并保存
        points_tensor = torch.from_numpy(points).float()
        torch.save(points_tensor, save_path)
        # print(f"已保存点云: {save_path}")
        
        # 如果启用可视化，渲染点云图像
        if visualize:
            try:
                # 提取类别ID和对象ID从保存路径
                path_parts = Path(save_path).parts
                if len(path_parts) >= 3:
                    # 找到类别ID和对象ID在路径中的位置
                    category_idx = -3 if len(path_parts) >= 3 else -2
                    object_idx = -2 if len(path_parts) >= 3 else -1
                    category_id = path_parts[category_idx]
                    object_id = path_parts[object_idx]
                    view_id = Path(save_path).stem
                else:
                    # 如果路径结构不符合预期，使用文件名作为标识
                    category_id = "unknown"
                    object_id = "unknown"
                    view_id = Path(save_path).stem
                
                # 构造可视化输出路径，使用绝对路径
                viz_dir = os.path.join(os.getcwd(), '.output', 'point_cloud_viz', category_id, object_id)
                os.makedirs(viz_dir, exist_ok=True)
                
                # 构造PNG文件名
                pc_png_path = os.path.join(viz_dir, f"{view_id}.png")
                
                # 渲染并保存点云图像
                point_cloud_render(points_tensor, pc_png_path)
                # print(f"已保存点云可视化: {pc_png_path}")
            except Exception as e:
                print(f"保存点云可视化失败: {str(e)}")
        
        return True, save_path
    except Exception as e:
        print(f"采样点云失败 {save_path}: {str(e)}")
        return False, save_path

def load_image(view_path):
    """加载图像函数"""
    try:
        img = Image.open(view_path)
        img.load()  # 立即加载图像数据到内存
        return img
    except Exception as e:
        return None

def process_image_file(pipeline, image, save_path, view_path=None, failed_path=None, seed=42, 
                      sampling_method="poisson_disk", num_points=16384, visualize=False):
    """处理单个图像文件，生成mesh并直接采样为点云"""
    try:
        # 确保采样线程池已初始化
        init_sampling_thread_pool()
        
        # 确保图像已加载
        if image is None:
            if view_path:  # 尝试重新加载
                image = Image.open(view_path)
            else:
                return False
        
        # 构造点云保存路径
        pc_save_path = save_path.replace('.dat', '.pt')
        
        # 如果点云文件已存在，跳过处理
        if os.path.exists(pc_save_path):
            print(f"点云文件已存在: {pc_save_path}")
            return True
        
        # 运行pipeline生成mesh
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=['mesh'],  # 只生成mesh格式
            preprocess_image=True,
            sparse_structure_sampler_params={
                "steps": 17,
                "cfg_strength": 9,
            },
            slat_sampler_params={
                "steps": 13, 
                "cfg_strength": 5.5,
            },

        )
        mesh = outputs['mesh'][0]
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(pc_save_path), exist_ok=True)
        
        # 如果启用可视化，渲染mesh视频
        if visualize:
            try:
                # 提取类别ID和对象ID从保存路径
                path_parts = Path(pc_save_path).parts
                if len(path_parts) >= 3:
                    # 找到类别ID和对象ID在路径中的位置
                    category_idx = -3 if len(path_parts) >= 3 else -2
                    object_idx = -2 if len(path_parts) >= 3 else -1
                    category_id = path_parts[category_idx]
                    object_id = path_parts[object_idx]
                    view_id = Path(pc_save_path).stem
                else:
                    # 如果路径结构不符合预期，使用文件名作为标识
                    category_id = "unknown"
                    object_id = "unknown"
                    view_id = Path(pc_save_path).stem
                
                # 构造可视化输出路径，使用绝对路径
                viz_dir = os.path.join(os.getcwd(), '.output', 'mesh_viz', category_id, object_id)
                os.makedirs(viz_dir, exist_ok=True)
                
                # 构造视频文件名
                mesh_mp4_path = os.path.join(viz_dir, f"{view_id}.mp4")
                
                # 尝试使用较低的参数渲染mesh视频，避免内存问题
                print(f"开始渲染mesh视频到: {mesh_mp4_path}")
                try:
                    # 降低渲染参数减少内存占用
                    video = render_utils.render_video(
                        mesh, 
                        n_frames=20,     # 减少帧数
                        resolution=256   # 降低分辨率
                    )['normal']

                    imageio.mimsave(mesh_mp4_path, video, fps=20)
                    print(f"已保存mesh视频: {mesh_mp4_path}")
                except KeyboardInterrupt:
                    # 捕获键盘中断避免整个程序中断
                    print(f"渲染mesh视频被用户中断，跳过此视频")
                except Exception as e:
                    print(f"渲染mesh视频失败: {str(e)}")
                    # 尝试渲染单帧作为备选
                    try:
                        single_frame_path = mesh_mp4_path.replace('.mp4', '.png')
                        single_frame = render_utils.render_mesh(mesh)['normal']
                        imageio.imsave(single_frame_path, single_frame)
                        print(f"已保存mesh单帧图像: {single_frame_path}")
                    except Exception as inner_e:
                        print(f"渲染mesh单帧图像也失败: {str(inner_e)}")
            except Exception as e:
                print(f"准备mesh可视化失败: {str(e)}")
        
        # 异步提交点云采样任务到线程池
        with SAMPLING_LOCK:
            future = SAMPLING_THREAD_POOL.submit(
                sample_mesh_to_pc, 
                mesh, 
                pc_save_path,
                sampling_method,
                num_points,
                visualize
            )
            SAMPLING_FUTURES.append(future)
            
            # 清理已完成的采样任务
            completed = [f for f in SAMPLING_FUTURES if f.done()]
            for f in completed:
                SAMPLING_FUTURES.remove(f)
                
            # print(f"提交点云采样任务: {pc_save_path}，当前队列中任务数: {len(SAMPLING_FUTURES)}")
        
        # 不等待点云采样完成，直接返回成功
        return True
    except Exception as e:
        # 记录失败文件
        if failed_path and view_path:
            try:
                with open(failed_path, 'a') as f:
                    f.write(f"{view_path}\n")
                # 写入错误信息
                with open(failed_path.replace('.txt', '_error.txt'), 'a') as f:
                    f.write(f"{view_path}\n{str(e)}\n")
            except:
                pass
        return False

def wait_for_sampling_tasks():
    """等待所有点云采样任务完成"""
    global SAMPLING_FUTURES
    
    if not SAMPLING_FUTURES:
        return
        
    print(f"等待 {len(SAMPLING_FUTURES)} 个点云采样任务完成...")
    
    # 创建进度条
    with tqdm(total=len(SAMPLING_FUTURES), desc="点云采样") as pbar:
        remaining = list(SAMPLING_FUTURES)
        while remaining:
            completed = [f for f in remaining if f.done()]
            for f in completed:
                remaining.remove(f)
                pbar.update(1)
            
            if remaining:
                time.sleep(0.01)
    
    # 清空完成的任务
    SAMPLING_FUTURES = []
    print("所有点云采样任务已完成")

class DataPreloader:
    """图像预加载器，使用线程池预加载图像到队列中"""
    def __init__(self, max_size=10, num_loader_threads=4, output_dir=None):
        self.queue = queue.Queue(maxsize=max_size)
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=num_loader_threads)
        self.futures = []
        self.output_dir = output_dir
        self.processed_files = set()  # 用于跟踪已处理的文件
        
    def add_processed_file(self, key):
        """添加已处理文件记录"""
        self.processed_files.add(key)
    
    def preload_images(self, file_list, data_path):
        """预加载一批图像"""
        print(f"[Preload] Starting preload_images for {len(file_list)} files.") # Log start
        batch_size = 10  # 每次检查的文件数量
        futures_limit = 50  # 限制最大未完成任务数量
        
        # 分批处理文件
        for i in range(0, len(file_list), batch_size):
            if not self.running:
                print("[Preload] Detected not running, exiting outer loop.") # Log exit
                break
                
            batch = file_list[i:i+batch_size]
            print(f"[Preload] Processing batch {i//batch_size + 1}, size {len(batch)}.") # Log batch start
            
            for key in batch:
                # print(f"[Preload] Checking key: {key}") # Optional: Log every key check
                if not self.running:
                    print("[Preload] Detected not running, exiting inner loop.") # Log exit
                    break # Exit inner loop
                
                if key in self.processed_files:
                    # print(f"[Preload] Key {key} already processed, skipping.") # Optional: Log skip
                    continue
                    
                try:
                    # 构造视图图像的路径
                    view_path = os.path.join(
                        data_path, 
                        'ShapeNetViPC-View', 
                        key.split('/')[0],  # 类别
                        key.split('/')[1],  # 对象ID
                        'rendering', 
                        key.split('/')[-1].replace('\n', '') + '.png'  # 视角ID
                    )
                    
                    # 构造保存路径
                    save_path = os.path.join(self.output_dir, key.replace('\n', '') + '.pt' )
        
                    # 限制未完成任务的数量，避免内存问题
                    while len(self.futures) >= futures_limit and self.running:
                        # print("[Preload] Futures limit reached, checking completed...") # Optional: Log waiting
                        self._check_completed_futures()
                        time.sleep(0.005)
                        
                    if not self.running: # Re-check after potential wait
                         print("[Preload] Detected not running after future limit wait, exiting inner loop.")
                         break
                         
                    # 提交加载任务到线程池
                    # print(f"[Preload] Submitting load task for key: {key} | view_path: {view_path}") # Log before submit
                    future = self.executor.submit(load_image, view_path)
                    self.futures.append((future, view_path, save_path, key))
                    # print(f"[Preload] Submitted task for key: {key}. Total futures: {len(self.futures)}") # Log after submit
                    
                except Exception as e:
                    print(f"[Preload] Exception during task setup/submission for key {key}: {str(e)}") # Log exception
            
            # If inner loop was broken, break outer loop too
            if not self.running:
                break
                
            # 每批次结束后检查已完成的任务
            # print("[Preload] Checking completed futures after batch.") # Optional: Log check
            self._check_completed_futures()
            
            # 如果队列较满，暂停添加新任务
            if self.queue.qsize() > self.queue.maxsize * 0.8:
                # print("[Preload] Queue filling up, pausing...") # Optional: Log pause
                while self.queue.qsize() > self.queue.maxsize * 0.5 and self.running:
                    self._check_completed_futures()
                    time.sleep(0.05)
        
        print("[Preload] Finished preload_images loop.") # Log end
    
    def _check_completed_futures(self):
        """检查已完成的Future并将结果放入队列"""
        completed = []
        for i, (future, view_path, save_path, key) in enumerate(self.futures):
            if future.done():
                completed.append(i)
                try:
                    image = future.result()
                    # 如果成功加载，则放入队列
                    if image is not None:
                        # 阻塞直到队列有空间
                        while self.running:
                            try:
                                self.queue.put((image, view_path, save_path, key), timeout=0.1)
                                break
                            except queue.Full:
                                if not self.running:
                                    break
                                time.sleep(0.01)
                except Exception as e:
                    print(f"加载图像失败 {view_path}: {str(e)}")
        
        # 从futures列表中移除已完成的任务
        for idx in reversed(completed):
            self.futures.pop(idx)
    
    def get_item(self, timeout=None):
        """从队列获取预加载的图像"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def shutdown(self):
        """关闭预加载器"""
        self.running = False
        self.executor.shutdown(wait=False)
        # 清空队列
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                pass

def process_dataset(data_path, tasks, output_dir, gpu_id, worker_id, num_workers, 
                 prefetch_size=20, loader_threads=4, sampling_threads=16, num_points=16384, 
                 sampling_method="poisson_disk", visualize=False):
    """处理整个数据集"""
    # 初始化点云采样线程池
    init_sampling_thread_pool(max_workers=sampling_threads)
    print(f"Worker {worker_id} on GPU {gpu_id}: 初始化点云采样线程池，线程数: {sampling_threads}")
    
    # 设置GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    torch.backends.cudnn.benchmark = True  # 优化性能
    
    # 创建失败文件记录路径
    failed_path = os.path.join(output_dir, f"failed_files_worker{worker_id}_gpu{gpu_id}.txt")
    
    # 错开启动时间，避免资源冲突
    time.sleep(worker_id * 3)
    
    # 加载预训练模型
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.to(device)
    
    # 分配给当前worker的文件
    worker_files = tasks[worker_id::num_workers]
    
    # 创建数据预加载器
    preloader = DataPreloader(max_size=prefetch_size, num_loader_threads=loader_threads, output_dir=output_dir)
    
    # 启动预加载线程
    preload_thread = threading.Thread(
        target=preloader.preload_images, 
        args=(worker_files, data_path)
    )
    preload_thread.daemon = True
    preload_thread.start()
    
    # 处理每个文件
    success_count = 0
    failed_count = 0
    
    # 初始化进度条
    pbar = tqdm(total=len(worker_files), desc=f"Worker {worker_id}")
    
    # 处理计数和同步时间
    processed_count = 0
    last_sync_time = time.time()
    
    # # 等待队列中有数据
    # time.sleep(0.5)
    
    # GPU处理循环
    while processed_count < len(worker_files):
        # 从队列获取预加载的图像
        item = preloader.get_item(timeout=0.1)
        
        if item is None:
            # 队列为空，检查是否应退出
            # 退出条件：预加载线程结束 AND 预加载队列为空 AND 没有待处理的future
            if not preload_thread.is_alive() and preloader.queue.empty() and not preloader.futures:
                print(f"Worker {worker_id}: 预加载完成，队列为空，且无待处理 futures，结束处理")
                break
                
            # 定期同步和清理内存 (可以保留，但可能不是主要问题)
            current_time = time.time()
            if current_time - last_sync_time > 30:
                print(f"Worker {worker_id}: 30秒内无新数据，但仍有 futures 或线程运行中。同步GPU并清理内存。 Futures: {len(preloader.futures)}")
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                last_sync_time = current_time
                
            # 主动检查一次 futures 完成情况，以防 _check_completed_futures 调用不够频繁
            preloader._check_completed_futures()
            time.sleep(0.01) # 短暂暂停避免忙等
            continue
        
        # 更新同步时间
        last_sync_time = time.time()
        
        # 解包数据
        image, view_path, save_path, key = item
        
        # 记录正在处理的文件
        preloader.add_processed_file(key)
        
        # 处理图像
        if process_image_file(pipeline, image, save_path, view_path, failed_path, 
                            sampling_method=sampling_method, num_points=num_points, visualize=visualize):
            success_count += 1
        else:
            failed_count += 1
        
        # 更新进度条
        processed_count += 1
        pbar.update(1)
        
        # 每20个文件更新一次进度条后缀
        if processed_count % 20 == 0:
            gpu_util = torch.cuda.utilization(device)
            pbar.set_postfix({
                "成功": success_count, 
                "失败": failed_count, 
                "队列": preloader.queue.qsize(),
                "GPU利用率": f"{gpu_util}%",
                "采样任务": len(SAMPLING_FUTURES)
            })
        
        # 定期同步和内存管理
        if processed_count % 1000 == 0:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    
    # 关闭进度条和预加载器
    pbar.close()
    preloader.shutdown()
    
    # 等待点云采样任务完成
    print(f"Worker {worker_id} on GPU {gpu_id}: 等待点云采样任务完成...")
    wait_for_sampling_tasks()
    
    print(f"Worker {worker_id} on GPU {gpu_id}: 处理完成，成功: {success_count}, 失败: {failed_count}")

def main():
    parser = argparse.ArgumentParser(description='为ShapeNetViPC数据集生成点云')
    parser.add_argument('--data_path', type=str, default='./data/ShapeNetViPC-Dataset', 
                        help='ShapeNetViPC数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./data/ShapeNetViPC-Dataset/ShapeNetViPC-Gen/trellis',
                        help='生成的点云保存目录')  
    parser.add_argument('--categories', type=str, default='', 
                        help='要处理的类别，用逗号分隔（例如：plane,bench）。为空时处理所有类别')
    parser.add_argument('--gpu_ids', type=str, default='0', help='使用的GPU ID，多个用逗号分隔')
    parser.add_argument('--num_workers_per_gpu', type=int, default=1, help='每个GPU的worker数量')
    parser.add_argument('--prefetch_size', type=int, default=20, help='预取队列大小')
    parser.add_argument('--loader_threads', type=int, default=5, help='每个worker的图像加载线程数')
    parser.add_argument('--sampling_threads', type=int, default=10, help='点云采样线程数')
    parser.add_argument('--num_points', type=int, default=2048, help='采样点云的点数')
    parser.add_argument('--sampling_method', type=str, default='poisson_disk', 
                        choices=['poisson_disk', 'uniform', 'random'], help='点云采样方法')
    parser.add_argument('--no_warnings', action='store_true', help='禁止警告消息')
    parser.add_argument('--visualize', action='store_true', help='启用可视化，保存点云PNG图像和mesh MP4视频到.output目录')
    parser.add_argument('--max_visualize', type=int, default=100, help='最大可视化文件数量，避免生成过多图片和视频')
    args = parser.parse_args()
    
    # 禁止警告输出
    if args.no_warnings:
        import warnings
        warnings.filterwarnings("ignore")
    
    # 确保输出目录存在
    args.output_dir = os.path.join(args.output_dir, args.sampling_method, "num_points_" + str(args.num_points))
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 类别名称与对应ID的映射字典
    CAT_MAP = {
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

    # 解析要处理的类别
    categories = []
    if args.categories:
        categories = [cat.strip() for cat in args.categories.split(',')]
        # 验证类别是否有效
        for cat in categories:
            if cat not in CAT_MAP:
                print(f"警告: 未知类别 '{cat}'")
                categories.remove(cat)
    else:
        # 如果未指定类别，处理所有类别
        categories = list(CAT_MAP.keys())
    
    if not categories:
        print("错误: 没有有效的类别可处理")
        return
    
    # 根据类别构建要处理的文件列表
    tasks = []
    # 记录每个类别的统计信息
    category_stats = {}
    
    for cat in categories:
        cat_id = CAT_MAP[cat]
        cat_dir = os.path.join(args.data_path, "ShapeNetViPC-View", cat_id)
        
        if not os.path.exists(cat_dir):
            print(f"警告: 类别 '{cat}' (ID: {cat_id}) 的目录不存在: {cat_dir}")
            continue
        
        # 初始化该类别的统计信息
        category_stats[cat] = {
            'total': 0,
            'existing': 0
        }
        
        # 遍历类别目录下的所有模型文件
        for model_dir in os.listdir(cat_dir):
            model_path = os.path.join(cat_dir, model_dir)
            # Find all rendering views (PNG files) for this model
            rendering_dir = os.path.join(model_path, 'rendering')
            if os.path.exists(rendering_dir):
                # Get all PNG files in the rendering directory
                for view_file in os.listdir(rendering_dir):
                    if view_file.endswith('.png'):
                        # Extract view ID (without extension)
                        view_id = os.path.splitext(view_file)[0]
                        
                        # Create the task identifier in the format expected by the rest of the code
                        # Format: cat_id/model_id/view_id
                        task_id = f"{cat_id}/{model_dir}/{view_id}"
                        
                        # 增加该类别的总计数
                        category_stats[cat]['total'] += 1
                        
                        # 检查对应的点云文件是否已经存在
                        pc_save_path = os.path.join(args.output_dir, task_id + '.pt')
                        if os.path.exists(pc_save_path):
                            # 增加该类别已存在的文件计数
                            category_stats[cat]['existing'] += 1
                        else:
                            tasks.append(task_id)
            else:
                print(f"警告: 渲染目录不存在: {rendering_dir}")
    
    # 打印每类模型的已存在点云文件的百分比
    print("\n各类别点云文件统计:")
    print("-" * 50)
    print(f"{'类别':15} {'总数':10} {'已处理':10} {'百分比':10}")
    print("-" * 50)
    
    total_files = 0
    total_existing = 0
    
    for cat, stats in category_stats.items():
        total = stats['total']
        existing = stats['existing']
        percentage = (existing / total * 100) if total > 0 else 0
        
        total_files += total
        total_existing += existing
        
        print(f"{cat:15} {total:10} {existing:10} {percentage:10.2f}%")
    
    # 打印总计
    overall_percentage = (total_existing / total_files * 100) if total_files > 0 else 0
    print("-" * 50)
    print(f"{'总计':15} {total_files:10} {total_existing:10} {overall_percentage:10.2f}%")
    print("-" * 50)
    
    print(f"找到 {len(tasks)} 个模型需要处理")
    
    
    print("PC save dir:", args.output_dir)
    
    # 如果启用可视化，确保可视化输出目录存在
    if args.visualize:
        output_base = os.path.join(os.getcwd(), '.output')
        os.makedirs(output_base, exist_ok=True)
        os.makedirs(os.path.join(output_base, 'point_cloud_viz'), exist_ok=True)
        os.makedirs(os.path.join(output_base, 'mesh_viz'), exist_ok=True)
        print(f"已启用可视化，最多可视化 {args.max_visualize} 个文件")
        print(f"可视化输出将保存到: {output_base}/point_cloud_viz 和 {output_base}/mesh_viz")
    
    # 清除可能影响性能的环境变量
    for env_var in ["CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING"]:
        if env_var in os.environ:
            del os.environ[env_var]
    
    # 设置PyTorch性能选项
    torch.backends.cudnn.benchmark = True
    
    # 解析GPU ID
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    
    # 验证GPU
    valid_gpus = []
    for gpu_id in gpu_ids:
        if gpu_id < torch.cuda.device_count():
            valid_gpus.append(gpu_id)
    
    if not valid_gpus:
        print("错误: 没有有效的GPU可用")
        return
        
    # 按GPU利用率排序
    try:
        gpu_utils = []
        for gpu_id in valid_gpus:
            try:
                util = torch.cuda.utilization(gpu_id)
                gpu_utils.append((gpu_id, util))
            except:
                gpu_utils.append((gpu_id, 100))
        
        gpu_utils.sort(key=lambda x: x[1])
        valid_gpus = [gpu_id for gpu_id, _ in gpu_utils]
        
        print(f"按利用率排序后的GPU: {[(gpu_id, util) for gpu_id, util in gpu_utils]}")
    except:
        pass
        
    # 更新GPU列表和总worker数
    gpu_ids = valid_gpus
    num_gpus = len(gpu_ids)
    total_workers = num_gpus * args.num_workers_per_gpu
    
    print(f"启动处理 - 使用 {total_workers} 个worker, 每个worker有 {args.loader_threads} 个加载线程")
    print(f"预取队列大小: {args.prefetch_size}, 使用GPU: {gpu_ids}")
    print(f"点云采样: 线程数 {args.sampling_threads}, 点数 {args.num_points}, 采样方法: {args.sampling_method}")
    
    # 多进程处理
    if total_workers > 1:
        # 确保使用spawn方法启动进程
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        processes = []
        
        # 为每个worker分配GPU
        for i in range(total_workers):
            gpu_id = gpu_ids[i % num_gpus]
            
            # 启动子进程
            p = mp.Process(
                target=process_dataset,
                args=(args.data_path, tasks, args.output_dir, 
                     gpu_id, i, total_workers, args.prefetch_size, args.loader_threads,
                     args.sampling_threads, args.num_points, args.sampling_method, 
                     args.visualize and i == 0)  # 只在第一个进程中启用可视化
            )
            p.daemon = True
            p.start()
            processes.append(p)
            
            # 给每个进程初始化时间
            time.sleep(5)
        
        # 等待所有进程完成
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                if p.is_alive():
                    try:
                        p.terminate()
                    except:
                        pass
    else:
        # 单进程处理
        process_dataset(args.data_path, tasks, args.output_dir, gpu_ids[0], 0, 1, 
                       args.prefetch_size, args.loader_threads, args.sampling_threads, args.num_points, 
                       args.sampling_method, args.visualize)
    
    print("所有处理完成!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc() 