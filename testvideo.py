import cv2
import torch
import time
import numpy as np
import os
from options.test_options import TestOptions
from models import create_model


class VideoDehazeInference:
    def __init__(self, opt):
        self.opt = opt
        self.opt.num_threads = 0
        self.opt.batch_size = 1
        self.opt.serial_batches = True
        self.opt.no_flip = True
        self.opt.isTrain = False

        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()

        gpu_id = self.opt.gpu_ids[0] if isinstance(self.opt.gpu_ids, list) else 0
        self.device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    def pre_process(self, frame_bgr):
        """ 预处理：保持原始尺寸（调整为4的倍数） """
        h, w = frame_bgr.shape[:2]

        # 确保尺寸是 4 的倍数 (模型通常有 2 层下采样)
        # 如果模型层数更多，建议改为 8 或 16 的倍数
        new_w = (w // 4) * 4
        new_h = (h // 4) * 4

        # 如果尺寸需要调整，则进行微调缩放
        if new_w != w or new_h != h:
            frame_input = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            frame_input = frame_bgr

        img_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        # 归一化 [-1, 1]
        img_data = (img_rgb.astype(np.float32) / 127.5) - 1.0
        img_tensor = torch.from_numpy(img_data).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return img_tensor, (new_w, new_h)

    @torch.no_grad()
    def run_on_video(self, input_path, output_path=None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: 无法打开视频文件 {input_path}")
            return

        # 获取原视频属性
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 输出视频保持原视频完全一致的尺寸
            writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
            print(f"保存视频至: {output_path} | 分辨率: {orig_w}x{orig_h}")

        print(f"开始全尺寸处理... (共 {total_frames} 帧)")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            start_t = time.time()

            # 1. 预处理 (不改变大尺寸，仅做对齐)
            input_tensor, aligned_size = self.pre_process(frame)

            # 2. 推理
            self.model.set_input({'A': input_tensor, 'B': input_tensor,
                                  'A_paths': ['video'], 'B_paths': ['video']})
            self.model.forward()

            # 3. 后处理
            output = self.model.fake_B[0].detach()
            output = (output + 1.0) / 2.0 * 255.0
            dehazed_rgb = output.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

            # 4. 转换回 BGR 并还原到原始视频精确尺寸（防止对齐带来的几个像素偏差）
            res_bgr = cv2.cvtColor(dehazed_rgb, cv2.COLOR_RGB2BGR)
            if aligned_size[0] != orig_w or aligned_size[1] != orig_h:
                res_bgr = cv2.resize(res_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            # 5. 写入文件
            if writer:
                writer.write(res_bgr)

            # 6. 显示进度
            frame_count += 1
            elapsed = time.time() - start_t
            if frame_count % 1 == 0:
                print(f"Progress: {frame_count}/{total_frames} | Frame Time: {elapsed:.3f}s", end='\r')

            # 预览：由于原图可能很大，预览时缩小显示
            preview_frame = cv2.resize(res_bgr, (orig_w // 2, orig_h // 2))
            cv2.imshow('Dehazing (Full Size Processing)', preview_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print(f"\n处理完成！")


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.gpu_ids = [0]

    input_video = r"detect.mp4"
    output_video = r"detect_dehazed.mp4"

    if os.path.exists(input_video):
        app = VideoDehazeInference(opt)
        app.run_on_video(input_video, output_video)
    else:
        print(f"错误：找不到文件 {input_video}")