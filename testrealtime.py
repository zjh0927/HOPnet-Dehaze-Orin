import cv2
import torch
import time
import numpy as np
from options.test_options import TestOptions
from models import create_model


class CameraDehazeInference:
    def __init__(self, opt):
        self.opt = opt
        # 强制工程推理参数
        self.opt.num_threads = 0
        self.opt.batch_size = 1
        self.opt.serial_batches = True
        self.opt.no_flip = True
        self.opt.isTrain = False

        # 初始化模型
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()

        # 设备配置
        gpu_id = self.opt.gpu_ids[0] if isinstance(self.opt.gpu_ids, list) else 0
        self.device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

        self.window_size = 5
        # 建议设置为 512 或 256 以保证实时性
        self.inference_size = 640

    def pre_process(self, frame_bgr):
        """ 针对实时流优化的预处理 """
        # 缩放到指定推理分辨率
        frame_resized = cv2.resize(frame_bgr, (self.inference_size, self.inference_size),
                                   interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # 归一化 [-1, 1]
        img_data = (img_rgb.astype(np.float32) / 127.5) - 1.0
        img_tensor = torch.from_numpy(img_data).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return img_tensor, img_rgb

    def calculate_metrics(self, hazy_rgb, dehazed_rgb):
        """ 计算对比度增益 (Gain) """

        def get_contrast(img_rgb):
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            mean = cv2.blur(gray, (self.window_size, self.window_size))
            sq_mean = cv2.blur(gray ** 2, (self.window_size, self.window_size))
            variance = np.maximum(0, sq_mean - mean ** 2)
            return np.mean(variance / (mean + 1e-5))

        return get_contrast(dehazed_rgb) - get_contrast(hazy_rgb)

    @torch.no_grad()
    def run(self):
        # 调用默认摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: 无法打开摄像头")
            return

        # 优化摄像头读取性能
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"实时推理启动 (分辨率: {self.inference_size})。按 'Q' 键退出...")

        while True:
            ret, frame = cap.read()
            if not ret: break

            start_t = time.time()

            # 1. 预处理
            input_tensor, hazy_rgb_small = self.pre_process(frame)
            # 2. 推理
            # 为了兼容HOPnet_model的set_input方法，提供一个虚拟的'B'键
            self.model.set_input({'A': input_tensor, 'B': input_tensor, 'A_paths': ['camera_stream'], 'B_paths': ['camera_stream']})
            self.model.forward()

            # 3. 后处理
            output = self.model.fake_B[0].detach()
            output = (output + 1.0) / 2.0 * 255.0
            dehazed_rgb_small = output.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

            # 4. 指标计算
            gain = self.calculate_metrics(hazy_rgb_small, dehazed_rgb_small)

            # 5. 放大回原始预览尺寸
            dehazed_rgb_full = cv2.resize(dehazed_rgb_small, (frame.shape[1], frame.shape[0]),
                                          interpolation=cv2.INTER_CUBIC)

            fps_val = 1 / (time.time() - start_t)

            # 6. 显示结果
            res_bgr = cv2.cvtColor(dehazed_rgb_full, cv2.COLOR_RGB2BGR)
            cv2.putText(res_bgr, f"FPS: {fps_val:.1f}  Gain: {gain:.2f}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow('Real-time HOPnet Dehaze (Camera)', res_bgr)

            # 按 Q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 获取基础配置
    opt = TestOptions().parse()
    opt.gpu_ids = [0]

    app = CameraDehazeInference(opt)
    app.run()