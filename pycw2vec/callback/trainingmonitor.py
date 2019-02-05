#encoding:utf-8
import numpy as np
import json
from os import path
import matplotlib.pyplot as plt
from ..utils.utils import ensure_dir
plt.switch_backend('agg') # 防止ssh上绘图问题

class TrainingMonitor():
    def __init__(self, fig_dir, arch,json_dir=None, start_at=0):
        '''
        :param figPath: 保存图像路径
        :param jsonPath: 保存json文件路径
        :param startAt: 重新开始训练的epoch点
        '''
        self.start_at = start_at
        self.H = {}
        self.loss_path = path.sep.join([fig_dir,arch+'_loss.png'])
        self.acc_path = path.sep.join([fig_dir,arch+"_accuracy.png"])
        self.f1_path  = path.sep.join([fig_dir,arch+"_f1.png"])
        self.json_path = path.sep.join([json_dir,arch+"_training_monitor.json"])
        self.use = 'on_epoch_end'

        ensure_dir(fig_dir)
        ensure_dir(json_dir)

    def _restart(self):
        if self.start_at > 0:
            # 如果jsonPath文件存在，咋加载历史训练数据
            if self.json_path is not None:
                if path.exists(self.json_path):
                    self.H = json.loads(open(self.json_path).read())
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]

    def step(self,logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            # np.float32会报错
            if not isinstance(v,np.float):
                v = round(float(v),4)
            l.append(v)
            self.H[k] = l

        # 写入文件
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        #保存train图像
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"],label="train_loss")
            plt.plot(N, self.H["val_loss"],label="val_loss")
            plt.legend()
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.title("Training Loss [Epoch {}]".format(len(self.H["loss"])))
            plt.savefig(self.loss_path)
            plt.close()




