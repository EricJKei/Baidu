#{6: 3169, 1: 6783, 4: 3102, 2: 3226, 7: 2359, 0: 8562, 8: 2601, 5: 4955, 3: 1243}

from config import cfg
from factory import imdb
from progressBar import ShowProcess
import scipy.misc
import os, time
import pickle
import random
import cv2

def dataBalance(im_paths, ann_paths):
    for i in range(len(im_paths)):
        assert im_paths[i][-14:-4] == ann_paths[i][-14:-4]
    res = {}    #计数
    for i in range(len(im_paths)):
        lb = int(im_paths[i][-5]) - 1
        if lb not in res.keys():
            res[lb] = 1
        else:
            res[lb] += 1
    # print(res)
    # import time
    # time.sleep(20)
    index = 40000
    add_im_paths = []
    add_ann_paths = []
    aug_img_dir = os.path.join(cfg.DATA_DIR, 'primary', 'train_aug_image_aug')
    aug_txt_dir = os.path.join(cfg.DATA_DIR, 'primary', 'train_aug_visit_aug')

    if not os.path.exists(aug_img_dir):
        os.mkdir(aug_img_dir)
    if not os.path.exists(aug_txt_dir):
        os.mkdir(aug_txt_dir)

    # 以下操作分为两个阶段,第一个阶段是数据生成阶段,在训练之前完成,第二个阶段是去除多余数据
    # 对于小于7000的：读取已存在所有该类别文件,对图像进行旋转,对文本进行减1操作,然后保存在新的文件夹中
    # 对于大于7000的：读取已存在所有该类别文件,直接对列表进行切片操作去除多余数据, 然后将剩余的数据拷贝到新的文件夹中

    for label in list(res.keys()):
        nums = res[label]

        # 直接读取文件,在每个循环中只读取一个类别的数据
        name = str(label + 1).zfill(3)
        im_dir = os.path.join(cfg.DATA_DIR, 'primary', 'train_image', name)
        ann_dir = os.path.join(cfg.DATA_DIR, 'primary', 'train_visit_pkl')
        # 只有数据名,没有具体的路径
        im_ls = os.listdir(im_dir)
        ann_ls = [im_name[:-4] + '.pkl' for im_name in im_ls]
        #具体的路径
        im_paths = [os.path.join(im_dir, im_name) for im_name in im_ls]
        ann_paths = [os.path.join(ann_dir, ann_name) for ann_name in ann_ls]

        # stage 1
        print("Label name: ", label)
        if nums < 7000:
            iters = ShowProcess(7000 - nums)
            for i in range(7000 - nums):
                iters.show_process(i)
                new_index = str(index).zfill(6)
                sd = random.randint(0, nums-1)

                im_opt = os.path.join(aug_img_dir, im_ls[sd])
                ann_opt = os.path.join(aug_txt_dir, ann_ls[sd])
                im_opt = im_opt[:-14] + new_index + im_opt[-8:]
                ann_opt = ann_opt[:-14] + new_index + ann_opt[-8:]

                add_im_paths.append(im_opt)
                add_ann_paths.append(ann_opt)

                #保存图片
                im = cv2.imread(im_paths[sd])
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                mode = random.randint(-1, 1)
                im = cv2.flip(im, mode)
                scipy.misc.imsave(im_opt, im)

                #保存文本
                with open(ann_paths[sd], 'rb') as f:
                    ann = pickle.load(f)

                sa = random.randint(1, 2)
                ann += sa

                if sa == 1:
                    ann[ann==1] = 0
                elif sa ==2:
                    ann[ann==2] = 0
                with open(ann_opt, 'wb') as f:
                    pickle.dump(ann, f)

                index += 1

        print('\n')


if __name__=='__main__':
    data=imdb('train')
    dataBalance(data.trainImgPaths, data.trainAnnPaths)
