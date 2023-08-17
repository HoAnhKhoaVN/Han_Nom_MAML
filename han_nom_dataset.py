from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
from torch import LongTensor, tensor, int64, from_numpy
from PIL import Image
import os
import numpy as np

IMG_CACHE = {}
class HanNomDataset(Dataset):
    def __init__(
        self,
        mode = 'train',
        root= 'demo_ds/train',
    ):
        super(HanNomDataset, self).__init__()
        self.root = root
        self.mode = mode
        self.dir = os.path.join(self.root, mode)
        self.ds_dummy: ImageFolder = ImageFolder(root= self.dir)
        self.imgs_lbs = self.ds_dummy.imgs
        self.y = LongTensor(self.ds_dummy.targets)
        self.classes = self.ds_dummy.classes


    def load_img(self, path: str)-> Image:
        if path in IMG_CACHE:
            x = IMG_CACHE[path]
        else:
            x = Image.open(path)
            IMG_CACHE[path] = x
        x = x.resize((32, 32))
        shape = 3, x.size[0], x.size[1]
        x = np.array(x, np.float32, copy=False)
        x = from_numpy(x/255)
        x = x.transpose(0, 1).contiguous().view(shape)

        return x

    def __getitem__(self, idx):
        img_path, target = self.imgs_lbs[idx]
        img = self.load_img(img_path)

        return img, target

    def __len__(self):
        return len(self.imgs_lbs)


class HanNomDatasetNShot(Dataset):
    def __init__(
            self,
            mode,
            root,
            batchsz,
            n_way,
            k_shot,
            k_query,
            imgsz
        ):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """
        self.resize = imgsz
        self.x = HanNomDataset(
            mode = mode,
            root= root
        )
        # region Create dict label with image
        temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
        labels = []
        for (img, label) in self.x:
            labels.append(label)
            if label in temp.keys():
                temp[label].append(img)
            else:
                temp[label] = [img]
        self.x = []
        for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
            self.x.append(np.array(imgs))
        # endregion


        # region change type label
        # as different class may have different number of imgs
        self.x = np.array(self.x).astype(np.float32)  # [[20 imgs],..., 1623 classes in total]
        print(f'temp: {len(temp)}')
        print('self.x.shape: ',self.x.shape)
        # each character contains 20 imgs
        print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
        # endregion

        # region get class
        self.num_classes = self.x.shape[0]
        # endregion

        self.batchsz = batchsz
        print(f'batchsz = {batchsz}')
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20

        # save pointer of current read batch in total cache
        self.indexes = 0
        self.datasets = self.x  # original data cached
        print("DB: train", self.x.shape)

        self.datasets_cache = self.load_data_cache(self.datasets)  # current epoch data cached
        
    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for _ in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for _ in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 3, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 3, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 3, 32, 32]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 3, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int64).reshape(self.batchsz, setsz)
            # [b, qrysz, 3, 32, 32]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 3, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int64).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes >= len(self.datasets_cache):
            self.indexes = 0
            self.datasets_cache = self.load_data_cache(self.datasets)

        next_batch = self.datasets_cache[self.indexes]
        self.indexes+= 1

        return next_batch

class ClassificationDataset(Dataset):
    def __init__(
        self,
        mode: str,
        root_dir: str,
        transform = None,
    ):
        super(ClassificationDataset, self).__init__()
        self.root = root_dir
        self.mode = mode
        self.dir = os.path.join(self.root, mode)
        self.dataset: ImageFolder = ImageFolder(root= self.dir, transform= transform)

    def __len__(self):
        return len(self.dataset.imgs)
    
    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

        

if __name__ == '__main__':
    DS_PATH = 'D:/Master/term_2/NLP/22C15033/Code/Dataset/demo_ds'
    import  torch
    db = HanNomDatasetNShot(
        mode = 'train',
        root= DS_PATH,
        batchsz=20,
        n_way=5,
        k_shot=5,
        k_query=5,
        imgsz=32
    )

    for i in range(1000):
        x_spt, y_spt, x_qry, y_qry = db.next()


        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        x_spt = torch.from_numpy(x_spt)
        x_qry = torch.from_numpy(x_qry)
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)

        task_num, setsz, c_, h, w = x_spt.size()
        print(f'task_num: {task_num}')
        
        # print(f'''
        # - x_spt : {x_spt}
        # - y_spt : {y_spt}
        # - y_qry : {y_qry}
        # - x_qry : {x_qry}
        # ''')
        break
        # batchsz, setsz, c, h, w = x_spt.size()


        # viz.images(x_spt[0], nrow=5, win='x_spt', opts=dict(title='x_spt'))
        # viz.images(x_qry[0], nrow=15, win='x_qry', opts=dict(title='x_qry'))
        # viz.text(str(y_spt[0]), win='y_spt', opts=dict(title='y_spt'))
        # viz.text(str(y_qry[0]), win='y_qry', opts=dict(title='y_qry'))


        # time.sleep(10)


    # train_path = os.path.join(DS_PATH, 'train')
    # ds = HanNomDataset(root= train_path)
    

    # print(f'Length of training: {len(ds)}')
    # print(f'Classes in training: {len(ds.classes)}')

    # x, y = next(iter(ds))
    # print(f'Type x: {type(x)}')
    # print(f'Type y: {type(y)}')

    # print(f'Size X: {x.size()}')
    # print(f'Y: {y}')

    # print(f"X: {x}")


    # valset = ImageFolder(root=f'{DS_PATH}/val', transform=transform)
    # testset = ImageFolder(root=f'{DS_PATH}/test', transform=transform)
