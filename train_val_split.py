# -*- coding: UTF-8 -*-
#训练集和验证集分类

import os
import random
import PIL.Image as Image
flag = 0
def mkdir(path):
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path)
        print(' 创建成功')
        return True
    else:
        print(path)
        print(' 目录已存在')
        return False


def eachFile1(filepath):
    dir_list = []
    name_list = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        name_list.append(allDir)
        child = os.path.join('%s%s' % (filepath+'\\', allDir))
        dir_list.append(child)
    return dir_list, name_list


if __name__ == '__main__':
    parent_path = r'C:\Users\wangning\Desktop\建筑材质\\'#当需要对另外两种分类方式进行分割时，只需要将“建筑材质”修改一下即可
    print(parent_path)
    parent_list = os.listdir(parent_path)
    dir_name = parent_list
    for i in parent_list:
        path = i+'-train'
        for k in parent_list:
            if k == path:
                flag = 1       
    if flag == 0:
        for i in parent_list:
            path = parent_path+i+'-train'
            mkdir(path) 
            path = parent_path+i+'-val'
            mkdir(path)

    train_pic_dir = []
    test_pic_dir = []
    for i in dir_name:
        path_in = os.path.join(parent_path + i)
        pic_dir, pic_name = eachFile1(path_in)
        random.shuffle(pic_dir)
        train_list = pic_dir[0:int(0.7*len(pic_dir))]
        test_list = pic_dir[int(0.7*len(pic_dir)):]
        for j in train_list:   
            fromImage = Image.open(j)
            j = j.replace(i, i+'-train')
            fromImage = fromImage.convert('RGB')
            fromImage.save(j)
        for k in test_list:
            fromImage = Image.open(k)
            k = k.replace(i, i+'-val')
            fromImage = fromImage.convert('RGB')
            fromImage.save(k)   