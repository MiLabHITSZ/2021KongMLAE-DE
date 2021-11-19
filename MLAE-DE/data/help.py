#@Time      :2021/1/15 14:48
#@Author    :zhounan
#@FileName  :help.py



def adjust_nuswide():
    import os
    import numpy as np
    filePath = '../data/NUSWIDE/images'
    name = os.listdir(filePath)
    name1 = sorted(name, key=lambda i: int(i.split('_')[0]))

    img_list_path = os.path.join('../data/NUSWIDE/', '{}Imagelist.txt'.format('Train'))
    tag_list_path = os.path.join('../data/NUSWIDE/', '{}_Tags81.txt'.format('Train'))
    img_name_list = []
    with open(img_list_path, 'r') as f:
        for line in f.readlines():
            line = line[0:len(line) - 1]
            img_name_list.append(line.split('\\')[-1])
    tag_list = []
    with open(tag_list_path, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            line = line[0:len(line) - 1]
            line = [int(i) for i in line]
            tag_list.append(line)

    name2 = [i.split('_')[1] for i in name1]

    new_tag = []
    new_image = []
    error_images = []
    for img_name, tag in zip(img_name_list, tag_list):
        img_name = img_name.split('_')[1].split('.')[0]
        try:
            idx = name2.index(img_name)
            new_tag.append(tag)
            new_image.append(name1[idx])
        except:
            error_images.append(name1[idx])

    img_list_path = os.path.join('../data/NUSWIDE/', '{}ImagelistFilter.txt'.format('Train'))
    tag_list_path = os.path.join('../data/NUSWIDE/', '{}_Tags81Filter.txt'.format('Train'))
    with open(img_list_path, 'w') as f:
        for img in new_image:
            f.write(img + '\n')
    with open(tag_list_path, 'w') as f:
        for tag in new_tag:
            tag = [str(t) for t in tag]
            f.write(' '.join(tag) + '\n')


    img_list_path = os.path.join('../data/NUSWIDE/', '{}Imagelist.txt'.format('Test'))
    tag_list_path = os.path.join('../data/NUSWIDE/', '{}_Tags81.txt'.format('Test'))
    img_name_list = []
    with open(img_list_path, 'r') as f:
        for line in f.readlines():
            line = line[0:len(line) - 1]
            img_name_list.append(line.split('\\')[-1])
    tag_list = []
    with open(tag_list_path, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            line = line[0:len(line) - 1]
            line = [int(i) for i in line]
            tag_list.append(line)

    name2 = [i.split('_')[1] for i in name1]

    new_tag = []
    new_image = []
    error_images = []
    for img_name, tag in zip(img_name_list, tag_list):
        img_name = img_name.split('_')[1].split('.')[0]
        try:
            idx = name2.index(img_name)
            new_tag.append(tag)
            new_image.append(name1[idx])
        except:
            error_images.append(name1[idx])

    img_list_path = os.path.join('../data/NUSWIDE/', '{}ImagelistFilter.txt'.format('Test'))
    tag_list_path = os.path.join('../data/NUSWIDE/', '{}_Tags81Filter.txt'.format('Test'))
    with open(img_list_path, 'w') as f:
        for img in new_image:
            f.write(img + '\n')
    with open(tag_list_path, 'w') as f:
        for tag in new_tag:
            tag = [str(t) for t in tag]
            f.write(' '.join(tag) + '\n')

if __name__ == '__main__':
    adjust_nuswide()
    # import os
    # img_list_path = os.path.join('../data/NUSWIDE/', '{}ImagelistFilter.txt'.format('Train'))
    # tag_list_path = os.path.join('../data/NUSWIDE/', '{}_Tags81Filter.txt'.format('Train'))
    # img_name_list = []
    # # with open(img_list_path, 'r') as f:
    # #     for line in f.readlines():
    # #         line = line.strip()
    # #         img_name_list.append(line.split('\\')[-1])
    # tag_list = []
    # with open(tag_list_path, 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         line = line.split(' ')
    #         line = [int(i) for i in line]
    #         tag_list.append(line)