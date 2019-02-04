
import torch
from PIL import Image
import os

def tensor_for_board(img_tensor):
    # map into [0,1]
    tensor = (img_tensor.clone()+1) * 0.5
    tensor.cpu().clamp(0,1)

    if tensor.size(1) == 1:
        tensor = tensor.repeat(1,3,1,1)

    return tensor

def tensor_list_for_board(img_tensors_list):
    grid_h = len(img_tensors_list)
    grid_w = max(len(img_tensors)  for img_tensors in img_tensors_list)
    
    batch_size, channel, height, width = tensor_for_board(img_tensors_list[0][0]).size()
    canvas_h = grid_h * height
    canvas_w = grid_w * width
    canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)
    for i, img_tensors in enumerate(img_tensors_list):
        for j, img_tensor in enumerate(img_tensors):
            offset_h = i * height
            offset_w = j * width
            tensor = tensor_for_board(img_tensor)
            canvas[:, :, offset_h : offset_h + height, offset_w : offset_w + width].copy_(tensor)

    return canvas

def board_add_images(img_tensors_list, epoch, iter, save_dir):
    tensor = tensor_list_for_board(img_tensors_list)
    array = tensor_for_image(tensor[0]) # Save first image
    Image.fromarray(array).save(os.path.join(save_dir, 'ep{:02}_iter{:03}.jpg'.format(epoch,iter)))




def board_add_image(board, tag_name, img_tensor, step_count):
    tensor = tensor_for_board(img_tensor)

    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)




def tensor_for_image(img_tensor):
    tensor = img_tensor.clone() * 255
    tensor = tensor.cpu().clamp(0,255)
    array = tensor.detach().numpy().astype('uint8')
    if array.shape[0] == 1:
        array = array.squeeze(0)
    elif array.shape[0] == 3:
        array = array.swapaxes(0, 1).swapaxes(1, 2)
    return array


def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        array = tensor_for_image(img_tensor)        
        Image.fromarray(array).save(os.path.join(save_dir, img_name))

def save_visual(img_tensors_list, img_names, save_dir):
    img_tensors = tensor_list_for_board(img_tensors_list)

    for img_tensor, img_name in zip(img_tensors, img_names):
        array = tensor_for_image(img_tensor)
        Image.fromarray(array).save(os.path.join(save_dir, img_name))