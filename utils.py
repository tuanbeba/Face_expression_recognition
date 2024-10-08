import torch
import os
import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader



def multiclass_accuracy(y_pred,y_true):
    # Lấy nhãn dự đoán
    y_pred_labels = torch.argmax(y_pred, dim=1)
    accuracy = (y_pred_labels == y_true).float().mean().item()

    return accuracy

def get_transforms():
    train_augs = T.Compose([
        T.RandomHorizontalFlip(p = 0.5),
        T.RandomRotation(degrees=(-20, +20)),
        T.ToTensor()
    ])

    test_augs = T.Compose([
        T.ToTensor()
    ])
    return train_augs, test_augs

def create_dataloader(train_folder_path, test_folder_path, train_augs, test_augs, batch_size, num_workers):
    trainset = ImageFolder(train_folder_path,transform= train_augs)
    testset = ImageFolder(test_folder_path,transform= test_augs)
    print(f"Total no. of examples in trainset : {len(trainset)}")
    print(f"Total no. of examples in validset : {len(testset)}")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,num_workers=num_workers)


    print(f"Total no. of batches in trainloader : {len(trainloader)}")
    print(f"Total no. of batches in validloader : {len(testloader)}")

   
    return trainloader, testloader

def create_new_output_dir(base_dir):
    all_subdir = os.listdir(base_dir) 
    create_new_dir = os.path.join(base_dir,str(len(all_subdir)+1))
    os.makedirs(create_new_dir)

    return create_new_dir

def view_visualize(train_losses, test_losses,train_accuracies, test_accuracies, output_dir):

    # Vẽ biểu đồ đường cho array1 và array2 với các màu khác nhau
    plt.figure()
    plt.plot(train_losses, 'b', label='train')  # Màu xanh (blue)
    plt.plot(test_losses, 'r', label='test')  # Màu đỏ (red)
    # Đặt nhãn cho trục x và y
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # Đặt tiêu đề cho biểu đồ
    plt.title('model loss')
    # Hiển thị chú giải để phân biệt hai đường
    plt.legend()
    #lưu biểu đồ
    save_path = os.path.join(output_dir,'loss_curve.png')
    plt.savefig(save_path)
    plt.close()

    # Vẽ biểu đồ đường cho array1 và array2 với các màu khác nhau
    plt.figure()
    plt.plot(train_accuracies, 'b', label='train')  # Màu xanh (blue)
    plt.plot(test_accuracies, 'r', label='test')  # Màu đỏ (red)
    # Đặt nhãn cho trục x và y
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    # Đặt tiêu đề cho biểu đồ
    plt.title('model accuracy')

    # Hiển thị chú giải để phân biệt hai đường
    plt.legend()
    #lưu biểu đồ
    save_path = os.path.join(output_dir,'accuracy_curve.png')
    plt.savefig(save_path)
    plt.close()

