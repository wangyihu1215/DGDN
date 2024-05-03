import argparse
import os
import time

import torch
from PIL import Image
from torchvision import transforms

from nets import Net


def test(config):
    # net load
    net = Net().cuda()

    # snapshot load
    snapshot_name = config.snapshot.split('/')[-1]
    print(f"load snapshot {snapshot_name} for testing")
    net.load_state_dict(torch.load(config.snapshot))
    net.eval()

    with torch.no_grad():

        img_list = [img_name for img_name in os.listdir(config.input)]

        total_time = 0

        for idx, img_name in enumerate(img_list):

            img = Image.open(os.path.join(config.input, img_name)).convert('RGB')
            w, h = img.size
            img_var = transform(img).unsqueeze(0).cuda()

            start_time = time.time()

            result, dep = net(img_var)

            torch.cuda.synchronize()

            cost_time = time.time() - start_time

            if idx != 0:
                total_time += cost_time

            print('predicting: %d / %d, avg_time: %.5f' % (idx + 1, len(img_list), cost_time))

            # result save
            output_path = os.path.join(config.output, 'prediction_%s' % snapshot_name.split('.')[0])
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            result = to_pil(result.data.squeeze(0).cpu())
            result = result.resize((w, h), resample=Image.Resampling.LANCZOS)
            result.save(os.path.join(output_path, img_name), 'png')

            # depth map save
            if config.gen_depth:
                output_depth_path = os.path.join(config.output_depth, 'prediction_%s' % snapshot_name.split('.')[0])
                if not os.path.exists(output_depth_path):
                    os.mkdir(output_depth_path)
                dep = to_pil(dep.data.squeeze(0).cpu())
                depth = dep.resize((w, h), resample=Image.Resampling.LANCZOS)
                depth.save(os.path.join(output_depth_path, img_name), 'png')
        print(f'total avg time is {total_time / idx:.5f} seconds!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='F:/Datasets/Test/SYNS/haze/')
    parser.add_argument('--output', type=str, default='F:/Datasets/Test/SYNS/dehaze/')
    parser.add_argument('--output_depth', type=str, default='F:/Datasets/SYNS/Test/depth/')
    parser.add_argument('--snapshot', type=str, default='./checkpoint/100000.pth')
    parser.add_argument('--gen_depth', default=True, help='generate depth map or not')

    config = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(2024)
    torch.cuda.set_device(0)

    transform = transforms.Compose([
        transforms.Resize([1024, 512]),
        transforms.ToTensor()])

    to_pil = transforms.ToPILImage()

    test(config)
