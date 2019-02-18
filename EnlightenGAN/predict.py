import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from psnr import test_psnr
import torchvision.transforms as transforms

from util.util import tensor2im, save_image
from tqdm import tqdm

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(len(dataset))

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

for i, data in enumerate(tqdm(dataset)):
    model.set_input(data)
    visuals, _ = model.predict()
    img_path = model.get_image_paths()
    # print('process image... %s' % img_path)
    save_image(visuals['fake_B'], "/ssd1/chenwy/bdd100k/images_enhanced/100k/val/" + img_path[0].split('/')[-1])
    # visualizer.save_images(webpage, visuals, img_path)

webpage.save()