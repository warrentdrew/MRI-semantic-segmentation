import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default= '/tmp/med_data_local/', type = str)
parser.add_argument('--in_size', default = 32, type = int)
parser.add_argument('--k', default = 16, type = int, help= 'growth rate of dense block')
parser.add_argument('--ls', default = [8,8,8,12], type = list, help = 'layers in dense blocks')
parser.add_argument('--theta', default = 0.5, type = float, help = 'compression factor for dense net')
parser.add_argument('--k_0', default = 32, type = int, help = 'num of channel in input layer')
parser.add_argument('--lbda', default = 0, type = float, help = 'lambda for l2 reg')
parser.add_argument('--out_res', default=24, type = int, help = 'output resolution')
parser.add_argument('--feed_pos', default= False, help = 'add position information in model', action = 'store_true')    #TODO check!!
parser.add_argument('--pos_noise_stdv', default = 0, type = float, help = 'noise for position')
parser.add_argument('--epochs', default = 50, type = int)
parser.add_argument('--batch_size', default= 48, type= int)
