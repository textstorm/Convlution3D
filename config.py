
import argparse

def get_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('--random_seed', type=int, default=1023, help='random seed')

  #data
  parser.add_argument('--save_dir', type=str, default='save/saves', help="save path")
  parser.add_argument('--log_dir', type=str, default='save/logs', help='log path')
  parser.add_argument('--load_path', type=str, default='preprocess/conv3d_deepnetA_sport1m_iter_1900000_TF.model',\
                     help="load path")
  parser.add_argument('--use_pretrained', type=bool, default=True, help='use pretrained model')

  # model
  parser.add_argument('--nb_classes', type=int, default=101, help="classes num")
  parser.add_argument('--img_h', type=int, default=112, help="image height")
  parser.add_argument('--img_w', type=int, default=112, help="image weight")
  parser.add_argument('--frame_size', type=int, default=16, help="frames a clip")
  parser.add_argument('--channels', type=int, default=3, help="num of channels")

  #optim
  parser.add_argument('--num_gpu', type=int, default=2, help='gpu num')
  parser.add_argument('--batch_size', type=int, default=16, help='example numbers every batch')
  parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
  parser.add_argument('--nb_steps', type=int, default=5000, help='number of steps')
  parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Max norm of gradient')
  parser.add_argument('--log_step', type=int, default=1, help='log steps')

  return parser.parse_args()