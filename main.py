
import numpy as np
import config
import utils
import time
import os

from model import *
import tensorflow as tf

def valid(args, test_file, trainer, sess):
  next_start_pos = 0
  infer_list = []
  labels = []
  num_test_videos =  len(list(open(test_file)))
  all_steps = int((num_test_videos - 1) / (args.batch_size * args.num_gpu) + 1)
  for step in range(all_steps):
    start_time = time.time()
    test_images, test_labels, next_start_pos, _, valid_len = utils.read_clip_and_label(
                    test_file, args.batch_size * args.num_gpu, start_pos=next_start_pos)
    infer_list.extend(trainer.test(sess, test_images, test_labels))
    labels.extend(test_labels)

  return np.mean(np.equal(infer_list[:num_test_videos], labels[:num_test_videos])), time.time()-start_time

def main(args):
  save_dir = os.path.join(args.save_dir)
  log_dir = os.path.join(args.log_dir)

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  summary_writer = tf.summary.FileWriter(log_dir)
  config_proto = utils.get_config_proto()

  sess = tf.Session(config=config_proto)
  models = get_multi_gpu_models(args, sess)
  trainer = MultiGPU(args, models)
  sess.run(tf.global_variables_initializer())
  if args.use_pretrained:
    models = restore_models(args, sess, models)

  for step in range(1, args.nb_steps+1):
    step_start_time = time.time()
    train_images, train_labels, _, _, _ = utils.read_clip_and_label(filename='list/trainlist.txt',
          batch_size=args.batch_size * args.num_gpu, num_frames_per_clip=args.frame_size,
          crop_size=args.img_h, shuffle=True)
    _, loss, accuracy, summaries = trainer.train(sess, train_images, train_labels)
    summary_writer.add_summary(summaries, step)

    if step % args.log_step == 0:
      print ("step %d, loss %.5f, accuracy %.5f, time %.2fs" % (step, loss, accuracy, time.time() - step_start_time))

    if step % args.eval_step == 0:
      val_accuracy, test_time = valid(args, 'list/testlist.txt', trainer, sess)
      print ("test accuracy: %.5f, test time: %.5f" % (val_accuracy, test_time))

if __name__ == '__main__':
  args = config.get_args()
  main(args)