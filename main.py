
import numpy as np
import config
import utils
import time
import os

from model import *
import tensorflow as tf

def valid(valid_data, valid_label, trainer, sess):
  total_logits = []
  total_labels = []
  for batch in valid_data:
    comments, comments_length, labels = batch
    loss_t, logits_t, batch_size = model.test(sess, comments, comments_length, labels, 1.0)
    total_logits += logits_t.tolist()
    total_labels += labels
  auc = metrics.roc_auc_score(np.array(total_labels), np.array(total_logits))
  print ("auc %f in valid comments" % auc)


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

    if step & args.eval_step == 0:
      val_images, val_labels, _, _, _ = utils.read_clip_and_label(filename='list/testlist.txt',
            batch_size=args.batch_size * args.num_gpu, num_frames_per_clip=args.frame_size,
            crop_size=args.img_h, shuffle=False)
      val_accuracy = valid(val_images, val_labels, trainer, sess)

if __name__ == '__main__':
  args = config.get_args()
  main(args)