
import numpy as np
import config
import utils
import time
import os

from model import C3D, MutiGPU
import tensorflow as tf

def valid(valid_data, model, sess):
  total_logits = []
  total_labels = []
  for batch in test_data:
    comments, comments_length, labels = batch
    loss_t, logits_t, batch_size = model.test(sess, comments, comments_length, labels, 1.0)
    total_logits += logits_t.tolist()
    total_labels += labels
  auc = metrics.roc_auc_score(np.array(total_labels), np.array(total_logits))
  print "auc %f in valid comments" % auc

def main(args):
  save_dir = os.path.join(args.save_dir, args.model_type)
  log_dir = os.path.join(args.log_dir, args.model_type)

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)

  summary_writer = tf.summary.FileWriter(log_dir)
  config_proto = utils.get_config_proto()

  sess = tf.Session(config=config_proto)
  model = C3D(args, name="c3d")
  models = utils.get_multi_gpu_models(args, model)
  trainer = MultiGPU(args, sess, models)

  for step in range(1, args.nb_steps+1):
    step_start_time = time.time()
    train_images, train_labels, _, _, _ = input_data.read_clip_and_label(filename='list/train.list',
          batch_size=args.batch_size * args.agpu_num, num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
          crop_size=c3d_model.CROP_SIZE, shuffle=True)
    _, loss, summaries = trainer.train(sess, train_images, train_labels)
    writer.add_summary(summary, step)

  if step % args.log_step == 0:
    print "step %d, loss %.f, time %.2fs" % (step, loss, time.time() - step_start_time)


if __name__ == '__main__':
  args = config.get_args()
  main(args)
  test(args)
