import datetime
import numpy as np
import os
import argparse
import tensorflow as tf


epoch = 10




def main(jtg, tig, is_chief):
    # Create train data
    train_X = np.linspace(-1, 1, 100)
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
    learning_rate = 0.01
    start_training_time = datetime.datetime.now()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Exampmle: {"cluster": {"ps": ["127.0.0.1:3001"], "worker": ["127.0.0.1:3002", "127.0.0.1:3003"]}, "task": {"index": 0, "type": "worker"}}
    #env = json.loads(os.environ.get("TF_CONFIG", "{}"))
    #task_data = env.get("task", None)
    cluster_spec = {'ps': ['128.135.24.250:1111','128.135.24.251:1111'], 'worker': ['128.135.24.252:1111', '128.135.24.253:1111']}
    task_type = jtg
    task_index = tig

    cluster = tf.train.ClusterSpec(cluster_spec)
    server = tf.train.Server(cluster,
                             job_name=task_type,
                             task_index=task_index)

    if task_type == "ps":
        server.join()
    elif task_type == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:{}/task:{}".format(task_type, task_index),
                cluster=cluster)):

            # Define the model
            keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
            keys = tf.identity(keys_placeholder)
            X = tf.placeholder("float", shape=[None, 1])
            Y = tf.placeholder("float", shape=[None, 1])
            w = tf.Variable(0.0, name="weight")
            b = tf.Variable(0.0, name="bias")
            global_step = tf.Variable(0, name="global_step", trainable=False)
            loss = tf.reduce_sum(tf.square(Y - tf.multiply(X, w) - b))
            train_op = optimizer.minimize(loss, global_step=global_step)
            predict_op = tf.multiply(X, w) + b
            #tf.summary.scalar("loss", loss)
            #summary_op = tf.summary.merge_all()
            #init_op = tf.global_variables_initializer()
            #saver = tf.train.Saver()
            #saver = tf.train.Saver(sharded=True)
            
            with tf.Session(server.target) as sess:
                print("Run training with epoch number: {}".format(epoch))
                sess.run(tf.global_variables_initializer())
                for i in range(epoch):
                    for (x, y) in zip(train_X, train_Y):
                        x = np.array([[x]])
                        y = np.array([[y]])
                        sess.run(train_op, feed_dict={X: x, Y: y})

                end_training_time = datetime.datetime.now()
                print("[{}] End of distributed training.".format(
                    end_training_time - start_training_time))
            '''
            sv = tf.train.Supervisor(is_chief=is_chief,
                                     logdir='./checkpoint/',
                                     init_op=init_op,
                                     #summary_op=summary_op,
                                     summary_op=None,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=60)

            try:
                with sv.managed_session(server.target) as sess:
                    #print("Save tensorboard files into: {}".format(FLAGS.output_path))
                    #writer = tf.summary.FileWriter(FLAGS.output_path, sess.graph)

                    print("Run training with epoch number: {}".format(epoch))
                    for i in range(epoch):
                        for (x, y) in zip(train_X, train_Y):
                            x = np.array([[x]])
                            y = np.array([[y]])
                            sess.run(train_op, feed_dict={X: x, Y: y})

                    end_training_time = datetime.datetime.now()
                    print("[{}] End of distributed training.".format(
                        end_training_time - start_training_time))


            except Exception as e:
                print(e)
            '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobname', action='store', type=str, default='none', help='ps or worker')
    parser.add_argument('-t', '--taskid', action='store', type=int, default=0)
    parser.add_argument('-c', '--chief', action='store_true', default=False)
    args = parser.parse_args()
    job_type_global = args.jobname
    task_idx_global = args.taskid
    is_chief_global = args.chief
    print()

    main(job_type_global, task_idx_global, is_chief_global)