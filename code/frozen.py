import tensorflow as tf

meta_path = '../code/spaces/save_data/gen_-01/model.meta' # Your .meta file
output_node_names = ['I_0_train/Adam']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('../code/spaces/save_data/gen_-01/'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
