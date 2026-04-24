import tensorflow as tf


def kinova_rlds_to_octo(traj):
    if "steps" in traj:
        traj = traj["steps"]

    image = tf.convert_to_tensor(traj["observation"]["image"])
    T = tf.shape(image)[0]

    # Unconditional hindsight goal relabeling:
    # for each timestep t, sample a future index uniformly from [t, T-1]
    t_idx = tf.range(T, dtype=tf.int32)  # [T]
    remaining = T - t_idx                # [T]
    u = tf.random.uniform([T], minval=0.0, maxval=1.0, dtype=tf.float32)
    offsets = tf.cast(
        tf.floor(u * tf.cast(remaining, tf.float32)),
        tf.int32,
    )                                    # [T], each in [0, remaining-1]
    future_idx = t_idx + offsets         # [T]
    goal_image = tf.gather(image, future_idx)  # [T, H, W, C]

    action = tf.cast(tf.convert_to_tensor(traj["action"]), tf.float32)
    action_dim = tf.shape(action)[-1]

    def strip_to_7():
        return action[..., :7]

    def keep_as_is():
        return action

    action_7 = tf.cond(tf.equal(action_dim, 8), strip_to_7, keep_as_is)

    if "is_terminal" in traj:
        done = tf.cast(traj["is_terminal"], tf.bool)
    elif "is_last" in traj:
        done = tf.cast(traj["is_last"], tf.bool)
    else:
        def term_from_action():
            return tf.greater(action[..., 7], 0.5)

        def all_false():
            return tf.zeros([T], dtype=tf.bool)

        done = tf.cond(tf.equal(action_dim, 8), term_from_action, all_false)

    task_completed = tf.expand_dims(done, axis=-1)

    obs = {
        "image_primary": image,
        "goal_image": goal_image,
        "task_completed": task_completed,
    }

    return {
        "observation": obs,
        "action": action_7,
    }
