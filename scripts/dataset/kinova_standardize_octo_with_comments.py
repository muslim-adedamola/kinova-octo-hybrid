"""
Octo standardization function for the custom Kinova RLDS dataset.

Octo's dataset loader calls kinova_rlds_to_octo on each trajectory. The function
renames the raw TFDS keys into Octo's expected schema, strips the terminate flag
from 8-D actions to produce 7-D robot actions, and creates goal-image tasks using
unconditional hindsight relabeling.
"""

import tensorflow as tf


def kinova_rlds_to_octo(traj):
    """Convert one Kinova RLDS trajectory into Octo observation/action format.

    Returned observation keys:
      - image_primary: current scene camera image
      - goal_image: future image sampled independently for each timestep
      - task_completed: terminal/done indicator expanded as a task-completion bit

    Returned action is 7-D: 6 arm deltas + 1 absolute gripper command.
    """
    if "steps" in traj:
        traj = traj["steps"]

    image = tf.convert_to_tensor(traj["observation"]["image"])
    T = tf.shape(image)[0]

    # Unconditional hindsight goal relabeling:
    # Even if a raw goal image exists elsewhere, sample a future image for every
    # timestep. This creates many valid goal-conditioned training pairs from the
    # same demonstration and matches the goal-relabeling setup used in training.
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
