# Kinova-Octo Hybrid

Real-robot finetuning and deployment of Octo on a Kinova manipulator for goal-image-conditioned bottle pick-and-lift.

This project uses a hybrid action design: Octo's diffusion action head for arm motion and a binary BCE gripper head for open/close control. The deployed policy runs with a latency-aware automatic receding-horizon controller.

> Status: research engineering prototype validated on a Kinova robot.
