import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

with tf.device('/GPU:0'):

    # Parameter für die Schedulers
    initial_learning_rate = 0.001
    decay_steps = 898*300    # Gesamtzahl der Training-Schritte
    alpha = 0.0  # Endwert der Lernrate für CosineDecay
    decay_rate = 0.631  # Decay-Rate für ExponentialDecay

    # Erstellen der Schedulers
    cosine_lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha
    )

    exponential_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps // 200,  # Anpassung der decay_steps für eine langsamere Reduktion
        decay_rate=decay_rate,
        staircase=True
    )

    # Visualisierung der Lernrate über die Trainingsschritte
    steps = np.arange(0, decay_steps)
    cosine_learning_rates = [cosine_lr_schedule(step).numpy() for step in steps]
    exponential_learning_rates = [exponential_lr_schedule(step).numpy() for step in steps]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, cosine_learning_rates, label='Cosine Decay')
    plt.plot(steps, exponential_learning_rates, label='Exponential Decay')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule Comparison: Cosine Decay vs Exponential Decay')
    plt.legend()
    plt.grid(True)
    plt.show()
