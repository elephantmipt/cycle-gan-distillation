# Distillation

First of all you should train your teacher and save it in a dict with this structure

```python
state_dict = {
    "model_state_dict": {
        "generator_ab": # state dict here
        "generator_ba": # state dict here
        "discriminator_a": # state dict here
        "discriminator_b": # state dict here
}

torch.save("logdir/last.pth", state_dict)
```

After that the API is very similar to training API.

{% page-ref page="training.md" %}

But there are several callbacks you should add. Let's take a look on this part of the pipeline.

```python
callbacks = [
    PrepareGeneratorPhase(),
    GANLoss(),
    CycleGANLoss(),
    IdenticalGANLoss(ba_key="generator_s"),
    ##########################################
    ########## Distilation losses ############
    HiddenStateLoss(transfer_layer=[1, 4, 8]),
    TeacherStudentLoss(),
    ##########################################
    GeneratorOptimizerCallback(
        keys=[
            "gan_loss",
            "cycle_loss",
            "identical_loss",
            "hidden_state_loss",
            "ts_difference",
        ],
        weights=[1, 10, 5, 1, 10],
    ),
    PrepareDiscriminatorPhase(),
    DiscriminatorLoss(),
    DiscriminatorOptimizerCallback(),
    LogImageCallback(model_key="generator_s"),
    LogImageCallback(key="mipt", img=mipt_photo, model_key="generator_s"),
    LogImageCallback(key="vk", img=zinger_photo, model_key="generator_s"),
]
```

