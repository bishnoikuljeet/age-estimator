"""
model_utils.py:
    version 0.0.1

    A python file/module which contains the functions to
	related to model for `Age Estimator` solution.

    - Step 1: Create python environment from `setup\requirements.txt`
    - Step 2: Run `python app.py`
"""

# Importing the required modules/packages
from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


def get_model(cfg):
    """
    Function to get the required model.

    Parameters:
    -----------
    cfg: OmegaConf object
        Configuration details for model selection.

    Returns:
    --------
    model: `tensorflow.keras.models.Model`
        A valid `tensorflow.keras.models.Model` object.

    References:
    -----------
    NA

    Examples:
    ---------
    get_model(cfg)
    """

    # Getting base model
    base_model = getattr(applications, cfg.model.model_name)(
        include_top=False,
        input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
        pooling="avg",
    )

    # Getting model features
    features = base_model.output

    # Creating dense layers for gender
    pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)

    # Creating dense layers for age
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)

    # Adding new layers to the model
    model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])

    # Returning the model
    return model


def get_optimizer(cfg):
    """
    Function to get the optimizer for the model.

    Parameters:
    -----------
    cfg: OmegaConf object
        Configuration details for model selection.

    Returns:
    --------
    model: `tensorflow.keras.optimizers.SGD` or`tensorflow.keras.optimizers.Adam`
        A valid optimizer object.

    References:
    -----------
    NA

    Examples:
    ---------
    get_optimizer(cfg)
    """
    # If SGD optimizer required
    if cfg.train.optimizer_name == "sgd":
        return SGD(lr=cfg.train.lr, momentum=0.9, nesterov=True)

    # If ADAM optimizer required
    elif cfg.train.optimizer_name == "adam":
        return Adam(lr=cfg.train.lr)

    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def get_scheduler(cfg):
    """
    Function to get the Schedule for the model.

    Parameters:
    -----------
    cfg: OmegaConf object
        Configuration details for model selection.

    Returns:
    --------
    model: `Schedule`
        A valid Schedule object.

    References:
    -----------
    NA

    Examples:
    ---------
    get_scheduler(cfg)
    """

    class Schedule:
        """
        Class to get the schedule for the model.
        """

        def __init__(self, nb_epochs, initial_lr):
            """
            Constructor to get the schedule for the model.

            Parameters:
            -----------
            nb_epochs: `int`
                A valid python `int` representing no of epochs for the model.

            initial_lr: `float`
                A valid python `float` representing initial learning rate
                for the model.

            References:
            -----------
            NA

            Examples:
            ---------
            Schedule(nb_epochs=100, initial_lr=0.01)
            """
            # Assigning epocs attributes of the class
            self.epochs = nb_epochs

            # Assigning initial_lr attributes of the class
            self.initial_lr = initial_lr

        def __call__(self, epoch_idx):
            """
            Method to get the updated learning rate for the model
            depending on current epoch no.

            Parameters:
            -----------
            epoch_idx: `int`
                A valid python `int` representing epoch no for the model.

            References:
            -----------
            NA

            Examples:
            ---------
            obj.call(epoch_idx=50)
            """

            # If current epoch is within 25% of total epochs
            if epoch_idx < self.epochs * 0.25:
                return self.initial_lr

            # If current epoch is within 50% of total epochs
            elif epoch_idx < self.epochs * 0.50:
                return self.initial_lr * 0.2

            # If current epoch is within 75% of total epochs
            elif epoch_idx < self.epochs * 0.75:
                return self.initial_lr * 0.04

            return self.initial_lr * 0.008

    # Returning Schedule object
    return Schedule(cfg.train.epochs, cfg.train.lr)
