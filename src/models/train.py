import time
import dill
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from src.file_management.file_handling import hms_string

def train_model(model, train_dataset, val_dataset, run_dir, initial_epoch=0, max_epochs=10, batch_size=32, validation=True):
    start_time = time.time()
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_logs_dir = run_dir / "tensorboard_logs"

    

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


    checkpoint_latest_cb = MyModelCheckpoint(model,
        checkpoint_dir / "model-{epoch:02d}-{val_loss:.2f}.hdf5",
        monitor='val_loss',
    )
    
    # checkpoint_best_cb = ModelCheckpoint(
    #     os.path.join(checkpoint_dir, "model_best.hdf5"),
    #     save_best_only=True,
    #     monitor="val_loss",
    # )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        tensorboard_logs_dir,
    )


    if validation:
        cb = [checkpoint_latest_cb, early_stopping_cb, tensorboard_cb]
        # Train and validate the model
        model.fit(
            train_dataset,
            batch_size=batch_size,
            epochs=max_epochs,
            initial_epoch=initial_epoch,
            callbacks=cb,
            validation_data=val_dataset,
            verbose=2,
        )
    else:
        # Train the model on all the data
        # Combine the training and validation datasets
        

    #     cb = [checkpoint_latest_cb, tensorboard_cb]
    #     X_train_all = np.concatenate([input_data["X_train"], input_data["X_val"]], axis=0)
    #     y_train_all = np.concatenate([input_data["y_train"], input_data["y_val"]], axis=0)

    #     model.fit(
    #         X_train_all,
    #         y_train_all,
    #         batch_size=batch_size,
    #         epochs=max_epochs,
    #         initial_epoch=initial_epoch,
    #         callbacks=cb,
    #     )
        pass
    
    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(hms_string(elapsed_time)))

    return model




class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch,logs)\

        # Also save the optimizer state
        filepath = self._get_file_path(epoch=epoch, 
                logs=logs, batch=None)
        filepath = filepath.rsplit(".", 1)[ 0 ] # comment if NOT using h5
        filepath += ".pickle"

        with open(filepath, 'wb') as fp:
            dill.dump(
                {
                    'opt': self.model.optimizer.get_config(),
                    'epoch': epoch+1
                 # Add additional keys if you need to store more values
                }, fp, protocol=dill.HIGHEST_PROTOCOL)
        print('\nEpoch %05d: saving optimizer to %s' % (epoch + 1, filepath))


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return LearningRateScheduler(schedule)
