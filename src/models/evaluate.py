def evaluate_model(model, test_dataset):
    score = model.evaluate(test_dataset, verbose=0)
    print("Test loss: {}".format(score[0]))
    print("Test accuracy: {}".format(score[1])) 



def display_tuner_trials(tuner, num_trials=1000):
    best_trials = tuner.oracle.get_best_trials(num_trials)

    best_params = []
    for trial in best_trials:
        params = trial.hyperparameters.values
        val_loss = trial.metrics.get_best_value("val_loss")
        best_params.append((trial, params, val_loss))

    for trial, params, val_loss in best_params:
        print(f"Trial index: {trial.trial_id}")
        print(f"val_loss: {val_loss}")
        print(params)