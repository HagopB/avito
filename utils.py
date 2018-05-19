import os
import pandas as pd

def post_process(pred):
    # simple post processing
    if pred > 1:
        return 1
    elif pred < 0:
        return 0
    else:
        return pred

def post_softmax(preds):
    return None

def write_output(preds, conf):
    if not os.path.exists(conf.path.output_path):
        os.mkdir(conf.path.output_path)
        
    test = pd.read_csv(os.path.join(conf.path.input_path, "test.csv.zip"), nrows = conf.data_prep.nrows)
    sub = pd.DataFrame(test.item_id)
    sub["deal_probability"] = preds.flatten()
    sub.deal_probability = sub.deal_probability.map(lambda x: post_process(x))

    print("writing csv yo...")
    sub_path = os.path.join(conf.path.output_path, "submission.csv")
    sub.to_csv(sub_path, index = False)

def plot_history(hist, save=False):
    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
    ax1.plot(hist.history['rmse'], lw=2.0, color='b', label='train')
    ax1.plot(hist.history['val_rmse'], lw=2.0, color='r', label='val')

    ax1.set_title('CNN extra features')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('rmse')
    ax1.legend(loc='upper right')


    ax2.plot(hist.history['loss'], lw=2.0, color='b', label='train')
    ax2.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')

    ax2.set_title('CNN extra features')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('loss')
    ax2.legend(loc='upper right')

    if save:
        f.savefig("training_plot.pdf")
    else:
        f.show()