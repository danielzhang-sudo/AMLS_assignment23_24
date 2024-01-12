from keras.models import Model, load_model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from helper import testing
import matplotlib.pyplot as plt

def cnn(task, mode, alg, ckpt_path, x_train, y_train, x_val, y_val, x_test, y_test, classes):
    
    if mode == "training" or mode == "validation":
        

        y_train_1hot = to_categorical(y_train)
        y_val_1hot = to_categorical(y_val)

        cnn = create_model(x_train, classes)
        """
        tf.keras.utils.plot_model(
            cnn,
            to_file='CNN_model.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96,
        )
        """
        fit_model = train_validate(task, cnn, x_train, y_train_1hot, x_val, y_val_1hot)

        fit_model.save(f'./{task}/weights/cnn.keras')
    elif mode == "testing":
        fit_model = load_model(ckpt_path)
        #model = create_model(x_train, num_classes)
        #fit_model = model.load_weights(ckpt_path)
        testing(fit_model, task, mode, alg, x_test, y_test, classes)

    return

def create_model(x_train, classes):
    input = Input(shape=x_train.shape[1:])

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(len(classes), activation='softmax', name='predictions')(x)

    model = Model(inputs=[input], outputs=[x])
    print(model.summary())

    return model

def train_validate(task, model, x_train, y_train_onehot, x_val, y_val_onehot):
    batch_size = 128   
    epochs = 50
    init_lr = 0.001
    opt = Adam(lr=init_lr)
    model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])

    ckpt_path = f'./{task}'+'/weights/checkpoint-{epoch:02d}.keras'
    model_checkpoint_callback = ModelCheckpoint(filepath=ckpt_path,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True,
                                                save_freq='epoch'
                                                )

    model.save_weights(ckpt_path.format(epoch=0))

    cnn_history = model.fit(x_train, y_train_onehot,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_val, y_val_onehot),
                  verbose=2,
                  callbacks=[model_checkpoint_callback],
                  workers=10)

    # cnn_training = open(f'./{task}/cnn_training.txt', 'a')

    print(cnn_history)

    plt.plot(cnn_history.history['accuracy'])
    plt.plot(cnn_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'./{task}/figures/cnn_accuracy.png')

    plt.figure()
    plt.plot(cnn_history.history['loss'])
    plt.plot(cnn_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'./{task}/figures/cnn_loss.png')    

    return model