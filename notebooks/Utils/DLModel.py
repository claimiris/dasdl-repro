'''
	Guided DL for DAS Denoising
	Dec, 10, 2023
	by Omar M. Saad

	--- dasdl-repro changes (see README "Edits to the original code") ---
	E1  `from keras...` imports -> `from tensorflow.keras...` (Keras 2 / Keras 3 mix-up on TF>=2.16).
	E3  Train() gained OPTIONAL keyword arguments. With their defaults the function is
	    byte-for-byte the original computation; they are only used by the extended experiments:
	      train_ratio  fraction of patches used for fitting (paper Sec. 4.3); prediction still uses all patches
	      patch_fn     alternative (memory-efficient, identical-output) patching function for the large SAFOD data
	      dtype        cast patches (e.g. np.float32) to fit SAFOD (800 x 14,999) in Colab RAM
	      seed         RNG seed for the training-ratio subset
	      model_dir    where the best-loss checkpoint is written
	      verbose      Keras fit verbosity
	      batch_feed   feed mini-batches from the in-memory arrays (keras Sequence) instead of letting Keras copy
	                   all patches into TF memory; same shuffling/batch size. Used for SAFOD (Colab RAM).
	      return_extras  also return {'history', 'dense_weights', 'fit_seconds', 'n_patches'} (paper Sec. 4.4/4.5)
'''
# Import Libs.
import time
import numpy as np
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Lambda, concatenate, Flatten, Add, Reshape, Dense, Input, GlobalAvgPool1D, Activation, multiply
from Utils.Utils import *
from tensorflow.keras.layers import Lambda  # E1 (was: from keras.layers import Lambda)
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K  # E1 (was: from keras import backend as K)


# Attention Network.
def Attention(inp1):
    filters = inp1.shape[-1]
    sa   = (1,filters)
    inp2 = GlobalAvgPool1D()(inp1)
    inp2 = Reshape(sa)(inp2)

    x = Dense(filters, activation='relu')(inp2)
    x = Dense(filters, activation='sigmoid')(x)
    x = multiply([inp1,x])
    x = Add()([x,inp1])

    return x

# Core Block of the Compact Layer.
def Block(inp,D):

    # Fully-connected Layers.
    x = Dense(D,activation='relu')(inp)
    x = Dense(D,activation='relu')(x)
    x = Reshape((x.shape[-1],1))(x)

    return x

# Compact Layer which extracts the spatial relation between the CWT scale and band-pass filtered data.
def compactlayer(y,D):

    # Split the Data into Two Channels (Branches).
    s0 = Lambda(lambda x:x[:,:,0])(y)
    s1 = Lambda(lambda x:x[:,:,1])(y)

    # The Band-pass Filter Branch
    B1 = Block(s0,D)
    # The CWT Scale Branch.
    B2 = Block(s1,D)
    # Applying the Attention to the Concatenated Data.
    B    = concatenate([B1,B2], axis=-1)
    Batt = Attention(B)

    return Batt


class _PatchBatches(tensorflow.keras.utils.Sequence):
    """dasdl-repro E3 helper: mini-batches sliced from in-memory patch arrays (no full-array copy)."""
    def __init__(self, xF2, x2, y, batch_size, shuffle=True, seed=None):
        super().__init__()
        self.xF2, self.x2, self.y, self.bs, self.shuffle = xF2, x2, y, batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self.idx = np.arange(xF2.shape[0])
        if shuffle:
            self.rng.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.bs))

    def __getitem__(self, i):
        j = np.sort(self.idx[i * self.bs:(i + 1) * self.bs])
        x = (self.xF2[j], self.x2[j])
        # Keras unpacks a returned tuple as (x, y[, w]); wrap x so both inputs stay inputs
        return (x,) if self.y is None else (x, self.y[j])

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.idx)


# Training the DL Model.
def Train(CWT,BP,EPOCHNO,BATCHSIZE,w1,w2,s1z,s2z,D1,eq,
          train_ratio=1.0, patch_fn=None, dtype=None, seed=None,
          model_dir='./models', verbose=1, return_extras=False, batch_feed=False):

    patchf = patch if patch_fn is None else patch_fn

    # Normalize the CWT Scale
    ma = np.max(np.abs(CWT))
    dataInput = CWT/ma

    #ma1 = np.max(np.abs(dataInputF))
    #dataInputF = dataInputF/ma1

    # Patching the CWT SCALE.
    dataInputP = patchf(dataInput,w1,w2,s1z,s2z)
    if dtype is not None:
        dataInputP = dataInputP.astype(dtype, copy=False)
    dataInput2 = np.reshape(dataInputP,(dataInputP.shape[0],w1*w2,1))

    # Patching the Band-pass Filtered Data.
    dataInputF = patchf(BP,w1,w2,s1z,s2z)
    if dtype is not None:
        dataInputF = dataInputF.astype(dtype, copy=False)
    dataInputF2 = np.reshape(dataInputF,(dataInputF.shape[0],w1*w2,1))


    # Setting the Inputs to the DL Model.
    input_shape = (w1, w1,1)
    inp1 = layers.Input(shape=(w1*w2,1),name='input_layer1')
    inp2 = layers.Input(shape=(w1*w2,1),name='input_layer2')

    # Setting the Number of Neurons for the Encoder/Decoder.
    D2 = int(D1/2)
    D3 = int(D2/2)
    D4 = int(D3/2)
    D5 = int(D4/2)
    D6 = int(D5/2)

    # Concatenate the Patches.
    inp4 = concatenate([inp1,inp2])

    # Encoder Compact Layers
    e1 = compactlayer(inp4,D1)
    e2 = compactlayer(e1,D2)
    e3 = compactlayer(e2,D3)

    # Decoder Compact Layers
    d4 = compactlayer(e3,D2)
    d5 = compactlayer(d4,D1)
    d6 = compactlayer(d5,D1)

    # Output Layer
    d6 = Flatten()(d6)
    y = Dense(w1*w2, activation='linear')(d6)

    # Creating the Model
    model = Model(inputs=[inp1,inp2], outputs=[y])

    # Schedule the Learning Rate.
    A = 50
    def lr_schedule(epoch):
        initial_lr = 1e-3

        if epoch <= A:
            lr = initial_lr
        elif epoch <= A+10:
            lr = initial_lr / 2
        elif epoch <= A+30:
            lr = initial_lr / 10
        else:
            lr = initial_lr / 20
        return lr
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=0.5,
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6,
                                       monitor='loss')

    # Setting the Optimizer and the Loss Function. Also, Saving the Best Model Based on the Loss.
    sgd = tensorflow.keras.optimizers.Adam()
    model.compile(optimizer=sgd, loss=tensorflow.keras.losses.LogCosh(),metrics=['mse'])
    early_stopping_monitor = EarlyStopping(monitor= 'loss', patience = 5)
    ckpt = model_dir + '/DAS_PATCH_N_eq_' + str(eq) + '.h5'
    checkpoint = ModelCheckpoint(ckpt,
                                 monitor='loss',
                                 mode = 'auto',
                                 verbose=0,
                                 save_weights_only=True,
                                 save_best_only=True)

    callbacks0 = [lr_scheduler]
    callbacks = [lr_scheduler, early_stopping_monitor, checkpoint]

    # Print the Model Summary.
    if verbose:
        model.summary()

    # E3: optional random subset of patches for training (paper Sec. 4.3). Default: all patches, unshuffled input as in the original.
    if train_ratio < 1.0:
        rng = np.random.default_rng(seed)
        nsel = int(round(train_ratio * dataInputF.shape[0]))
        idx = np.sort(rng.choice(dataInputF.shape[0], nsel, replace=False))
        fitF2, fit2, fitF = dataInputF2[idx], dataInput2[idx], dataInputF[idx]
    else:
        fitF2, fit2, fitF = dataInputF2, dataInput2, dataInputF

    # Training the DL Model
    t0 = time.time()
    if batch_feed:
        # E3: same data, same per-epoch shuffling and batch size, but batches are sliced from the existing
        # NumPy arrays instead of Keras first copying all patches into TF memory (OOM for SAFOD on Colab).
        history = model.fit(_PatchBatches(fitF2, fit2, fitF, BATCHSIZE, shuffle=True, seed=seed),
                            epochs=EPOCHNO, callbacks = callbacks, verbose = verbose)
    else:
        history = model.fit([fitF2,fit2], [fitF], epochs=EPOCHNO, batch_size=BATCHSIZE, shuffle=True, callbacks = callbacks, verbose = verbose)
    fit_seconds = time.time() - t0
    del fitF2, fit2, fitF

    # Loading the optimal Model.
    model.load_weights(ckpt)
    if batch_feed:
        out = model.predict(_PatchBatches(dataInputF2, dataInput2, None, 256, shuffle=False), verbose=0)
    else:
        out = model.predict([dataInputF2,dataInput2],batch_size=32)
    # Just in Case We Normalize the Band-pass Filtered Data.
    #out = out*ma1
    # Unpatching to Reconstruct the Input Data.
    out = np.reshape(out,(out.shape[0],w1*w2))
    outA = np.transpose(out)
    n1,n2=np.shape(BP)
    outB = patch_inv(outA,n1,n2,w1,w2,s1z,s2z)
    outB = np.array(outB)

    extras = None
    if return_extras:
        # sort Dense layers by creation order (auto names dense, dense_1, dense_2, ...)
        def _order(l):
            s = l.name.rsplit('_', 1)
            return int(s[1]) if len(s) == 2 and s[1].isdigit() else 0
        dense = sorted([l for l in model.layers if isinstance(l, Dense)], key=_order)
        extras = {'history': history.history,
                  'fit_seconds': fit_seconds,
                  'n_patches': int(dataInputF.shape[0]),
                  'n_train_patches': int(round(train_ratio * dataInputF.shape[0])),
                  # first Dense of the band-pass branch, first Dense of the CWT branch, output Dense
                  'dense_weights': {'bp_to_enc1': dense[0].get_weights()[0],
                                    'cwt_to_enc1': dense[2].get_weights()[0],
                                    'dec_to_out': dense[-1].get_weights()[0]}}

    # Saving the Denoised Data.
    #np.save(r'denoised_128_eq_' + str(eq),outB)
    # Release the GPU Memory.
    K.clear_session()

    if return_extras:
        return outB, extras
    return outB
