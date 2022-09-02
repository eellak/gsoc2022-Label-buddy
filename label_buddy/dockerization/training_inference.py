from utils import *

# extract_train_zip()
# extract_val_zip()

print("Getting labels...")
# get_labels()

print("Getting training examples...")
train_examples = get_np_arrays()

print("Partitioning...")
partition = {}
partition['train'] = create_train_partition(train_examples)
partition['validation'] = validation_data()

# Parameters
params = {'dim': (1, ),
          'batch_size': 32,
          'epoch_size': 0,
          'n_classes': 2,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=2500,
    decay_rate=0.84,
    staircase=True)

root_dir = "./Models"
model_name = 'YOHO-1'
model_dir = os.path.join(root_dir, model_name)

try: 
    os.mkdir(model_dir) 
except OSError as error: 
    pass  

print("Defining YOHO...")
model = define_YOHO()

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=my_loss_fn, metrics=[binary_acc])


initial_epoch = 0
p = os.path.join(model_dir, 'custom_params.pickle')
if os.path.isfile(p):
  print("Existing model found. Loading weights and training ...")
  with open(p, 'rb') as f:
    custom_params = pickle.load(f)
    last_epoch = custom_params['last_epoch']
    initial_epoch = last_epoch
  model_path = os.path.join(model_dir, 'model-best.h5')
  print("Model path: " + str(model_path))
  model.load_weights(model_path)

  # model.load_weights(model_path)
  model.fit(training_generator, validation_data=validation_generator, epochs=300, initial_epoch=initial_epoch,
            callbacks=[MyCustomCallback_3(model_dir, patience=15)], verbose=1)
  
else:
  print("No existing model found. Begin training ...")

  model.fit(training_generator, validation_data=validation_generator, epochs=300,
            callbacks=[MyCustomCallback_3(model_dir, patience=15)], verbose=1)