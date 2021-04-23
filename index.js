import * as tf from '@tensorflow/tfjs';

document.getElementById('root').innerHTML = "tenserflow js here bro!"

//tenserflow model
// Define a machine learning model for linear regression
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Specify loss and optimizer for model
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Prepare training data
const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

// Train the model
model.fit(xs, ys, {epochs: 500}).then(() => {
});

// Train the model
model.fit(xs, ys, {epochs: 500}).then(() => {
    // Use model to predict values
    model.predict(tf.tensor2d([5], [1,1])).print();
});