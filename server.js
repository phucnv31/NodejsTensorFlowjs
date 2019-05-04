var tf = require("@tensorflow/tfjs");
const csv = require("csv-parser");
const fs = require("fs");
Canvas = require("canvas");
const IMG_Width = 64;
const IMG_Height = 64;
const IMAGE_SIZE = IMG_Height * IMG_Width;
var NUM_CLASSES;
var currentModel;
var datas;
var xs;
var ys;
readFileCsv();
// await train(currentModel, () => {
//     console.log('done');
// });

function readFileCsv() {
  const results = [];
  const labels = [];
  fs.createReadStream("E:/AI_ML_DL/Data/vn_celeb_face_recognition/train.csv")
    .pipe(csv())
    .on("data", data => {
      results.push(data);
      labels.push(data.label);
    })
    .on("end", () => {
      datas = results;
      NUM_CLASSES = labels.filter(distinct).length;
      currentModel = createConvModel();
      console.log(currentModel.summary());
      console.log(getTrainData());
      console.log("start train");
      train(currentModel);
    });
}

function createConvModel() {
  // Create a sequential neural network model. tf.sequential provides an API
  // for creating "stacked" models where the output from one layer is used as
  // the input to the next layer.
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [IMG_Height, IMG_Width, 3],
      kernelSize: 3,
      filters: 16,
      activation: "relu"
    })
  );

  // After the first layer we include a MaxPooling layer. This acts as a sort of
  // downsampling using max values in a region instead of averaging.
  // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  // Our third layer is another convolution, this time with 32 filters.
  model.add(
    tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
  );

  // Max pooling again.
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  // Add another conv2d layer.
  model.add(
    tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
  );

  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({ units: 64, activation: "relu" }));

  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: "softmax" }));

  return model;
}

// async function train(model, onIteration) {
//     const optimizer = 'rmsprop';
//     model.compile({
//         optimizer,
//         loss: 'categoricalCrossentropy',
//         metrics: ['accuracy'],
//     });
//     const batchSize = 200;
//     const validationSplit = 0.15;
//     const trainEpochs = 25;
//     let trainBatchCount = 0;

//     const trainData = data.getTrainData();
//     // const testData = data.getTestData();

//     const totalNumBatches = Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) * trainEpochs;
//     let valAcc;
//     await model.fit(trainData.xs, trainData.labels, {
//         batchSize,
//         validationSplit,
//         epochs: trainEpochs,
//         callbacks: {
//             onBatchEnd: async (batch, logs) => {
//                 trainBatchCount++;
//                 console.log(
//                     `Training... (` +
//                     `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
//                     ` complete). To stop training, refresh or close page.`);
//                 // ui.plotLoss(trainBatchCount, logs.loss, 'train');
//                 // ui.plotAccuracy(trainBatchCount, logs.acc, 'train');
//                 if (onIteration && batch % 10 === 0) {
//                     onIteration('onBatchEnd', batch, logs);
//                 }
//                 await tf.nextFrame();
//             },
//             onEpochEnd: async (epoch, logs) => {
//                 valAcc = logs.val_acc;
//                 // ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
//                 // ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
//                 if (onIteration) {
//                     onIteration('onEpochEnd', epoch, logs);
//                 }
//                 await tf.nextFrame();
//             }
//         }
//     });

// const testResult = model.evaluate(testData.xs, testData.labels);
// const testAccPercent = testResult[1].dataSync()[0] * 100;
// const finalValAccPercent = valAcc * 100;
// ui.logStatus(
//     `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
//     `Final test accuracy: ${testAccPercent.toFixed(1)}%`);
// }
function getTrainData() {
  const labels = [];
  var i = 0;
  for (const item of datas) {
    i++;
    if (!xs) {
      // For the first example that gets added, keep example and y so that the
      // ControllerDataset owns the memory of the inputs. This makes sure that
      // if addExample() is called in a tf.tidy(), these Tensors will not get
      // disposed.
      xs = tf.keep(
        getImgAndResize(
          "E:/AI_ML_DL/Data/vn_celeb_face_recognition/train/" + item.image
        )
      );
    } else {
      const oldX = xs;
      xs = tf.keep(
        oldX.concat(
          getImgAndResize(
            "E:/AI_ML_DL/Data/vn_celeb_face_recognition/train/" + item.image
          ),
          0
        )
      );
      oldX.dispose();
    }
    labels.push(+item.label);
    console.log("i:" + i);
  }
  console.log("label start");
  ys = tf.tensor1d(labels);
  console.log("label done");
  return { xs, ys };
}
function getImgAndResize(path) {
  var img = new Canvas.Image(); // Create a new Image
  img.src = path;
  var canvas = Canvas.createCanvas(img.width, img.height);
  var ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, img.width / 4, img.height / 4);
  var image = tf.browser.fromPixels(canvas);
  // Crop the image so we're using the center square of the rectangular
  // webcam.
  image = tf.image.resizeBilinear(image, [IMG_Height, IMG_Width], false);
  // Expand the outer most dimension so we have a batch size of 1.
  const batchedImage = image.expandDims(0);
  // Normalize the image between -1 and 1. The image comes in between 0-255,
  // so we divide by 127 and subtract 1.
  return batchedImage
    .toFloat()
    .div(tf.scalar(127))
    .sub(tf.scalar(1));
}
const distinct = (value, index, self) => {
  return self.indexOf(value) === index;
};

function train(model) {
  const batchSize = 200;
  const optimizer = tf.train.adam(0.0005);
  model.compile({ optimizer: optimizer, loss: "categoricalCrossentropy" });
  model.fit(xs, ys, {
    batchSize,
    epochs: 20,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log("Loss: " + logs.loss.toFixed(5));
      }
    }
  });
}
