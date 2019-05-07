var tf = require("@tensorflow/tfjs");
const csv = require("csv-parser");
const fs = require("fs");
Canvas = require("canvas");
const IMG_Width = 28;
const IMG_Height = 28;
const IMAGE_SIZE = IMG_Height * IMG_Width;
var NUM_CLASSES;
var CLASSES;
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
    .on("end", async () => {
      datas = results;
      CLASSES = labels.filter(distinct);
      NUM_CLASSES = CLASSES.length;
      currentModel = createConvModel();
      // await model.save('file:///E:/AI_ML_DL/Data/vn_celeb_face_recognition/');
      getTrainData();
      console.log("start train");
      await train(currentModel);
      const result = currentModel.predict(
        getImgAndResize(
          "E:/AI_ML_DL/Data/vn_celeb_face_recognition/test/3b74899aa5634d8ab858efec894916da.png"
        )
      );
      const arr = result.arraySync();
      console.log("arr", arr);
      const index = arr[0].indexOf(Math.max(...arr[0]));
      console.log("index", index);
      console.log("result", CLASSES[index]);
      console.log("dcx", arr[0][index]);
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
  ys = labelsToLabelTrain(labels);
  console.log("get train data done");
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

async function train(model) {
  const batchSize = 500;
  const optimizer = tf.train.adam(0.005);
  model.compile({ optimizer: optimizer, loss: "categoricalCrossentropy" });
  try {
    await model.fit(xs, ys, {
      batchSize,
      epochs: 3,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          console.log("Loss: " + logs.loss.toFixed(5));
        }
      }
    });
  } catch (error) {
    console.log(error);
  }
}
function labelsToLabelTrain(labels) {
  const result = [];
  for (const item of labels) {
    let arr = new Array(NUM_CLASSES).fill(0);
    for (let index = 0; index < CLASSES.length; index++) {
      if (item === +CLASSES[index]) {
        arr[index] = 1;
      }
    }
    result.push(arr);
  }
  return tf.tensor2d(result, [datas.length, NUM_CLASSES]);
}
