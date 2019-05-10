var tf = require("@tensorflow/tfjs");
const csv = require("csv-parser");
const fs = require("fs");
Canvas = require("canvas");
const IMG_Width = 64;
const IMG_Height = 64;
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
      CLASSES = labels.filter(distinct).sort((a, b) => {
        return a - b;
      });
      NUM_CLASSES = CLASSES.length;
      currentModel = createConvModel();
      currentModel.summary();
      // await currentModel.save('E:/AI_ML_DL/Data/vn_celeb_face_recognition/');
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
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [IMG_Height, IMG_Width, 1],
      kernelSize: 3,
      filters: 16,
      activation: "relu"
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(
    tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(
    tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: "relu" })
  );

  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dropout(0.5));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: "softmax" }));

  return model;
}
function getTrainData() {
  const labels = [];
  var i = 0;
  for (const item of datas) {
    i++;
    const img = getImgAndResize(
      "E:/AI_ML_DL/Data/vn_celeb_face_recognition/train/" + item.image
    );
    if (!xs) {
      xs = tf.keep(img);
    } else {
      const oldX = xs;
      xs = tf.keep(oldX.concat(tf.keep(img), 0));
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
  ctx.drawImage(img, 0, 0);

  let pixels = ctx.getImageData(0, 0, img.width, img.height);
  for (let y = 0; y < pixels.height; y++) {
    for (let x = 0; x < pixels.width; x++) {
      let i = y * 4 * pixels.width + x * 4;
      let avg = (pixels.data[i] + pixels.data[i + 1] + pixels.data[i + 2]) / 3;

      pixels.data[i] = avg;
      pixels.data[i + 1] = avg;
      pixels.data[i + 2] = avg;
    }
  }
  ctx.putImageData(pixels, 0, 0, 0, 0, pixels.width, pixels.height);

  var image = tf.browser.fromPixels(canvas, 1);
  image = tf.image.resizeBilinear(image, [IMG_Height, IMG_Width], false);
  const batchedImage = image.expandDims(0);
  return batchedImage
    .toFloat()
    .div(tf.scalar(127))
    .sub(tf.scalar(1));
}
const distinct = (value, index, self) => {
  return self.indexOf(value) === index;
};

async function train(model) {
  const batchSize = 400;
  const optimizer = tf.train.adam(0.01);
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metric: "accuracy"
  });
  try {
    await model.fit(xs, ys, {
      batchSize,
      epochs: 4,
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
        break;
      }
    }
    result.push(arr);
  }
  return tf.tensor2d(result, [datas.length, NUM_CLASSES]);
}
