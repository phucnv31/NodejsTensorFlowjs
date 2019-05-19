var tf = require("@tensorflow/tfjs");
const readline = require('readline');
require('@tensorflow/tfjs-node')
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});
const csv = require("csv-parser");
const fs = require("fs");
Canvas = require("canvas");
let truncatedMobileNet;
const IMG_Width = 224;
const IMG_Height = 224;
const IMAGE_SIZE = IMG_Height * IMG_Width;
var NUM_CLASSES;
var CLASSES;
var currentModel;
var datas;
var xs;
var ys;
readFileCsv();

function readFileCsv() {
  const results = [];
  const labels = [];
  fs.createReadStream("D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/train.csv")
    .pipe(csv())
    .on("data", data => {
      results.push(data);
      labels.push(data.label);
    })
    .on("end", async () => {
      datas = results;
      CLASSES = labels.filter(distinct);
      NUM_CLASSES = CLASSES.length;
      truncatedMobileNet = await loadTruncatedMobileNet();
      truncatedMobileNet.summary()
      try {
        currentModel = await tf.loadLayersModel("file://D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/model.json");
      } catch (error) {
        currentModel = createConvModel();
        // await currentModel.save('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/');
        getTrainData();
        console.log("start train");
        await train(currentModel);
        console.log("train done");
        currentModel.save('file://D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition');
        // const pre = currentModel.predict(getImgAndResize('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test/0b4c2cf7352f40a3b55339f13c04bcda.png'))
        // const arr = pre.arraySync();
        // console.log('arr', arr);
        // console.log('max score', Math.max(...arr[0]));
        // const index = arr[0].indexOf(Math.max(...arr[0]))
        // console.log('max score', CLASSES[index]);
      }
      // input();
      predictFolder();
    });
}

function input() {
  rl.question('Input path image to predict: ', (answer) => {
    if (answer !== "stop") {
      predict(answer);
    } else {
      rl.close();
    }
  });
}
function moveImage2() {
  fs.readdir("D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test/", function (err, files) {
    //handling error
    if (err) {
      return console.log('Unable to scan directory: ' + err);
    }
    //listing all files using forEach
    files.forEach(function (file) {
      // Do whatever you want to do with the file
      var img = new Canvas.Image(); // Create a new Image
      img.src = "D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test/" + file;
      var canvas = Canvas.createCanvas(img.width, img.height);
      var ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      let pixels = ctx.getImageData(0, 0, img.width, img.height);
      for (let y = 0; y < pixels.height; y++) {
        for (let x = 0; x < pixels.width; x++) {
          let i = (y * 4) * pixels.width + x * 4;
          let avg = (pixels.data[i] + pixels.data[i + 1] + pixels.data[i + 2]) / 3;
          pixels.data[i] = avg;
          pixels.data[i + 1] = avg;
          pixels.data[i + 2] = avg;
        }
      }
      ctx.putImageData(pixels, 0, 0, 0, 0, pixels.width, pixels.height);
      var buf = canvas.toBuffer();
      fs.writeFileSync("D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test_move/" + file, buf);
    });
  });

}

function moveImage(path, label) {
  var img = new Canvas.Image(); // Create a new Image
  img.src = "D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/train/" + path;
  var canvas = Canvas.createCanvas(img.width, img.height);
  var ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  let pixels = ctx.getImageData(0, 0, img.width, img.height);
  for (let y = 0; y < pixels.height; y++) {
    for (let x = 0; x < pixels.width; x++) {
      let i = (y * 4) * pixels.width + x * 4;
      let avg = (pixels.data[i] + pixels.data[i + 1] + pixels.data[i + 2]) / 3;
      pixels.data[i] = avg;
      pixels.data[i + 1] = avg;
      pixels.data[i + 2] = avg;
    }
  }
  ctx.putImageData(pixels, 0, 0, 0, 0, pixels.width, pixels.height);
  var buf = canvas.toBuffer();
  fs.writeFileSync("D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/train_move/" + "Label" + label + "_" + path, buf);
}
function logFile(content) {
  fs.writeFile("log.txt", content, (err) => {
    if (err) console.log(err);
    console.log("Successfully Written to File.");
  });
}

function predict(path) {
  const pre = currentModel.predict(truncatedMobileNet.predict(getImgAndResize('D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test/' + path)))
  const arr = pre.arraySync();
  const maxScore = Math.max(...arr[0]);
  const index = arr[0].indexOf(maxScore)
  return { maxScore: maxScore, class: CLASSES[index] };
}
function predictFolder() {
  let i = 0;
  fs.readdir("D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/test/", function (err, files) {
    //handling error
    if (err) {
      return console.log('Unable to scan directory: ' + err);
    }
    //listing all files using forEach
    let lines = '';
    for (const file of files) {
      if (i++ > 1000) {
        break;
      }
      const pre = predict(file);
      if (pre.maxScore >= 0.5) {
        lines += file + " : " + pre.class + " : " + pre.maxScore + "\n";
      }
    }
    logFile(lines);
  });
}
async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

function createConvModel() {
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten(
        { inputShape: truncatedMobileNet.outputs[0].shape.slice(1) }),
      // Layer 1.
      tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });
  return model;
}

function getTrainData() {
  var i = 0;
  for (const item of datas) {
    i++;
    const y = tf.tidy(
      () => tf.oneHot(tf.tensor1d([+item.label]).toInt(), NUM_CLASSES));
    if (!xs) {
      xs = tf.keep(truncatedMobileNet.predict(getImgAndResize(
        "D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/train/" + item.image
      )));
      ys = tf.keep(y);
    } else {
      const oldX = xs;
      xs = tf.keep(
        oldX.concat(
          truncatedMobileNet.predict(getImgAndResize(
            "D:/LapTrinh/AI/ANN/DataSet/vn_celeb_face_recognition/train/" + item.image
          )),
          0
        )
      );
      const oldY = ys;
      ys = tf.keep(oldY.concat(y, 0));
      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
    console.log("i:" + i);
  }
  ys.print();
  // ys = labelsToTrainLabels(labels)
}
function getImgAndResize(path) {
  var img = new Canvas.Image(); // Create a new Image
  img.src = path;
  var canvas = Canvas.createCanvas(img.width, img.height);
  var ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  var image = tf.browser.fromPixels(canvas);
  image = tf.image.resizeBilinear(image, [IMG_Height, IMG_Width], false);
  const batchedImage = image.expandDims(0);
  return batchedImage
    .toFloat()
    .div(tf.scalar(127))
    .sub(tf.scalar(1));
}

function getImgResizeAndGrayScale(path) {
  var img = new Canvas.Image(); // Create a new Image
  img.src = path;
  var canvas = Canvas.createCanvas(img.width, img.height);
  var ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  let pixels = ctx.getImageData(0, 0, img.width, img.height);
  for (let y = 0; y < pixels.height; y++) {
    for (let x = 0; x < pixels.width; x++) {
      let i = (y * 4) * pixels.width + x * 4;
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
  const batchSize = 200;
  const optimizer = tf.train.adam(0.0001);
  model.compile({ optimizer: optimizer, loss: "categoricalCrossentropy", metrics: ['accuracy'] });
  await model.fit(xs, ys, {
    batchSize,
    epochs: 60,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log("Loss: " + logs.loss.toFixed(5));
        console.log("accuracy: " + logs.acc);
      }
    }
  }).then(info => {
    console.log('Final accuracy', info.history.acc);
  });
}
// function labelsToTrainLabels(labels) {
//   const result = [];
//   for (const item of labels) {
//     const arr = new Array(NUM_CLASSES).fill(0);
//     for (let index = 0; index < arr.length; index++) {
//       if (+CLASSES[index] === item) {
//         arr[index] = 1;
//       }
//     }
//     result.push(arr);
//   }
//   return tf.tensor2d(result, [datas.length, NUM_CLASSES]);
// }
