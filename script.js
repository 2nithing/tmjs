const imageElements = [];
const labels = [];
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
let mobilenet = undefined;
let model = undefined;
let predict = false;
async function loadMobileNetFeatureModel() {
  // const URL =
  //   "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";

  // mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  mobilenet = await tf.loadGraphModel(
    "https://www.kaggle.com/models/google/mobilenet-v3/TfJs/large-100-224-feature-vector/1",
    { fromTFHub: true }
  );

  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    let answer = mobilenet.predict(
      tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
    );
    console.log(answer.shape);
  });
}

// Call the function immediately to start loading.
loadMobileNetFeatureModel();

function handleImageUpload(event, previewId, classNameInputId) {
  const imagePreview = document.getElementById(previewId);
  const className = document.getElementById(classNameInputId);
  imagePreview.innerHTML = ""; // Clear any existing thumbnails

  const files = Array.from(event.target.files);
  for (const file of files) {
    const img = document.createElement("img");
    const reader = new FileReader();

    reader.onload = function (e) {
      img.src = e.target.result;
      img.className = "thumbnail";
      imagePreview.appendChild(img);

      img.onload = () => {
        imageElements.push(img); // Store the image element
        labels.push(className.getAttribute("data-label")); // Store the corresponding label
      };
    };

    reader.readAsDataURL(file);
  }
}

let trainingStatus = document.getElementById("trainingStatus");

async function Training() {
  predict = false;

  const imageTensors = [];
  for (const img of imageElements) {
    const tensor = await imageToTensor(img);
    imageTensors.push(mobilenet.predict(tensor).squeeze());
  }
  console.log(imageTensors); // Display the array of tensors in the console
  console.log(labels); // Display the array of labels in the console

  model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [1280], units: 128, activation: "relu" })
  );
  model.add(tf.layers.dense({ units: 2, activation: "softmax" }));

  model.summary();

  // Compile the model with the defined optimizer and specify a loss function to use.
  model.compile({
    // Adam changes the learning rate over time which is useful.
    optimizer: "adam",
    // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
    // Else categoricalCrossentropy is used if more than 2 classes.
    loss: "binaryCrossentropy",
    // As this is a classification problem you can record accuracy in the logs too!
    metrics: ["accuracy"],
  });
  tf.util.shuffleCombo(imageTensors, labels);
  let outputsAsTensor = tf.tensor1d(labels, "int32");
  let oneHotOutputs = tf.oneHot(outputsAsTensor, 2);
  console.log(oneHotOutputs.arraySync());
  let inputsAsTensor = tf.stack(imageTensors);

  let results = await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batches: 5,
    epochs: 50,
    callbacks: { onEpochEnd: logProgress, onTrainEnd: trainSuccess },
  });
  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
  predict = true;
}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, logs);
  trainingStatus.innerHTML = `
            <p>Training: ${epoch + 1}</p>
        `;
}
function trainSuccess() {
  trainingStatus.innerHTML = `
            <p>Training Completed</p>
        `;
}

async function imageToTensor(imageElement) {
  return tf.browser
    .fromPixels(imageElement)
    .div(255)
    .resizeBilinear([MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH], true)
    .expandDims();
}

function handleTestImage(event, preview) {
  const uploadedImage = document.getElementById(preview);
  const pred_result = document.getElementById("prediction");
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();

    reader.onload = function (e) {
      uploadedImage.src = e.target.result;
      uploadedImage.style.display = "block";
      uploadedImage.className = "thumbnail";
      uploadedImage.onload = function () {
        let videoFrameAsTensor = tf.browser.fromPixels(uploadedImage).div(255);
        let tensor = tf.image.resizeBilinear(
          videoFrameAsTensor,
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
        );

        console.log(tensor);
        let imageFeatures = mobilenet.predict(tensor.expandDims());
        let prediction = model.predict(imageFeatures).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let pred_class =
          document.querySelectorAll("[data-label]")[highestIndex].value;
        let predictionArray = prediction.arraySync();
        console.log(predictionArray);
        const value = prediction.max().arraySync(); // or tensor.dataSync()[0];

        // Step 2: Convert to floating-point number and format to 4 decimal places
        const formattedValue = (value * 100).toFixed(2);
        pred_result.innerHTML = `
            <p>Predicted Class: ${pred_class}</p>
            <p>Confidence Score: ${formattedValue}%</p>
        `;
      };
    };

    reader.readAsDataURL(file);
  }
  // let x = imageToTensor(uploadedImage);
}

document
  .getElementById("imageUpload1")
  .addEventListener("change", function (event) {
    handleImageUpload(event, "imagePreview1", "className1");
  });

document
  .getElementById("imageUpload2")
  .addEventListener("change", function (event) {
    handleImageUpload(event, "imagePreview2", "className2");
  });

document.getElementById("train").addEventListener("click", () => {
  trainingStatus.innerHTML = "<p>Preparing Training Data</p>";
  Training();
});

document
  .getElementById("testImage")
  .addEventListener("change", function (event) {
    handleTestImage(event, "showTestImage");
  });
