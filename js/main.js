let model;
let canvas, ctx;
let drawing = false;
let predictTimeout;

// ===== INIT =====
async function init() {
    document.getElementById("result").innerText = "Loading...";

    try {
        // ✅ FIXED MODEL PATH (your current repo)
        model = await tf.loadLayersModel(
            "https://cdn.jsdelivr.net/gh/Akashcc702/Mnist_Model0.1/model/mnist_model.json"
        );

        document.getElementById("result").innerText = "Ready ✅";
        console.log("MODEL LOADED SUCCESS");

    } catch (err) {
        console.error("MODEL LOAD ERROR:", err);
        document.getElementById("result").innerText = "Model Error ❌";
        return;
    }

    canvas = document.getElementById("sketchpad");
    ctx = canvas.getContext("2d");

    // black background
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // mouse
    canvas.addEventListener("mousedown", () => drawing = true);
    canvas.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("mousemove", draw);

    // touch
    canvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        drawing = true;
    });

    canvas.addEventListener("touchend", stopDraw);

    canvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        draw(e);
    });

    document.getElementById("clear_button").addEventListener("click", clearCanvas);
}

// ===== DRAW =====
function stopDraw() {
    drawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!drawing) return;

    let rect = canvas.getBoundingClientRect();
    let x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    let y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;

    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);

    clearTimeout(predictTimeout);
    predictTimeout = setTimeout(predict, 400);
}

// ===== CLEAR =====
function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    document.getElementById("result").innerText = "-";
    document.getElementById("confidence").innerText = "Confidence: -";
}

// ===== PREPROCESS =====
function preprocess(sourceCanvas) {
    let temp = document.createElement("canvas");
    temp.width = 28;
    temp.height = 28;

    let tctx = temp.getContext("2d");
    tctx.drawImage(sourceCanvas, 0, 0, 28, 28);

    let img = tctx.getImageData(0, 0, 28, 28).data;

    let input = [];
    for (let i = 0; i < img.length; i += 4) {
        input.push((255 - img[i]) / 255);
    }

    return tf.tensor(input).reshape([1, 28, 28, 1]);
}

// ===== SINGLE PREDICT =====
function predict() {
    if (!model) return;

    let input = preprocess(canvas);
    let pred = model.predict(input);
    let probs = pred.dataSync();

    let max = Math.max(...probs);
    let result = probs.indexOf(max);

    document.getElementById("result").innerText = result;
    document.getElementById("confidence").innerText =
        "Confidence: " + (max * 100).toFixed(2) + "%";

    tf.dispose([input, pred]);
}

// ===== MULTI DIGIT =====
function predictMultiple() {
    if (!model) return;

    let width = canvas.width;
    let height = canvas.height;

    let imgData = ctx.getImageData(0, 0, width, height).data;

    let digits = [];
    let start = null;

    for (let x = 0; x < width; x++) {
        let hasPixel = false;

        for (let y = 0; y < height; y++) {
            let i = (y * width + x) * 4;
            if (imgData[i] > 50) {
                hasPixel = true;
                break;
            }
        }

        if (hasPixel && start === null) start = x;

        if (!hasPixel && start !== null) {
            let dCanvas = document.createElement("canvas");
            dCanvas.width = x - start;
            dCanvas.height = height;

            dCanvas.getContext("2d").drawImage(
                canvas,
                start,
                0,
                x - start,
                height,
                0,
                0,
                dCanvas.width,
                dCanvas.height
            );

            digits.push(dCanvas);
            start = null;
        }
    }

    let result = "";

    digits.forEach(d => {
        let input = preprocess(d);
        let pred = model.predict(input);
        let probs = pred.dataSync();

        result += probs.indexOf(Math.max(...probs));

        tf.dispose([input, pred]);
    });

    document.getElementById("result").innerText = result || "-";
}

// ===== CAMERA =====
async function startCamera() {
    if (!navigator.mediaDevices) {
        alert("Camera not supported");
        return;
    }

    let video = document.getElementById("camera");

    try {
        let stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        alert("Camera permission denied");
        return;
    }

    setInterval(() => {
        if (!model) return;

        let temp = document.createElement("canvas");
        temp.width = 28;
        temp.height = 28;

        let ctx2 = temp.getContext("2d");
        ctx2.drawImage(video, 0, 0, 28, 28);

        let input = preprocess(temp);
        let pred = model.predict(input);
        let probs = pred.dataSync();

        let result = probs.indexOf(Math.max(...probs));
        document.getElementById("result").innerText = result;

        tf.dispose([input, pred]);

    }, 1000);
}
