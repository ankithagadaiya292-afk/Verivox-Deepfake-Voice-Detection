let mediaRecorder;
let recordedBlob = null;


// RECORD AUDIO
async function recordAudio() {

    try {

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        mediaRecorder = new MediaRecorder(stream);

        let chunks = [];

        mediaRecorder.ondataavailable = e => {
            chunks.push(e.data);
        };

        mediaRecorder.onstop = () => {

            recordedBlob = new Blob(chunks, { type: "audio/wav" });

            const audioURL = URL.createObjectURL(recordedBlob);

            let audioPlayer = document.getElementById("audioPlayback");

            audioPlayer.src = audioURL;

            audioPlayer.style.display = "block";

            document.getElementById("result").innerText =
                "Recorded audio ready for testing";

        };

        mediaRecorder.start();

        document.getElementById("result").innerText =
            "Recording for 3 seconds...";

        setTimeout(() => {
            mediaRecorder.stop();
        }, 3000);

    } catch (err) {

        alert("Microphone permission required");

        console.error(err);

    }

}


// GET FILE OR RECORDED AUDIO
function getAudioFile() {

    let file = document.getElementById("audioFile").files[0];

    if (file) return file;

    if (recordedBlob) return recordedBlob;

    alert("Upload or record audio first");

    return null;

}


// SVM
function predictSVM() {

    let file = getAudioFile();

    if (!file) return;

    let formData = new FormData();

    formData.append("audio", file);

    fetch("http://127.0.0.1:5000/predict_svm", {

        method: "POST",

        body: formData

    })

        .then(res => res.json())

        .then(data => {

            document.getElementById("result").innerText =
                "SVM Prediction: " + data.prediction +
                " | Confidence: " + data.confidence;

        })

        .catch(err => {

            alert("Backend connection error");

            console.error(err);

        });

}


// CNN
function predictCNN() {

    let file = getAudioFile();

    if (!file) return;

    let formData = new FormData();

    formData.append("audio", file);

    fetch("http://127.0.0.1:5000/predict_cnn", {

        method: "POST",

        body: formData

    })

        .then(res => res.json())

        .then(data => {

            document.getElementById("result").innerText =
                "CNN Prediction: " + data.prediction +
                " | Confidence: " + data.confidence;

        })

        .catch(err => {

            alert("CNN backend error");

            console.error(err);

        });

}