const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startButton = document.getElementById('start-capture');
const stopButton = document.getElementById('stop-capture');
const predictButton = document.getElementById('predict-gesture');
const result = document.getElementById('prediction-result');

let mediaStream = null;
let frames = [];
let intervalId = null;

// Start capturing video
startButton.addEventListener('click', async () => {
    if (!mediaStream) {
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = mediaStream;
    }
    video.play();
    frames = [];
    intervalId = setInterval(captureFrame, 100); // Capture frames every 100ms
    startButton.disabled = true;
    stopButton.disabled = false;
    predictButton.disabled = true;
});

// Stop capturing video
stopButton.addEventListener('click', () => {
    clearInterval(intervalId);
    startButton.disabled = false;
    stopButton.disabled = true;
    predictButton.disabled = false;
});

// Capture a frame from the video
function captureFrame() {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/png');
    frames.push(imageData); // Store the frame
}

// Send frames for prediction
predictButton.addEventListener('click', async () => {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ frames }),
        });
        const data = await response.json();
        result.textContent = data.prediction;
    } catch (error) {
        console.error('Error during prediction:', error);
        result.textContent = 'Error predicting gesture.';
    }
});

